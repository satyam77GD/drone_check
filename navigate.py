# Aerius challenge - autonomous drone navigation
# ArduPilot SITL + Gazebo, 16 beam lidar (ROS2)
# run: python3 navigate.py

import math
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2

# ---------- settings (change before run only) ----------
MAV_URL = "udp:127.0.0.1:14550"   # try 14551 or tcp:127.0.0.1:5760 if no heartbeat

GOAL_NE = (30.0, 0.0)             # goal (north, east) in meters from start
GOAL_GPS = None                   # (lat, lon) if goal is given in gps

LIDAR_TOPIC = None                # None = find it automatically
ALT = 3.0                         # cruise altitude (m)
MAX_SPEED = 2.0                   # m/s, increase after testing
GOAL_RADIUS = 1.0                 # m
TIMEOUT = 240                     # seconds

DRONE_R = 0.7                     # drone radius + some margin
LOOK = 8.0                        # ignore stuff further than this
SLOW_DIST = 1.8                   # go slow if obstacle closer than this
PANIC_DIST = 1.0                  # back off if something this close
Z_BAND = 1.0                      # for 3d lidar, only use points near drone height
MIN_R = 0.35                      # ignore very close points (drone body)
SECTORS = 72                      # 5 deg each
YAW_RATE = math.radians(60)
KEEP_DIR = 0.5                    # how much to stick to previous direction
# --------------------------------------------------------

DT = 0.1


def wrap(a):
    # wrap angle to -pi..pi
    return (a + math.pi) % (2 * math.pi) - math.pi


class Lidar(Node):
    def __init__(self):
        super().__init__("aerius_lidar")
        self.lock = threading.Lock()
        self.theta = np.array([])
        self.r = np.array([])
        self.last = 0.0

    def find_topic(self):
        for _ in range(40):
            for name, types in self.get_topic_names_and_types():
                if LIDAR_TOPIC and name != LIDAR_TOPIC:
                    continue
                if "sensor_msgs/msg/LaserScan" in types:
                    self.create_subscription(LaserScan, name, self.on_scan, qos_profile_sensor_data)
                    print("lidar (LaserScan):", name)
                    return True
                if "sensor_msgs/msg/PointCloud2" in types:
                    self.create_subscription(PointCloud2, name, self.on_cloud, qos_profile_sensor_data)
                    print("lidar (PointCloud2):", name)
                    return True
            time.sleep(1)
        return False

    def on_scan(self, msg):
        r = np.array(msg.ranges, dtype=float)
        th = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > MIN_R) & (r < LOOK)
        with self.lock:
            self.theta = th[ok]
            self.r = r[ok]
            self.last = time.time()

    def on_cloud(self, msg):
        pts = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if pts.dtype.names:
            p = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(float)
        else:
            p = np.array(list(pts), dtype=float).reshape(-1, 3)
        if len(p) == 0:
            with self.lock:
                self.theta = np.array([])
                self.r = np.array([])
                self.last = time.time()
            return
        r = np.hypot(p[:, 0], p[:, 1])
        ok = (np.abs(p[:, 2]) < Z_BAND) & (r > MIN_R) & (r < LOOK)
        with self.lock:
            self.theta = np.arctan2(p[ok, 1], p[ok, 0])
            self.r = r[ok]
            self.last = time.time()

    def get(self):
        with self.lock:
            return self.theta.copy(), self.r.copy(), self.last


class Drone:
    def __init__(self):
        print("connecting to", MAV_URL)
        self.m = mavutil.mavlink_connection(MAV_URL)
        self.m.wait_heartbeat()
        print("got heartbeat")
        self.m.mav.request_data_stream_send(self.m.target_system, self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1)
        self.n = 0.0
        self.e = 0.0
        self.d = 0.0
        self.yaw = 0.0
        self.lat = None
        self.lon = None
        self.got_pos = False

    def update(self):
        while True:
            msg = self.m.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE", "GLOBAL_POSITION_INT"],
                                    blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.n, self.e, self.d = msg.x, msg.y, msg.z
                self.got_pos = True
            elif t == "ATTITUDE":
                self.yaw = msg.yaw
            else:
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7

    def alt(self):
        return -self.d

    def mode(self, name):
        self.m.set_mode_apm(name)
        time.sleep(0.5)

    def takeoff(self, h):
        start = time.time()
        while time.time() - start < 120:
            self.mode("GUIDED")
            self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                         mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                         0, 1, 0, 0, 0, 0, 0, 0)
            time.sleep(2)
            self.update()
            hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
            if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                break
            print("waiting to arm...")
        else:
            raise RuntimeError("arming failed")

        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, h)
        start = time.time()
        while time.time() - start < 40:
            self.update()
            if self.alt() > h * 0.92:
                break
            time.sleep(0.1)
        print("takeoff done, alt =", round(self.alt(), 1))

    def vel(self, vn, ve, vd, yaw):
        # velocity + yaw setpoint
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000101111000111,
            0, 0, 0, vn, ve, vd, 0, 0, 0, yaw, 0)


def pick_direction(theta, r, goal_th, prev_th, side):
    # returns chosen direction (body frame), free distance in that direction,
    # and direction/distance of the closest obstacle
    sec = np.linspace(-math.pi, math.pi, SECTORS, endpoint=False)

    if len(r) == 0:
        blocked = np.zeros(SECTORS, dtype=bool)
        free = np.full(SECTORS, LOOK)
        near_th, near_r = 0.0, LOOK
    else:
        # how wide each obstacle looks once we add the drone radius
        half = np.where(r > DRONE_R, np.arcsin(np.clip(DRONE_R / np.maximum(r, 1e-6), 0, 1)), math.pi / 2)
        diff = np.abs((sec[:, None] - theta[None, :] + math.pi) % (2 * math.pi) - math.pi)
        hit = diff < half[None, :]
        blocked = hit.any(axis=1)
        free = np.where(hit, r[None, :], np.inf).min(axis=1)
        free[np.isinf(free)] = LOOK
        i = np.argmin(r)
        near_th, near_r = float(theta[i]), float(r[i])

    to_goal = np.abs((sec - goal_th + math.pi) % (2 * math.pi) - math.pi)
    to_prev = np.abs((sec - prev_th + math.pi) % (2 * math.pi) - math.pi)
    cost = to_goal + KEEP_DIR * to_prev + side * 0.15 * np.sign(sec) * (to_goal > 0.3)
    cost = cost + np.where(blocked, 100.0, 0.0) - 0.05 * free

    k = int(np.argmin(cost))
    if blocked.all():
        k = int(np.argmax(free))
    return float(sec[k]), float(free[k]), near_th, near_r


def main():
    rclpy.init()
    lidar = Lidar()
    threading.Thread(target=rclpy.spin, args=(lidar,), daemon=True).start()
    if not lidar.find_topic():
        print("no lidar topic found, set LIDAR_TOPIC")
        sys.exit(1)

    drone = Drone()
    while not drone.got_pos:
        drone.update()
        time.sleep(0.1)

    drone.takeoff(ALT)
    drone.update()

    # goal in local NED
    if GOAL_GPS is not None and drone.lat is not None:
        gn = (GOAL_GPS[0] - drone.lat) * 111319.5
        ge = (GOAL_GPS[1] - drone.lon) * 111319.5 * math.cos(math.radians(drone.lat))
        goal_n, goal_e = drone.n + gn, drone.e + ge
    else:
        goal_n, goal_e = drone.n + GOAL_NE[0], drone.e + GOAL_NE[1]
    print("goal:", round(goal_n, 1), round(goal_e, 1))

    t0 = time.time()
    yaw_cmd = drone.yaw
    prev_th = 0.0
    side = 1.0
    ref_dist = math.hypot(goal_n - drone.n, goal_e - drone.e)
    ref_time = time.time()
    reached = False

    while time.time() - t0 < TIMEOUT:
        loop_start = time.time()
        drone.update()
        dn = goal_n - drone.n
        de = goal_e - drone.e
        dist = math.hypot(dn, de)
        if dist < GOAL_RADIUS:
            reached = True
            break

        theta, r, last = lidar.get()

        # hold altitude (NED so up is negative)
        vd = -max(-1.0, min(1.0, ALT - drone.alt()))

        if time.time() - last > 1.0:
            # no fresh lidar data, just hover
            drone.vel(0, 0, vd, yaw_cmd)
            time.sleep(DT)
            continue

        goal_bearing = math.atan2(de, dn)
        goal_th = wrap(drone.yaw - goal_bearing)
        th, free, near_th, near_r = pick_direction(theta, r, goal_th, prev_th, side)
        prev_th = th
        bearing = drone.yaw - th

        if near_r < PANIC_DIST:
            # too close, move directly away
            bearing = drone.yaw - (near_th + math.pi)
            speed = 0.7
        else:
            speed = MAX_SPEED * min(1.0, max(0.2, (free - SLOW_DIST) / (LOOK - SLOW_DIST)))
            speed = min(speed, dist)
            # slow down while still turning
            turn = min(abs(wrap(drone.yaw - bearing)), math.radians(80))
            speed *= max(0.25, math.cos(turn))

        # turn slowly towards where we want to go
        err = wrap(bearing - yaw_cmd)
        yaw_cmd += max(-YAW_RATE * DT, min(YAW_RATE * DT, err))

        drone.vel(speed * math.cos(bearing), speed * math.sin(bearing), vd, yaw_cmd)

        # if no progress for 15s, try the other side
        if time.time() - ref_time > 15:
            if ref_dist - dist < 1.5:
                side = -side
                print("stuck, switching side")
            ref_dist = dist
            ref_time = time.time()

        time.sleep(max(0.0, DT - (time.time() - loop_start)))

    if reached:
        print("goal reached in", round(time.time() - t0, 1), "s")
    else:
        print("timeout")

    for _ in range(10):
        drone.vel(0, 0, 0, yaw_cmd)
        time.sleep(0.1)
    drone.mode("LAND")
    print("landing")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
