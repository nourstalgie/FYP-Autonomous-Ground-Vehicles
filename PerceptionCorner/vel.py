#!/usr/bin/env python3
"""
Real-time display + simultaneous MP4 recording:
ZED + DetectNet + KF1 tracking + KF2 abs-velocity smoothing + on-frame overlay.
No ROS2.
"""
import time
import numpy as np
from dataclasses import dataclass
import pyzed.sl as sl
import jetson.inference
import jetson.utils

# ----------------------------
# KF1: 3D constant-velocity tracker
# State: [x,y,z,vx,vy,vz] (camera frame)
# Meas: [x,y,z]
# ----------------------------
class KF3D_CV:
    def __init__(self, x0):
        self.x = np.array(x0, dtype=np.float32).reshape(6, 1)
        self.P = np.eye(6, dtype=np.float32) * 1.0
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.R = np.eye(3, dtype=np.float32) * 0.10  # meas noise (m^2), tune
        self.q_pos = 0.05
        self.q_vel = 0.50

    def predict(self, dt):
        dt = float(max(1e-3, dt))
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        Q = np.eye(6, dtype=np.float32)
        Q[0, 0] *= self.q_pos
        Q[1, 1] *= self.q_pos
        Q[2, 2] *= self.q_pos
        Q[3, 3] *= self.q_vel
        Q[4, 4] *= self.q_vel
        Q[5, 5] *= self.q_vel
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_xyz):
        z = np.array(z_xyz, dtype=np.float32).reshape(3, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float32) - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[0:3, 0].copy()

    @property
    def vel(self):
        return self.x[3:6, 0].copy()


# ----------------------------
# KF2: velocity smoother
# State: [vx,vy,vz]
# Meas: [vx,vy,vz]
# ----------------------------
class KFVel3D:
    def __init__(self, v0=(0, 0, 0)):
        self.x = np.array(v0, dtype=np.float32).reshape(3, 1)
        self.P = np.eye(3, dtype=np.float32) * 1.0
        self.R = np.eye(3, dtype=np.float32) * 0.50  # (m/s)^2, tune
        self.Q = np.eye(3, dtype=np.float32) * 0.10

    def predict(self, dt):
        self.P = self.P + self.Q

    def update(self, v_meas):
        z = np.array(v_meas, dtype=np.float32).reshape(3, 1)
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3, dtype=np.float32) - K) @ self.P

    @property
    def v(self):
        return self.x[:, 0].copy()


@dataclass
class Track:
    track_id: int
    kf1: KF3D_CV
    kf2: KFVel3D
    label: str
    conf: float
    last_bbox: tuple = None
    hits: int = 0
    missed: int = 0


def l2(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def greedy_match(tracks, det_xyz, max_dist=1.8):
    pairs = []
    for ti, trk in enumerate(tracks):
        for di, xyz in enumerate(det_xyz):
            dist = l2(trk.kf1.pos, xyz)
            if dist <= max_dist:
                pairs.append((dist, ti, di))
    pairs.sort(key=lambda x: x[0])
    used_t, used_d, matches = set(), set(), []
    for dist, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))
    un_t = [i for i in range(len(tracks)) if i not in used_t]
    un_d = [i for i in range(len(det_xyz)) if i not in used_d]
    return matches, un_t, un_d


def zed_open():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {err}")
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("WARNING: positional tracking not enabled, ego vel may be None:", err)
    return zed


def get_ego_velocity_world(zed, pose_obj):
    st = zed.get_position(pose_obj, sl.REFERENCE_FRAME.WORLD)
    if st != sl.POSITIONAL_TRACKING_STATE.OK:
        return None
    v = pose_obj.get_velocity()
    vel_data = v.get()
    return np.array([vel_data[0], vel_data[1], vel_data[2]], dtype=np.float32)


def xyz_from_point_cloud(pc_mat, x, y):
    err, p = pc_mat.get_value(x, y)
    if err != sl.ERROR_CODE.SUCCESS:
        return None
    if len(p) >= 3:
        X, Y, Z = p[0], p[1], p[2]
    else:
        return None
    if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
        return None
    return np.array([X, Y, Z], dtype=np.float32)


def main():
    # ---- ZED ----
    zed = zed_open()
    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()
    pc_zed = sl.Mat()
    pose = sl.Pose()

    # ---- DetectNet ----
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    # ---- Outputs ----
    # Real-time window:
    disp = jetson.utils.videoOutput("display://0")
    # Simultaneous recording (same frames):
    rec = jetson.utils.videoOutput("file://tracked_output.mp4")
    font = jetson.utils.cudaFont()

    # ---- Tracking ----
    tracks = []
    next_id = 1
    assoc_max_dist = 1.8
    max_missed = 15
    min_confirm_hits = 2
    last_ts = time.time()

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        now = time.time()
        dt = now - last_ts
        last_ts = now

        # Capture
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(pc_zed, sl.MEASURE.XYZRGBA)

        # Ego velocity (optional)
        v_ego = get_ego_velocity_world(zed, pose)  # may be None

        # To CUDA
        frame = image_zed.get_data()  # RGBA
        cuda_img = jetson.utils.cudaFromNumpy(frame)

        # Detect
        detections = net.Detect(cuda_img)
        det_xyz = []
        det_info = []  # (label, conf, bbox)

        for d in detections:
            cx = int((d.Left + d.Right) * 0.5)
            cy = int((d.Top + d.Bottom) * 0.5)
            xyz = xyz_from_point_cloud(pc_zed, cx, cy)
            if xyz is None:
                continue
            label = net.GetClassDesc(d.ClassID)
            bbox = (int(d.Left), int(d.Top), int(d.Right), int(d.Bottom))
            det_xyz.append(xyz)
            det_info.append((label, float(d.Confidence), bbox))

        # Predict
        for trk in tracks:
            trk.kf1.predict(dt)
            trk.kf2.predict(dt)

        # Associate
        matches, un_t, un_d = greedy_match(tracks, det_xyz, max_dist=assoc_max_dist)

        # Update matched
        for ti, di in matches:
            label, conf, bbox = det_info[di]
            trk = tracks[ti]
            trk.kf1.update(det_xyz[di])  # KF1 update position
            v_rel = trk.kf1.vel  # relative vel (camera frame)

            # KF2 absolute velocity measurement (baseline)
            v_abs_meas = v_rel + v_ego if v_ego is not None else v_rel
            trk.kf2.update(v_abs_meas)

            trk.label = label
            trk.conf = conf
            trk.last_bbox = bbox
            trk.hits += 1
            trk.missed = 0

        # Missed
        for ti in un_t:
            tracks[ti].missed += 1

        # New tracks
        for di in un_d:
            label, conf, bbox = det_info[di]
            x0 = [det_xyz[di][0], det_xyz[di][1], det_xyz[di][2], 0.0, 0.0, 0.0]
            trk = Track(
                track_id=next_id,
                kf1=KF3D_CV(x0),
                kf2=KFVel3D((0.0, 0.0, 0.0)),
                label=label,
                conf=conf,
                last_bbox=bbox,
                hits=1,
                missed=0,
            )
            tracks.append(trk)
            next_id += 1

        # Prune
        tracks = [t for t in tracks if t.missed <= max_missed]

        # Draw boxes + overlay
        for (label, conf, bbox) in det_info:
            L, T, R, B = bbox
            jetson.utils.cudaDrawRect(cuda_img, (L, T, R, B), (255, 0, 0, 255))

        for trk in tracks:
            if trk.hits < min_confirm_hits or trk.last_bbox is None:
                continue
            L, T, R, B = trk.last_bbox
            v_abs = trk.kf2.v
            speed = float(np.linalg.norm(v_abs))
            text = (
                f"{trk.label} ID:{trk.track_id} "
                f"v_abs=({v_abs[0]:.2f},{v_abs[1]:.2f},{v_abs[2]:.2f}) m/s |v|={speed:.2f}"
            )
            font.OverlayText(cuda_img, cuda_img.width, cuda_img.height,
                             text, max(0, L), max(0, T - 18),
                             font.White, font.Gray40)

        # ---- Outputs (REAL-TIME + RECORDING) ----
        disp.Render(cuda_img)  # real-time stream on screen
        rec.Render(cuda_img)  # same frames recorded to MP4
        disp.SetStatus("Real-time detection + tracking overlay (recording MP4)")
        rec.SetStatus("Recording tracked_output.mp4")

        if not disp.IsStreaming() or not rec.IsStreaming():
            break

    zed.close()


if __name__ == "__main__":
    main()