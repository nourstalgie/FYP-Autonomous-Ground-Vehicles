#!/usr/bin/env python3
"""
ZED RGBD + YOLOv8 + Kalman tracking with ego velocity compensation
Focus: Perfect tracking and velocity estimation
"""
import time
import numpy as np
from dataclasses import dataclass
import pyzed.sl as sl
import cv2
from ultralytics import YOLO

# Kalman Filter for 3D tracking with velocity
class KF3D_CV:
    def __init__(self, x0):
        self.x = np.array(x0, dtype=np.float32).reshape(6, 1)
        self.P = np.eye(6, dtype=np.float32) * 1.0
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.R = np.eye(3, dtype=np.float32) * 0.10  # measurement noise
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

# Velocity smoother
class KFVel3D:
    def __init__(self, v0=(0, 0, 0)):
        self.x = np.array(v0, dtype=np.float32).reshape(3, 1)
        self.P = np.eye(3, dtype=np.float32) * 1.0
        self.R = np.eye(3, dtype=np.float32) * 0.50
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
    kf1: KF3D_CV  # Position and camera-frame velocity
    kf2: KFVel3D  # Smoothed absolute velocity
    label: str
    conf: float
    last_bbox: tuple = None
    hits: int = 0
    missed: int = 0

def l2(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def greedy_match(tracks, det_xyz, max_dist=2.0):
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

def get_depth_at_bbox(depth_map, bbox):
    """Get median depth in bounding box"""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
    
    roi = depth_map[y1:y2, x1:x2]
    valid_depths = roi[np.isfinite(roi) & (roi > 0)]
    
    if len(valid_depths) == 0:
        return None
    
    return np.median(valid_depths)

def main():
    # Initialize ZED
    print("Initializing ZED camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Best quality depth
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init_params.depth_minimum_distance = 0.3  # 30cm minimum
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return
    
    # Enable positional tracking for ego velocity
    print("Enabling positional tracking...")
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Warning: Positional tracking failed: {err}")
    
    # Load YOLO model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Fast nano model
    
    # ZED runtime
    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    pose = sl.Pose()
    
    # Tracking state
    tracks = []
    next_id = 1
    last_ts = time.time()
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    print("=" * 60)
    print("TRACKING STARTED")
    print("Press 'q' to quit")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        
        now = time.time()
        dt = now - last_ts
        last_ts = now
        frame_count += 1
        
        # Get RGB and Depth
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
        
        # Convert to numpy
        rgb_frame = image_zed.get_data()[:, :, :3]  # RGB
        depth_map = depth_zed.get_data()  # Depth in meters
        
        # Initialize video writer
        if out is None:
            h, w = rgb_frame.shape[:2]
            out = cv2.VideoWriter('tracked_output.mp4', fourcc, 30.0, (w, h))
        
        # Get ego velocity (camera motion in world frame)
        #st = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        #v_ego = None
        #if st == sl.POSITIONAL_TRACKING_STATE.OK:
        #    v = pose.get_linear_velocity()
        #    vel_data = v.get()
        #    v_ego = np.array([vel_data[0], vel_data[1], vel_data[2]], dtype=np.float32)
        
        # Run YOLO detection
        results = model(rgb_frame, conf=0.4, verbose=False)
        
        det_xyz = []
        det_info = []
        
        # Get camera calibration
        calibration = zed.get_camera_information().camera_configuration.calibration_parameters
        fx = calibration.left_cam.fx
        fy = calibration.left_cam.fy
        cx = calibration.left_cam.cx
        cy = calibration.left_cam.cy
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                # Get depth at bbox center
                cx_box = int((x1 + x2) / 2)
                cy_box = int((y1 + y2) / 2)
                
                # Get median depth in bbox for robustness
                depth = get_depth_at_bbox(depth_map, bbox)
                
                if depth is None or not np.isfinite(depth) or depth <= 0:
                    continue
                
                # Convert to 3D using camera intrinsics
                X = (cx_box - cx) * depth / fx
                Y = (cy_box - cy) * depth / fy
                Z = depth
                
                xyz = np.array([X, Y, Z], dtype=np.float32)
                
                det_xyz.append(xyz)
                det_info.append((label, conf, bbox))
        
        # Predict all tracks
        for trk in tracks:
            trk.kf1.predict(dt)
            trk.kf2.predict(dt)
        
        # Associate detections to tracks
        matches, un_t, un_d = greedy_match(tracks, det_xyz, max_dist=2.0)
        
        # Update matched tracks
        for ti, di in matches:
            label, conf, bbox = det_info[di]
            trk = tracks[ti]
            
            # Update position tracker
            trk.kf1.update(det_xyz[di])
            
            # Get camera-frame velocity
            v_rel = trk.kf1.vel
            
            # Compute absolute velocity (compensate for ego motion)
            #scale_factor = 0.5  # Adjust this factor to control the impact of ego velocity
            #v_abs_meas = v_rel + (v_ego * scale_factor) if v_ego is not None else v_rel
            v_abs_meas = v_rel  # Just use the relative velocity for now (no ego velocity compensation)


            # Update velocity smoother
            trk.kf2.update(v_abs_meas)
            
            trk.label = label
            trk.conf = conf
            trk.last_bbox = bbox
            trk.hits += 1
            trk.missed = 0
        
        # Handle missed tracks
        for ti in un_t:
            tracks[ti].missed += 1
        
        # Create new tracks
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
                missed=0
            )
            tracks.append(trk)
            next_id += 1
        
        # Prune old tracks
        tracks = [t for t in tracks if t.missed <= 15]
        
        # Visualization
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw confirmed tracks only
        for trk in tracks:
            if trk.hits < 3 or trk.last_bbox is None:  # Need 3 hits to confirm
                continue
            
            x1, y1, x2, y2 = trk.last_bbox
            
            # Get smoothed absolute velocity
            v_abs = trk.kf2.v
            speed = float(np.linalg.norm(v_abs))
            
            # Position
            pos = trk.kf1.pos
            
            # Display info
            text = f"ID:{trk.track_id} {trk.label}"
            text2 = f"Pos:({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})m"
            text3 = f"V_abs:({v_abs[0]:.2f},{v_abs[1]:.2f},{v_abs[2]:.2f})m/s"
            text4 = f"Speed:{speed:.2f}m/s"
            
            # Draw bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text
            y_offset = y1 - 10
            cv2.putText(frame_bgr, text, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_bgr, text2, (x1, y_offset - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame_bgr, text3, (x1, y_offset - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame_bgr, text4, (x1, y_offset - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Console output every 30 frames
            if frame_count % 30 == 0:
                print(f"Track {trk.track_id:3d} | {trk.label:10s} | "
                      f"Pos: [{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}]m | "
                      f"V_abs: [{v_abs[0]:5.2f}, {v_abs[1]:5.2f}, {v_abs[2]:5.2f}]m/s | "
                      f"Speed: {speed:5.2f}m/s")
        
        # Status overlay
        #if v_ego is not None:
        #    ego_speed = float(np.linalg.norm(v_ego))
        #    status = f"Ego V: ({v_ego[0]:.2f}, {v_ego[1]:.2f}, {v_ego[2]:.2f}) m/s | Speed: {ego_speed:.2f} m/s"
        #else:
        #    status = "Ego velocity: NOT AVAILABLE"
        status = "Ego velocity: NOT AVAILABLE"  # Use this instead
        
        cv2.putText(frame_bgr, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame_bgr, f"Tracks: {len([t for t in tracks if t.hits >= 3])}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Display and save
        cv2.imshow('ZED Tracking', frame_bgr)
        out.write(frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    zed.close()
    out.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("TRACKING STOPPED")
    print("Output saved to: tracked_output.mp4")
    print("=" * 60)

if __name__ == "__main__":
    main()
