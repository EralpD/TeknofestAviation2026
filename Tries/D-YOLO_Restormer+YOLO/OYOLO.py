import ultralytics
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from T3Dmap import DroneTrajectory3D

FPS = 20

def read_folder(folder_path):
    frames = []
    for frame in sorted(os.listdir(folder_path)):
        frames.append(os.path.join(folder_path, frame))
    return frames

FRAME_PATH = '../Data/Oturum1'
OUTPUT_PATH = '../Data/Oturum1_Tracked/tracked_output.mp4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO("yolov8l.pt")
model.to(device)
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1865, 767))

prev_gray = None 

def estimate_motion(gray):
    global prev_gray
    dx, dy = 0.0, 0.0
    good_prev, good_next = [], []

    if prev_gray is not None:
        feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7)
        prev_points = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

        if prev_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
            good_prev = prev_points[status == 1]
            good_next = next_points[status == 1]

            if len(good_prev) > 0:
                motion = good_next - good_prev
                dx = float(np.mean(motion[:, 0]))
                dy = float(np.mean(motion[:, 1]))

    prev_gray = gray
    return dx, dy, good_prev, good_next

def draw_motion(annotated_frame, good_prev, good_next, dx, dy):
    # Optical flow
    for prev, next in zip(good_prev, good_next):
        x1, y1 = map(int, prev)
        x2, y2 = map(int, next)
        cv2.arrowedLine(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1, tipLength=0.3)
        cv2.circle(annotated_frame, (x1, y1), 2, (0, 0, 255), -1)

    h, w = annotated_frame.shape[:2]
    cx, cy = w // 2, h // 2
    ex = int(cx + dx * 5)
    ey = int(cy + dy * 5)
    cv2.arrowedLine(annotated_frame, (cx, cy), (ex, ey), (0, 0, 255), 3, tipLength=0.3)

    # dx / dy information
    cv2.putText(annotated_frame, f"dx: {dx:.1f}px  dy: {dy:.1f}px",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return annotated_frame

trajectory = DroneTrajectory3D(canvas_w=500, canvas_h=767)

def compensate_ego_motion(prev_gray, curr_gray):
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7)
    prev_points = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

    if prev_points is None:
        return None, 0.0, 0.0

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)

    good_prev = prev_points[status == 1]
    good_next = next_points[status == 1]

    if len(good_prev) < 4:  # Affine için en az 4 nokta gerekli
        return None, 0.0, 0.0

    transform, _ = cv2.estimateAffinePartial2D(good_prev, good_next)

    if transform is None:
        return None, 0.0, 0.0

    dx = float(transform[0, 2])  # Drone'un x hareketi
    dy = float(transform[1, 2])  # Drone'un y hareketi

    # Stabilize edilmiş görüntü
    h, w = curr_gray.shape
    stabilized = cv2.warpAffine(curr_gray, transform, (w, h),
                                flags=cv2.WARP_INVERSE_MAP)

    return stabilized, dx, dy

def track(frame_path, elapsed, fps, frame_idx):
    global prev_gray

    img = cv2.imread(frame_path)

    result = model.track(
        img, persist=True,
        device=device,
        conf=0.3,
        iou=0.5,
        classes=list(range(9)),
        tracker="bytetrack.yaml",
        verbose=False)

    annotated_frame = result[0].plot()
    del result
    annotated_frame = cv2.resize(annotated_frame, (1365, 767))

    # Optical flow + ego-motion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (1365, 767))

    dx, dy = 0, 0
    if prev_gray is not None:
        _, dx, dy = compensate_ego_motion(prev_gray, gray)
    prev_gray = gray

    trajectory.update(dx, dy)

    if frame_idx % 3 == 0 or trajectory.last_canvas is None:
        traj_canvas = trajectory.draw_canvas()
    else:
        traj_canvas = trajectory.last_canvas

    # Başlık bandı
    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(annotated_frame, "Oturum1 - YOLOv8",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(annotated_frame, f"Elapsed: {mins:02d}:{secs:02d}  FPS: {fps}",
                (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"dx: {dx:.1f}  dy: {dy:.1f}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    combined = np.hstack([annotated_frame, traj_canvas])

    torch.cuda.empty_cache()
    out.write(combined)

if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))

    start_time = time.time()
    frames = read_folder(FRAME_PATH)

    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        elapsed = time.time() - start_time
        track(frame, elapsed, FPS, i)

    out.release()