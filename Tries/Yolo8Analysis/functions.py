import ultralytics
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import time
from tqdm import tqdm
# from T3Dmap import DroneTrajectory3D

class ModelFunctions:
    def __init__(self, fps, frame_path, output_path, device, model, out, width=1365, height=767):
        self.FPS = fps
        self.FRAME_PATH = frame_path
        self.OUTPUT_PATH = output_path
        if not os.path.exists(os.path.dirname(self.OUTPUT_PATH)):
            os.makedirs(os.path.dirname(self.OUTPUT_PATH))
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.out = out
        self.prev_gray = None

    def read_folder(self, folder_path):
        frames = []
        for frame in sorted(os.listdir(folder_path)):
            frames.append(os.path.join(folder_path, frame))
        return frames
    def estimate_motion(self, gray):
        dx, dy = 0.0, 0.0
        good_prev, good_next = [], []

        if self.prev_gray is not None:
            feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7)
            prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **feature_params)

            if prev_points is not None:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_points, None)
                good_prev = prev_points[status == 1]
                good_next = next_points[status == 1]

                if len(good_prev) > 0:
                    motion = good_next - good_prev
                    dx = float(np.mean(motion[:, 0]))
                    dy = float(np.mean(motion[:, 1]))

        self.prev_gray = gray
        return dx, dy, good_prev, good_next

    def draw_motion(self, annotated_frame, good_prev, good_next, dx, dy):
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

    def compensate_ego_motion(self, prev_gray, curr_gray):
        feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7)
        # Drone'un hareketini tahmin etmek için özellik noktalarını tespit et
        prev_points = cv2.goodFeaturesToTrack(prev_gray, **feature_params) 

        if prev_points is None:
            return None, 0.0, 0.0

        # Optik akış hesapla ve hareketi tahmin et
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

    def get_motion_score(self, flow_up, x1, y1, x2, y2):
        if flow_up is None:
            return 0.0
            
        _, _, h, w = flow_up.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        flow_crop = flow_up[0, :, y1:y2, x1:x2]
        
        if flow_crop.numel() == 0: 
            return 0.0
            
        mag = torch.sqrt(flow_crop[0]**2 + flow_crop[1]**2)
        return torch.mean(mag).item()

    def track(self, frame_path, elapsed, fps, frame_idx, flow_up=None):
        img = cv2.imread(frame_path)
        # trajectory = DroneTrajectory3D(canvas_w=500, canvas_h=767)

        result = self.model.track(
            img, persist=True,
            device=self.device,
            conf=0.3,
            iou=0.5,
            tracker="botsort.yaml",
            verbose=False)

        annotated_frame = result[0].plot()

        if flow_up is not None and result[0].boxes is not None:
            _, _, fh, fw = flow_up.shape
            for box in result[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(fw, x2), min(fh, y2)
                
                flow_crop = flow_up[0, :, cy1:cy2, cx1:cx2]
                motion_score = 0.0
                
                if flow_crop.numel() > 0: # Returns number of elements in the tensor
                    mag = torch.sqrt(flow_crop[0]**2 + flow_crop[1]**2)
                    motion_score = torch.mean(mag).item()
                
                is_moving = motion_score > 2.5
                label = f"{'Moving' if is_moving else 'Static'} ({motion_score:.1f})"
                color = (0, 255, 0) if is_moving else (0, 0, 255)
                
                cv2.putText(annotated_frame, label, (x1, max(y1-20, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1., color, 3)

        del result
        annotated_frame = cv2.resize(annotated_frame, (1365, 767))

        # Optical flow + ego-motion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (1365, 767))

        dx, dy = 0, 0
        if self.prev_gray is not None:
            _, dx, dy = self.compensate_ego_motion(self.prev_gray, gray)
        self.prev_gray = gray

        # trajectory.update(dx, dy)

        # Başlık bandı
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, "Oturum1 - YOLOv8",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        mins, secs = divmod(int(elapsed), 60)
        cv2.putText(annotated_frame, f"Elapsed: {mins:02d}:{secs:02d}  FPS: {fps}",
                    (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"dx: {dx:.1f}  dy: {dy:.1f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        combined = np.hstack([annotated_frame])

        torch.cuda.empty_cache()
        self.out.write(combined)