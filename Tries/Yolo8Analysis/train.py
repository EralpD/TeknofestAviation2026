from ultralytics import YOLO
import torch

FRAME_PATH = '../Data/Oturum1'
OUTPUT_PATH = '../Data/Oturum1_Tracked/tracked_output.mp4'

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="visdroneCustom.yaml", epochs=100, batch=16, imgsz=640)
    model.save("yolov8n_trained.pt")