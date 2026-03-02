import sys
import os
import time
import cv2
import torch
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from functions import ModelFunctions
import torch.nn.functional as F

sys.path.append(os.path.join(os.getcwd(), 'RAFT', 'core'))
from raft import RAFT
from utils.utils import InputPadder

FRAME_PATH = '../Data/T1'
OUTPUT_PATH = '../Data/T1_Tracked/tracked_output.mp4'
FPS = 20

def prepare_raft_tensor(img_bgr, device):
    img_bgr = cv2.imread(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).float()[None].to(device)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YOLO("yolov8n_trained.pt")
    
    # RAFT Kurulumu
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.small = True           
    args.mixed_precision = True 
    args.alternate_corr = False 
    
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load("RAFT/models/raft-small.pth", map_location=device))
    raft_model = raft_model.module.to(device).eval()

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1365, 767))
    functions = ModelFunctions(fps=FPS, frame_path=FRAME_PATH, output_path=OUTPUT_PATH, device=device, model=model, out=out)
    
    frames = functions.read_folder(FRAME_PATH)
    start_time = time.time()
    
    prev_tensor = None

    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        curr_tensor = prepare_raft_tensor(frame, device)
        flow_up = None
        if prev_tensor is not None:
            # RAFT'a girmeden önce tensörü %50 küçült (BELLEK KURTARICISI)
            scale_factor = 0.5
            curr_tensor_small = F.interpolate(curr_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            prev_tensor_small = F.interpolate(prev_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)

            padder = InputPadder(prev_tensor_small.shape)
            img1, img2 = padder.pad(prev_tensor_small, curr_tensor_small)
            
            with torch.no_grad():
                _, flow_up = raft_model(img1, img2, iters=8, test_mode=True)
                flow_up = padder.unpad(flow_up)
            
            # RAFT'tan çıkan küçük hareket haritasını orijinal YOLO boyutuna geri büyüt (gerek yok yüksek çözünürlüğe)
            flow_up = F.interpolate(flow_up, size=(curr_tensor.shape[2], curr_tensor.shape[3]), mode='bilinear', align_corners=False)
            
            # Resim %50 küçüldüğü için hareket vektörleri de yarıya inmişti, onları 2 ile çarpıp düzeltiyoruz
            flow_up = flow_up * (1.0 / scale_factor)

        elapsed_time = time.time() - start_time
        
        functions.track(frame, elapsed_time, FPS, i, flow_up=flow_up)
        
        prev_tensor = curr_tensor
    
    out.release()
    print("The tracking video has been saved to:", OUTPUT_PATH)