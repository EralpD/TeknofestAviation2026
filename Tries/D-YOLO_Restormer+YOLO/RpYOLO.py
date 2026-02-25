import torch
import torch.nn.functional as F
import cv2
import numpy as np
import requests
from runpy import run_path
from skimage import img_as_ubyte
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import time
import os

# SETTINGS
TASK = 'Single_Image_Defocus_Deblurring'  # Specific Task
WEIGHTS_PATH = 'Restormer/Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth'
FRAME_PATH = '../Data/Oturum1'  # Path to input frames (extracted from video)
OUTPUT_PATH = '../Data/Oturum1_Tracked/tracked_output.mp4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Loading
parameters = {
    'inp_channels': 3, 'out_channels': 3, 'dim': 48,
    'num_blocks': [4,6,6,8], 'num_refinement_blocks': 4,
    'heads': [1,2,4,8], 'ffn_expansion_factor': 2.66,
    'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False
}

load_arch = run_path('Restormer/basicsr/models/archs/restormer_arch.py')
rmodel = load_arch['Restormer'](**parameters)
model = YOLO("yolov8l.pt") # For this project, large model in use


checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
rmodel.load_state_dict(checkpoint['params'])
rmodel.to(device)
model.to(device)
rmodel.eval()

# Video Writer settings,
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1365, 767)) # Input Frames in 1365x767 format

# ---- INFERENCE FUNCTION ----
def restore(image_path, i: int, elapsed: float, save_path=OUTPUT_PATH):

    # !!! At every cache, clean the GPU memory to prevent out of memory results
    torch.cuda.empty_cache()

    # Load image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to tensor
    input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + 7) // 8) * 8
    W = ((w + 7) // 8) * 8
    input_ = F.pad(input_, (0, W-w, 0, H-h), 'reflect')
    
    ## Original resolution image 
    # with torch.no_grad():
    #     restored = rmodel(input_)
    # print_gpu_memory("Restormer inference sonrası")

    # del input_
    # torch.cuda.empty_cache()

    # TILE PROCESSING FOR LARGE IMAGES
    TILE = 512       
    OVERLAP = 32

    b, c, th, tw = input_.shape
    E = torch.zeros(b, c, th, tw).to(device)
    W_map = torch.zeros(b, 1, th, tw).to(device)

    stride = TILE - OVERLAP
    h_idxs = list(range(0, th - TILE, stride)) + [th - TILE]
    w_idxs = list(range(0, tw - TILE, stride)) + [tw - TILE]

    with torch.no_grad():
        for h_idx in h_idxs:
            for w_idx in w_idxs:
                patch = input_[..., h_idx:h_idx+TILE, w_idx:w_idx+TILE]
                out_tile = rmodel(patch)
                E[..., h_idx:h_idx+TILE, w_idx:w_idx+TILE] += out_tile
                W_map[..., h_idx:h_idx+TILE, w_idx:w_idx+TILE] += 1
                del patch, out_tile
                torch.cuda.empty_cache()

    restored = E / W_map
    del E, W_map, input_
    
    # Remove padding and save
    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    restored = restored.permute(0,2,3,1).cpu().numpy()
    restored = img_as_ubyte(restored[0])
    restored = np.ascontiguousarray(restored)

    # Replace the placement of models
    torch.cuda.empty_cache() 

    # Use YOLO for tracking
    result = model.track(
        restored, persist=True, 
        device=device, 
        conf=0.3, 
        iou=0.5,
        tracker="bytetrack.yaml",
        verbose=False,)
    annotated_frame = result[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1365, 767)) # Force to match the output video dimensions

    cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1) # Add a black rectangle for better text visibility
    cv2.putText(annotated_frame, "Oturum1 - YOLOv8 + Restormer Tracking",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(annotated_frame, f"Elapsed: {mins:02d}:{secs:02d}",
                (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    torch.cuda.empty_cache() # Clear GPU memory after processing each frame
    
    out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

def read_folder(folder_path):
    frames = []
    for frame in sorted(os.listdir(folder_path)):
        frames.append(os.path.join(folder_path, frame)) # in webp format
    return frames

if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))

    start_time = time.time()
    frames = read_folder(FRAME_PATH)
    for i, frame in tqdm(enumerate(frames), total=len(frames)):
        elapsed = time.time() - start_time
        restore(frame, i, elapsed)

    out.release()
    print(f"Tracking completed. Output saved to {OUTPUT_PATH}")