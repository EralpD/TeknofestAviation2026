import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'RAFT', 'core'))
import argparse

import torch
import numpy as np
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()

def run_fast_raft():
    # Fast-RAFT (Small) Konfigürasyonu
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.small = True           # Kritik: Small model bloğunu aktif eder
    args.mixed_precision = True # Hız için FP16 (Yarım Hassasiyet)
    args.alternate_corr = False # Bellek dostu ama yavaş olanı kapatıyoruz

    # Modeli Yükle
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load("RAFT/models/raft-small.pth"))
    model = model.module
    model.cuda()
    model.eval()

    with torch.no_grad():
        image1 = load_image("frame1.png")
        image2 = load_image("frame2.png")

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # AGRESİFLEŞTİRME NOKTASI: iters parametresi
        # Standart 12'dir. Bunu 6 veya 8 yaparak hızı 2 katına çıkarabilirsiniz.
        flow_low, flow_up = model(image1, image2, iters=8, test_mode=True)
        
        # flow_up artık optik akış vektörlerini içerir [1, 2, H, W]
        print(f"Optik Akış Çıktısı Şekli: {flow_up.shape}")

if __name__ == '__main__':
    run_fast_raft()