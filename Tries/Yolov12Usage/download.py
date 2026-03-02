from ultralytics.utils.downloads import download
from pathlib import Path

dir = Path("../../../Datasets/data")
download('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip', dir=dir)