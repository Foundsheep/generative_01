import torch
from pathlib import Path

class Config():
    # preprocess
    RESIZED_HEIGHT = 480 # height and width should be divisible by 32
    RESIZED_WIDTH = 640
    CROP_HEIGHT = 1248
    CROP_WIDTH = 1664
    TARGET_IMAGE_RATIO = 0.75
    TARGET_HEIGHT = 1920
    TARGET_WIDTH = 2560
