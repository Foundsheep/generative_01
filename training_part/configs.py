import torch
from pathlib import Path

class Config():
    # preprocess
    TARGET_IMAGE_RATIO = 0.75
    TARGET_HEIGHT = 1920
    TARGET_WIDTH = 2560

    # train
    HF_DATASET_REPO = "DJMOON/hm_spr_1_2_resized"
    MAX_EPOCHS = 3
    MIN_EPOCHS = 1
    SHUFFLE = True
    BATCH_SIZE = 2
    DL_NUM_WORKERS = 2
    LOG_EVERY_N_STEPS = 1
    TRAIN_LOG_FOLDER = str(Path(__file__).absolute().parent)
    DEVICE = "cuda"
    DEVICE_NUM = 2
    LR = 0.001
    NUM_CLASS_EMBEDS = 2