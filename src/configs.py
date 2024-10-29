import torch
from pathlib import Path

class Config():
    # preprocess
    TARGET_IMAGE_RATIO = 0.75
    TARGET_HEIGHT = 1920
    TARGET_WIDTH = 2560
    # RESIZED_HEIGHT = 240
    # RESIZED_WIDTH = 320
    RESIZED_HEIGHT = 512
    RESIZED_WIDTH = 512
    INFERENCE_HEIGHT = 240
    INFERENCE_WIDTH = 320

    # train 
    HF_DATASET_REPO = "DJMOON/hm_spr_1_2"
    MAX_EPOCHS = 3
    MIN_EPOCHS = 1
    SHUFFLE = True
    BATCH_SIZE = 2
    DL_NUM_WORKERS = 2
    LOG_EVERY_N_STEPS = 1
    TRAIN_LOG_FOLDER = str(Path(__file__).absolute().parent)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_DEVICE = 2
    LR = 0.001
    NUM_CLASS_EMBEDS = 2
    SCHEDULER_NAME = "DDPMScheduler"
    CHECKPOINT_MONITOR = "train_loss"
    CHECKPOINT_MODE = "min"
    SEED = 622
    NUM_TRAIN_TIMESTEPS = 1000
    
    # MEAN_NUM_PLATES = 2.5
    # MEAN_TYPES = 0.5
    # STD_NUM_PLATES = 0.5
    # STD_TYPES = 0.5
    C1_MIN = 2
    C1_MAX = 3
    C2_MIN = 0
    C2_MAX = 1
    IMG_MIN = 0
    IMG_MAX = 255
    
    # inference
    TYPES = "HM"
    NUM_INFERENCE_TIMESTEPS = 100
    INFERENCE_BATCH_SIZE = 2
    INFERENCE_SCHEDULER_NAME = "DDPMScheduler"
    C1 = 3
    C2 = "HM"
    