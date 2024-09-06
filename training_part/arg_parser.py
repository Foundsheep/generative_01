import argparse
from configs import Config

def get_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--hf_dataset_repo", type=str, default=Config.HF_DATASET_REPO)
    parser.add_argument("--max_epochs", type=int, default=Config.MAX_EPOCHS)
    parser.add_argument("--min_epochs", type=int, default=Config.MIN_EPOCHS)
    parser.add_argument("--shuffle", type=bool, default=Config.SHUFFLE)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--dl_num_workers", type=int, default=Config.DL_NUM_WORKERS)
    parser.add_argument("--log_every_n_steps", type=int, default=Config.LOG_EVERY_N_STEPS)
    parser.add_argument("--train_log_folder", type=str, default=Config.TRAIN_LOG_FOLDER)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    parser.add_argument("--device_num", type=int, default=Config.DEVICE_NUM)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--num_class_embeds", type=int, default=Config.NUM_CLASS_EMBEDS)

    return parser.parse_args()