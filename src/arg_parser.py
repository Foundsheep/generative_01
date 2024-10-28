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
    parser.add_argument("--num_device", type=int, default=Config.NUM_DEVICE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--num_class_embeds", type=int, default=Config.NUM_CLASS_EMBEDS)
    parser.add_argument("--checkpoint_monitor", type=str, default=Config.CHECKPOINT_MONITOR)
    parser.add_argument("--checkpoint_mode", type=str, default=Config.CHECKPOINT_MODE)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--fast_dev_run", action=argparse.BooleanOptionalAction)

    # inference
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--num_plates", type=int)
    parser.add_argument("--types", type=str, default=Config.TYPES)
    parser.add_argument("--inference_scheduler_name", type=str, default=Config.INFERENCE_SCHEDULER_NAME)
    parser.add_argument("--inference_batch_size", type=int, default=Config.INFERENCE_BATCH_SIZE)

    # both
    parser.add_argument("--scheduler_name", type=str, default=Config.SCHEDULER_NAME)
    parser.add_argument("--predict", action=argparse.BooleanOptionalAction)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction)
    return parser.parse_args()