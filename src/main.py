import torch
import lightning as L
import datetime

from data_loader import SprDM
from model_loader import LDM
from arg_parser import get_args

import numpy as np
import random

# import sys
# from pathlib import Path
# root = Path.cwd().absolute().parent
# print(root)

# training_part = root / "training_part"
# print(training_part)
# if str(root) not in sys.path:
#     sys.path.append(str(root))
# if str(training_part) not in sys.path:
#     sys.path.append(str(training_part))

# diffusers_path = training_part / "diffusers"
# print(diffusers_path)
# if str(diffusers_path) not in sys.path:
#     sys.path.append(str(diffusers_path))

# utils_path = root / "utils"
# print(utils_path)
# if str(utils_path) not in sys.path:
#     sys.path.append(str(utils_path))

# print(f"========\n\t\t{sys.path}\n========")



# GPU performance increases!
torch.set_float32_matmul_precision('medium')

def train(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print(".................................")
    print(f"The provided arguments are\n\t {args}")
    print(".................................")
    
    model = LDM(
        lr=args.lr,
        num_class_embeds=args.num_class_embeds,
        scheduler_name=args.scheduler_name,
        checkpoint_monitor=args.checkpoint_monitor,
        checkpoint_mode=args.checkpoint_mode,
        is_inherited=True,
    )
    dm = SprDM(
        hf_dataset_repo=args.hf_dataset_repo,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        dl_num_workers=args.dl_num_workers
    )
    
    train_log_dir = f"{args.train_log_folder}/{timestamp}_batch{args.batch_size}_epochs{args.max_epochs}"
    
    trainer = L.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=args.device_num,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=train_log_dir,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    print("DONE!!!!!!!!!!!!!")
    

def predict(args):
    model = LDM.load_from_checkpoint(
        args.checkpoint_path,
        lr=args.lr,
        num_class_embeds=args.num_class_embeds,
        scheduler_name=args.inference_scheduler_name,
        checkpoint_monitor=args.checkpoint_monitor,
        checkpoint_mode=args.checkpoint_mode,
    )
    
    # dm = SprDM(
    #     hf_dataset_repo=args.hf_dataset_repo,
    #     batch_size=args.batch_size,
    #     shuffle=args.shuffle,
    #     dl_num_workers=args.dl_num_workers
    # )
    
    # trainer = L.Trainer(
    #     accelerator="gpu" if args.device == "cuda" else "cpu",
    #     devices=args.device_num,
    #     deterministic=True,   
    # )
    # trainer.predict(model=model, datamodule=dm)
    model()
    
if __name__ == "__main__":
    args = get_args()
    
    # for reproducibility
    # TODO: check reproducibility
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    if args.predict:
        predict(args)
    else:
        train(args)