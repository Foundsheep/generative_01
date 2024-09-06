import torch
import lightning as L
import datetime

from data_loader import SPRDiffusionDataModule
from model_loader import SPRDiffusionModel
from arg_parser import get_args


def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    print(".................................")
    print(f"The provided arguments are\n\t {args}")
    print(".................................")
    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')
    
    model = SPRDiffusionModel(
        lr=args.lr,
        num_class_embeds=args.num_class_embeds,        
    )
    dm = SPRDiffusionDataModule(
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
        default_root_dir=train_log_dir
    )

    trainer.fit(model, datamodule=dm)

    print("DONE!!!!!!!!!!!!!")
if __name__ == "__main__":
    args = get_args()
    main(args)