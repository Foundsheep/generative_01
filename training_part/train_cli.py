import torch
import lightning as L
from data_loader import SPRDiffusionDataModule
from model_loader import SPRDiffusionModel

def main():
    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')
    
    model = SPRDiffusionModel()
    dm = SPRDiffusionDataModule()
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=2,
    )

    trainer.fit(model, datamodule=dm)

    print("DONE!!!!!!!!!!!!!")
if __name__ == "__main__":
    main()