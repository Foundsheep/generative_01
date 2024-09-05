import diffusers
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import sys
from pathlib import Path

curdir = Path.cwd().absolute().parent / "training_part"
print(curdir)
if str(curdir) not in sys.path:
    sys.path.append(str(curdir))
    
from configs import Config


class SPRDiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DConditionModel(
            sample_size=(Config.TARGET_HEIGHT, Config.TARGET_WIDTH),
            in_channels=3,
            out_channels=3,
            class_embed_type="simple_projection",
            projection_class_embeddings_input_dim=2,
        )
        self.scheduler = diffusers.schedulers.DDPMScheduler()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        
        
    def shared_step(self, batch, stage):
        images = batch[0]
        conditions = batch[1]
        
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0), ), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps, encoder_hidden_states=None, class_labels=conditions)

        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler
        }
    
    def configure_callbacks(self):
        callbacks = []
        
        checkpoint = ModelCheckpoint(
            monitor="train_loss",
            filename="{epoch}-{step}-{train_loss:.3f}"
        )
        
        callbacks.append(checkpoint)
        return callbacks