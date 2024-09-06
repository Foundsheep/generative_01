import sys
from pathlib import Path

root = Path.cwd().absolute().parent
print(root)
training_part = root / "training_part"
print(training_part)
if str(root) not in sys.path:
    sys.path.append(str(root))
if str(training_part) not in sys.path:
    sys.path.append(str(training_part))

print(f"========\n\t\t{sys.path}\n========")
    
import diffusers.src.diffusers as diffusers
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
    
from configs import Config


class SPRDiffusionModel(L.LightningModule):
    def __init__(self, lr, num_class_embeds):
        super().__init__()
        self.lr = lr
        self.num_class_embeds = num_class_embeds
        self.model = diffusers.models.UNet2DModel(
            sample_size=(240, 320),
            class_embed_type="vector",
            num_class_embeds=self.num_class_embeds,
        )
        
        self.scheduler = diffusers.schedulers.DDPMScheduler()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        
        
    def shared_step(self, batch, stage):
        images = batch[0]
        conditions = batch[1]
        
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0), ), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        unet_2d_outputs = self.model(noisy_images, steps, conditions)
        residual = unet_2d_outputs.sample
        
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
            filename="{epoch}-{step}-{train_loss:.4f}"
        )
        
        callbacks.append(checkpoint)
        return callbacks