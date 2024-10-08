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

diffusers_path = training_part / "diffusers"
print(diffusers_path)
if str(diffusers_path) not in sys.path:
    sys.path.append(str(diffusers_path))
print(f"========\n\t\t{sys.path}\n========")
    
import training_part.diffusers.src.diffusers as diffusers

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
    
from training_part.configs import Config
from utils.model_utils import get_scheduler


class SPRDiffusionModel(L.LightningModule):
    def __init__(self, lr, num_class_embeds, scheduler_name, checkpoint_monitor, checkpoint_mode, use_vanilla=True):
        super().__init__()
        self.lr = lr
        self.num_class_embeds = num_class_embeds
        self.model = diffusers.models.UNet2DModel(
            sample_size=(240, 320),
            class_embed_type="vector",
            num_class_embeds=self.num_class_embeds,
        )
        self.scheduler = get_scheduler(scheduler_name)
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.use_vanilla = use_vanilla
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        if not self.use_vanilla:
            self.vae = diffusers.AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float)
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = None
    
    def shared_step(self, batch, stage):
        images = batch[0]
        conditions = batch[1]
        
        if not self.use_vanilla:
            images = self.vae(images).sample
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0), ), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        unet_2d_outputs = self.model(noisy_images, steps, conditions)
        residual = unet_2d_outputs.sample
        
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss
    
    def forward(self, conditions):
        batch_size = conditions.size(0)
        noise = torch.randn(batch_size, 3, Config.RESIZED_HEIGHT, Config.RESIZED_WIDTH)
        steps = torch.Tensor([[1000]])
        steps = torch.concat([steps] * batch_size, axis=0)
        return self.model(noise, steps, conditions)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler
        }
    
    def configure_callbacks(self):
        checkpoint_save_last = ModelCheckpoint(
            save_last=True,
            filename="{epoch}-{step}-{train_loss:.4f}_save_last"
        )
        
        checkpoint_save_top_loss = ModelCheckpoint(
            save_top_k=3,
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
            every_n_epochs=1,
            filename="{epoch}-{step}-{train_loss:.4f}"
        )
        
        return [checkpoint_save_last, checkpoint_save_top_loss]
    
    def encode_img(self, input_img):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            latent = self.vae.encode(input_img*2 - 1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()
