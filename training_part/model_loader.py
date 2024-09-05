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
        # self.model = diffusers.models.UNet2DConditionModel(
        #     sample_size=(int(Config.TARGET_HEIGHT / 8), int(Config.TARGET_WIDTH / 8)),
        #     in_channels=3,
        #     out_channels=3,
        #     class_embed_type="simple_projection",
        #     projection_class_embeddings_input_dim=2,
        # )
        
        self.model = diffusers.models.UNet2DModel(
            sample_size=(240, 320),
            class_embed_type="vector",
            num_class_embeds=2,
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
        
        # encoder_hidden_states = torch.randn((images.size()[0], 1000, 1280)).cuda()
        
        # residual = self.model(noisy_images, steps, encoder_hidden_states=encoder_hidden_states, class_labels=conditions)
        unet_2d_outputs = self.model(noisy_images, steps, conditions, class_embed_type="vector")
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
            filename="{epoch}-{step}-{train_loss:.3f}"
        )
        
        callbacks.append(checkpoint)
        return callbacks