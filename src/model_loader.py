import diffusers.src.diffusers as diffusers

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
    
from configs import Config
from utils import model_utils
import random
import numpy as np
from tqdm import tqdm
import gc

class SprDDPM(L.LightningModule):
    # TODO: parameters to be received outside the class
    def __init__(
        self, 
        lr, 
        num_class_embeds, 
        scheduler_name,
        checkpoint_monitor, 
        checkpoint_mode, 
        is_inherited,
        inference_height,
        inference_width,
        inference_batch_size=Config.INFERENCE_BATCH_SIZE,
        inference_scheduler_name=Config.INFERENCE_SCHEDULER_NAME,
        inference_c_1=Config.C1,
        inference_c_2=Config.C2,
        num_inference_steps=Config.NUM_INFERENCE_TIMESTEPS,
        num_train_steps=Config.NUM_TRAIN_TIMESTEPS,
    ):
        super().__init__()
        self.lr = lr
        self.num_class_embeds = num_class_embeds
        self.is_inherited = is_inherited
        
        if self.is_inherited:
            self.model = diffusers.models.UNet2DModel(
                sample_size=(240, 320),
                class_embed_type="vector",
                num_class_embeds=self.num_class_embeds,
                in_channels=4,
                out_channels=4,
            )
        else:
            self.model = diffusers.models.UNet2DModel(
                sample_size=(240, 320),
                class_embed_type="vector",
                num_class_embeds=self.num_class_embeds,
            )
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.scheduler_name = scheduler_name
        self.scheduler = model_utils.get_scheduler(self.scheduler_name)
        
        # set train timesteps
        self.scheduler.set_timesteps(num_train_steps)
        
        self.inference_scheduler_name = inference_scheduler_name
        self.inference_batch_size = inference_batch_size
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        self.inference_c_1 = inference_c_1
        self.inference_c_2 = inference_c_2
        self.inference_height = inference_height
        self.inference_width = inference_width
    
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
        
        # TODO: FID
        # TODO: log FID
        return loss
    

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

        # checkpoint_save_per_500 = ModelCheckpoint(
        #     save_top_k=-1,
        #     every_n_epochs=500,
        #     filename="{epoch}-{step}-{train_loss:.4f}_save_per_500"
        # )
        
        checkpoint_save_top_loss = ModelCheckpoint(
            save_top_k=3,
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
            every_n_epochs=1,
            filename="{epoch}-{step}-{train_loss:.4f}"
        )
        
        return [checkpoint_save_last, checkpoint_save_top_loss]

    @torch.no_grad()
    def forward(self):
        # set scheduler
        scheduler = model_utils.get_scheduler(self.inference_scheduler_name)
        scheduler.set_timesteps(self.num_inference_steps)
        
        # get z
        white_noise = torch.randn((self.inference_batch_size, 4 if self.is_inherited else 3, Config.RESIZED_HEIGHT, Config.RESIZED_WIDTH))
        white_noise = white_noise.to(Config.DEVICE)
        
        # prepare conditions

        c1 = model_utils.normalise_to_minus_one_and_one(self.inference_c_1, Config.C1_MIN, Config.C1_MAX)
        c2 = 0 if self.inference_c_2 == Config.TYPES else 1
        c2 = model_utils.normalise_to_minus_one_and_one(c2, Config.C2_MIN, Config.C2_MAX)
        
        # batch conditions
        conditions = torch.Tensor([[c1, c2]])
        conditions = torch.concat([conditions] * self.inference_batch_size, axis=0)
        conditions = conditions.to(Config.DEVICE)
        
        # inference loop
        for t in tqdm(scheduler.timesteps):
            t = t.to(Config.DEVICE)
            outs = self.model(white_noise, t, conditions)
            white_noise = scheduler.step(outs.sample, t, white_noise).prev_sample
        
        # TODO: log images
        
        if not self.is_inherited:
            self.save_generated_image(white_noise)
        else:
            return white_noise

    def save_generated_image(self, batch_outs):
        # save images
        outs = model_utils.normalise_to_zero_and_one_from_minus_one(batch_outs)
        outs = model_utils.resize_to_original_ratio(outs, self.inference_height, self.inference_width)
        model_utils.save_image(outs)

class LDM(SprDDPM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            subfolder="vae", 
            torch_dtype=torch.float,
        )
        for param in self.vae.parameters():
            param.requires_grad = False
        self.scaling_factor = 0.18215
        
    def shared_step(self, batch, stage):
        images = batch[0]        
        batch[0] = self.encode_img(images) # reduce the size by the factor of 8 both sides
                
        return super().shared_step(batch, stage)
    
    @torch.no_grad()
    def forward(self):
        z = super().forward()
        x = self.decode_latent(z)
        
        # save images
        self.save_generated_image(x)
    
    @torch.no_grad()    
    def encode_img(self, input_img, is_scaling=False):
        # source 1: https://wandb.ai/capecape/ddpm_clouds/reports/Using-Stable-Diffusion-VAE-to-encode-satellite-images--VmlldzozNDA2OTgx
        # source 2: https://huggingface.co/blog/stable_diffusion#writing-your-own-inference-pipeline
        # TODO: to use scaling or not depends on the original normalisation range of the pre-trained vae input...?
        #       or to match what is expected from the next model, unet, as an input range?
        # source 3: https://forums.fast.ai/t/why-scaling-up-image-before-sending-to-vae/101370/4
        # source 4: https://huggingface.co/blog/annotated-diffusion
        # -----> it seems like in DDPM paper, it is assuming the input image to be ranged [-1, 1]
        
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
            
        if is_scaling:
            latent = self.vae.encode(input_img*2 - 1) # Note scaling
        else:
            latent = self.vae.encode(input_img)
        return self.scaling_factor * latent.latent_dist.sample()
    
    @torch.no_grad()
    def decode_latent(self, latent):
        scaled = 1. / self.scaling_factor * latent
        
        # to reduce GPU memory used in prediction
        # It used to throw an error of torch.cuda.OutOfMemoryError previously
        # that tried to allocate 4.00 GiB, but failed
        self.model.cpu()
        del self.model, latent
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.vae.decode(scaled).sample