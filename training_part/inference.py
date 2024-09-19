import torch
import lightning as L
import datetime
from PIL import Image

from data_loader import SPRDiffusionDataModule
from model_loader import SPRDiffusionModel
from arg_parser import get_args
from configs import Config

import diffusers.src.diffusers as diffusers
from tqdm import tqdm
from pathlib import Path



def main(args):
    unet = SPRDiffusionModel.load_from_checkpoint(args.checkpoint_path, lr=0.001, num_class_embeds=2)
    unet.eval()
    scheduler = diffusers.schedulers.DDPMScheduler()
    
    image = torch.randn((args.batch_size, 3, Config.RESIZED_HEIGHT, Config.RESIZED_WIDTH))
    image = image.to(Config.DEVICE)
    
    # TODO: normalisation through util. 
    # Now, there are multiple usages of this in data_loader.py and here
    num_plates = args.num_plates
    num_plates = (num_plates - Config.MEAN_NUM_PLATES) / Config.STD_NUM_PLATES
    types = 0 if args.types == Config.TYPES else 1
    types = (types - Config.MEAN_TYPES) / Config.STD_TYPES
    tensor = torch.Tensor([[num_plates, types]])
    conditions = torch.concat([tensor] * args.batch_size, axis=0)
    conditions = conditions.to(Config.DEVICE)

    scheduler.set_timesteps(Config.TIMESTEPS)
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            t = t.to(Config.DEVICE)
            outputs = unet.model(image, t, conditions)
            image = scheduler.step(outputs.sample, t, image).prev_sample


    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, one_img in enumerate(image):
        one_img = (one_img * 255).round().astype("uint8")
        img_to_save = Image.fromarray(one_img)
        
        folder_str = f"./{timestamp}_inference"
        folder = Path(folder_str)
        if not folder.exists():
            folder.mkdir()
            print(f"{folder} made..!")
        img_to_save.save(str(folder / f"{str(idx).zfill(2)}.png"))
        print(f"........{idx}th image saved!")
    
    print("==== inference DONE")
if __name__ == "__main__":
    args = get_args()
    main(args)