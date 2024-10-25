from ..diffusers.src import diffusers
from configs import Config

from torchmetrics.image.fid import FrechetInceptionDistance
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import datetime

def get_scheduler(scheduler_name):
    scheduler = None
    if scheduler_name == "DDPMScheduler":
        scheduler = diffusers.schedulers.DDPMScheduler()
    elif scheduler_name == "DDIMScheduler":
        scheduler = diffusers.schedulers.DDIMScheduler()
    elif scheduler_name == "DDPMParallelScheduler":
        scheduler = diffusers.schedulers.DDPMParallelScheduler()
    elif scheduler_name == "DDIMParallelScheduler":
        scheduler = diffusers.schedulers.DDIMParallelScheduler()
    elif scheduler_name == "AmusedScheduler":
        scheduler = diffusers.schedulers.AmusedScheduler()
    elif scheduler_name == "DDPMWuerstchenScheduler":
        scheduler = diffusers.schedulers.DDPMWuerstchenScheduler()
    elif scheduler_name == "DDIMInverseScheduler":
        scheduler = diffusers.schedulers.DDIMInverseScheduler()
    elif scheduler_name == "CogVideoXDDIMScheduler":
        scheduler = diffusers.schedulers.CogVideoXDDIMScheduler()
    # else:
    #     raise Exception, f"scheduler name should be given, but [{scheduler_name = }]"
    return scheduler


def get_fid(fake_images, real_images):
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()


def get_real_images():
    pass


def normalise_to_minus_one_and_one(x, x_min, x_max):
    normalised = (x - x_min) / x_max # to [0, 1]
    normalised = normalised * 2 - 1 # to [-1, 1]
    return normalised


def normalise_to_zero_and_one_from_minus_one(x, to_numpy=True):
    out = (x / 2 + 0.5).clamp(0, 1)

    if to_numpy:
        out = out.cpu().permute(0, 2, 3, 1).numpy()
    else:
        out = out.cpu()
    return out


def get_transforms():
    transforms = {
        "images": {
            "train": A.Compose(
                [   
                    A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
                    A.Normalize(mean=0.5, std=0.5), # supposed to make it range [-1, 1]
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
                    A.Normalize(mean=0.5, std=0.5),                    
                    ToTensorV2(),
                ]
            )
        },
    }
    
    return transforms


def save_image(images):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, img in enumerate(images):
        img = (img * 255).round().astype("uint8")
        img_to_save = Image.fromarray(one_img)
        
        folder_str = f"./{timestamp}_inference"
        folder = Path(folder_str)
        if not folder.exists():
            folder.mkdir()
            print(f"{folder} made..!")
        img_to_save.save(str(folder / f"{str(idx).zfill(2)}.png"))
        print(f"........{idx}th image saved!")