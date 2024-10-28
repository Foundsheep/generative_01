import sys
print(sys.module)
print("====")
print(__name__)
print("-----")

from ..diffusers.src import diffusers
from configs import Config

from torchmetrics.image.fid import FrechetInceptionDistance
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import datetime
import numpy as np
import torch

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


def normalise_to_zero_and_one_from_minus_one(x: torch.Tensor, to_numpy=True) -> np.ndarray:
    out = (x / 2 + 0.5).clamp(0, 1)

    out = out.cpu().permute(0, 2, 3, 1).numpy() if to_numpy else out.cpu()
    return out


def get_transforms():
    transforms = {
        "images": {
            "train": A.Compose(
                [   
                    # interpolation=0 means cv2.INTER_NEAREST.
                    # default value is 1(cv2.INTER_LINEAR), which causes the array to have 
                    # other values from those already in the image
                    A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH, interpolation=0),
                    A.Normalize(mean=0.5, std=0.5), # supposed to make it range [-1, 1]
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH, interpolation=0),
                    A.Normalize(mean=0.5, std=0.5),                    
                    ToTensorV2(),
                ]
            )
        },
    }
    
    return transforms


def save_image(images: np.ndarray) -> None:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, img in enumerate(images):
        img = (img * 255).round().astype("uint8")
        img_to_save = Image.fromarray(img)
        
        folder_str = f"./{timestamp}_inference"
        folder = Path(folder_str)
        if not folder.exists():
            folder.mkdir()
            print(f"{folder} made..!")
        img_to_save.save(str(folder / f"{str(idx).zfill(2)}.png"))
        print(f"........{idx}th image saved!")
        
        
def resize_to_original_ratio(images: np.ndarray, to_h: int, to_w: int) -> np.ndarray:
    if images.ndim == 3:
        images = np.array([images])
    elif images.ndim != 4:
        raise ValueError(f"{images.ndim = }, should be either 3 or 4")

    resize_func = A.Resize(height=to_h, width=to_w, interpolation=0)
    result = []
    for img in images:
        out = resize_func(image=img)["image"]
        result.append(out)

    return np.array(result)
    