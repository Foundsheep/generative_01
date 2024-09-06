import lightning as L
from datasets import load_dataset
import torch
import numpy as np
from PIL import Image

import sys
from pathlib import Path

root = Path.cwd().absolute().parent
print(root)
training_part = root / "training_part"
if str(root) not in sys.path:
    sys.path.append(str(root))
if str(training_part) not in sys.path:
    sys.path.append(str(training_part))
    
from configs import Config
from utils.image_utils import get_transforms



class SPRDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.ds = load_dataset("DJMoon/hm_spr_1_2_resized")["train"]
        self.transforms = get_transforms()
        self.is_train = is_train
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        image = self._adjust_ratio_and_convert_to_numpy(image)
        num_plates = self.ds[idx]["num_plates"]
        types = self.ds[idx]["types"]
        
        if self.transforms:
            if self.is_train:
                image = self.transforms["images"]["train"](image=image)["image"]
            else:
                image = self.transforms["images"]["val"](image=image)["image"]
        
        image = image.float()
        num_plates = torch.Tensor([num_plates])
        types = torch.Tensor([0] if types == "HM" else [1])
        
        conditions = torch.concat([num_plates, types], axis=0)
        return image, conditions
    
    def _adjust_ratio_and_convert_to_numpy(self, img):
        target_ratio = Config.TARGET_IMAGE_RATIO
        h = img.height
        w = img.width
        ratio = h / w
        new_ratio = 0.00

        if ratio == target_ratio:
            return np.array(img)
        
        elif ratio < target_ratio:
            new_h = int(w * target_ratio)
            half_new_h = (new_h - h) // 2
            if 2 * half_new_h != new_h:
                another_half = new_h - half_new_h - h
            else:
                another_half = half_new_h  
            
            img_np = np.array(img)
            img_new = np.pad(img_np, ((half_new_h, another_half), (0, 0), (0, 0)), "reflect")            
        else:
            new_w = int(h / target_ratio)
            half_new_w = (new_w - w) // 2
            if 2 * half_new_w != new_w:
                another_half = new_w - half_new_w - w
            else:
                another_half = half_new_w
            
            img_np = np.array(img)
            img_new = np.pad(img_np, ((0, 0), (half_new_w, another_half), (0, 0)), "reflect")

        new_ratio = img_new.shape[0] / img_new.shape[1]
        img_new = np.array(Image.fromarray(img_new).resize((Config.TARGET_WIDTH, Config.TARGET_HEIGHT)))
        assert f"{target_ratio :.2f}" == f"{new_ratio :.2f}", f"{target_ratio =}, {new_ratio =}"
        assert img_new.shape[0] == Config.TARGET_HEIGHT, f"{img_new.shape = }"
        return img_new
        

class SPRDiffusionDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        
    def prepare_data(self):
        load_dataset("DJMOON/hm_spr_1_2_resized")
        
    def setup(self, stage):
        self.ds = SPRDiffusionDataset()
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds, batch_size=4, shuffle=True, num_workers=2)

if __name__ == "__main__":
    spr_ds = SPRDiffusionDataset()
    dl = torch.utils.data.DataLoader(spr_ds, batch_size=5, shuffle=True, num_workers=2)

    d = next(iter(dl))
    
    print(len(d))
    print(d[0].size())