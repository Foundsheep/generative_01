import albumentations as A
from albumentations.pytorch import ToTensorV2
from training_part.configs import Config


def get_transforms():
    transforms = {
        "images": {
            "train": A.Compose(
                [   
                    A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),                    
                    ToTensorV2(),
                ]
            )
        },
    }
    
    return transforms