import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    transforms = {
        "images": {
            "train": A.Compose(
                [   
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [
                    ToTensorV2(),
                ]
            )
        },
    }
    
    return transforms