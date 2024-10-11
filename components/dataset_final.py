import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class SirstDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # img.convert(‘L’)为灰度图像
        # 有2种处理方式，第一种在mask进来前将其值放为0和（255）1
        mask = (mask > 127.5).astype(float)
        # 第二种 ，将其值归一化到0-1
        # mask = mask/255.0
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

