#!/usr/bin/python3
# coding = gbk
"""
@Author : zhaojinmiao;yuchuang
@Time :
@desc:
"""
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model.MSDA_Net.MSDA import MSDANet

from components.metric_all_2 import *
from components.utils_all import (
    load_checkpoint,
    get_loaders,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY = True
IMAGE_SIZE = 512
root_path = os.path.abspath('.')

TRAIN_IMG_DIR = root_path + "/dataset/IRSTD-1K/train/image"
TRAIN_MASK_DIR = root_path + "/dataset/IRSTD-1K/train/mask"
VAL_IMG_DIR = root_path + "/dataset/IRSTD-1K/test/image"
VAL_MASK_DIR = root_path + "/dataset/IRSTD-1K/test/mask"
TEST_NUM = len(os.listdir(VAL_MASK_DIR))


def main():
    def val_fn(loader, model):
        model.eval()
        loop = tqdm(loader)
        iou_metric.reset()
        nIoU_metric.reset()
        FA_PD_metric.reset()
        eval_losses = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loop):
                x = x.to(device=DEVICE)
                y = y.unsqueeze(1).to(device=DEVICE)
                preds = model(x)
                iou_metric.update(preds, y)
                nIoU_metric.update(preds, y)
                FA_PD_metric.update(preds, y)
                _, IoU = iou_metric.get()
                _, nIoU = nIoU_metric.get()
                FA, PD = FA_PD_metric.get(TEST_NUM)

        return IoU, nIoU, FA, PD

    train_transform = A.Compose(
        [

            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    model = MSDANet().to(DEVICE)
    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    FA_PD_metric = PD_FA_2(1)
    load_checkpoint(torch.load("./test_work_dir/MSDA_Net_IRSTD_1K.pth.tar"), model)
    mioU, nioU, fa, pd = val_fn(val_loader, model)
    print(f"miou:{round(mioU, 4)}  niou:{round(nioU, 4)} FA:{round(fa * 1000000, 3)}  PD:{round(pd, 4)} ")


if __name__ == "__main__":
    main()
