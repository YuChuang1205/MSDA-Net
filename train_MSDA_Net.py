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
import torch.optim as optim
from model.MSDA_Net.MSDA import MSDANet
from loss.softloss import SoftIoULoss
from components.metric_all_2 import *
from components.utils_all import (
    save_checkpoint,
    get_loaders,
)
from datetime import datetime
from components.drawing import drawing_iou, drawing_loss
import sys

# ----------------获取当前运行文件的文件名------------------#
# 获取当前正在运行的文件的绝对路径
file_path = os.path.abspath(sys.argv[0])
# 获取文件名
file_name = os.path.basename(file_path)
# 去掉扩展名
file_name_without_ext = os.path.splitext(file_name)[0]
# 分割并提取所需部分
parts = file_name_without_ext.split('_')
if len(parts) > 2:
    extracted_part = '_'.join(parts[1:])
else:
    extracted_part = file_name_without_ext
# ------------------------------------------------------#
MODEL_NAME = extracted_part

# Hyperparameters etc.
LEARNING_RATE = 1e-4
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1
NUM_EPOCHS = 500  # 500
NUM_WORKERS = 4
IMAGE_SIZE = 512  # NUDT—SIRST数据集是256
PIN_MEMORY = True
LOAD_MODEL = False
root_path = os.path.abspath('.')
TRAIN_IMG_DIR = root_path + "/dataset/IRSTD-1K/train/image"
TRAIN_MASK_DIR = root_path + "/dataset/IRSTD-1K/train/mask"
VAL_IMG_DIR = root_path + "/dataset/IRSTD-1K/test/image"
VAL_MASK_DIR = root_path + "/dataset/IRSTD-1K/test/mask"
num_images = len(os.listdir(VAL_MASK_DIR))


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def main():
    def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
        model.train()
        loop = tqdm(loader)
        iou_metric.reset()
        nIoU_metric.reset()
        train_losses = []
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.unsqueeze(1).to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            iou_metric.update(predictions, targets)
            nIoU_metric.update(predictions, targets)
            _, IoU = iou_metric.get()
            _, nIoU = nIoU_metric.get()

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_description(f"train epoch is：{epoch + 1} ")
            loop.set_postfix(loss=loss.item())
            loop.set_postfix(IoU=IoU.item(), nIoU=nIoU.item())

        return IoU, nIoU, np.mean(train_losses)

    def val_fn(loader, model, loss_fn, epoch):
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
                loss = loss_fn(preds, y)
                eval_losses.append(loss.item())

                iou_metric.update(preds, y)
                nIoU_metric.update(preds, y)
                FA_PD_metric.update(preds, y)
                _, IoU = iou_metric.get()
                _, nIoU = nIoU_metric.get()
                FA, PD = FA_PD_metric.get(num_images)
                loop.set_description(f"test epoch is：{epoch + 1} ")
                loop.set_postfix(IoU=IoU.item(), nIoU=nIoU.item())
        return IoU, nIoU, np.mean(eval_losses), FA, PD

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.SomeOf([
                A.VerticalFlip(p=0.5),  # 围绕X轴垂直翻转
                A.HorizontalFlip(p=0.5),  # 围绕Y轴垂直翻转
                A.Transpose(p=0.5),  # 转置
                A.RandomRotate90(p=0.5),  # 和上面中的3个有一样的效果
                A.RandomBrightness(limit=0.2, p=0.2),
                A.RandomContrast(limit=0.2, p=0.2),
                A.Rotate(limit=45, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.2),  # 仅作用在输入图像，而不作用在mask上
                A.NoOp(),
                A.NoOp(),
            ], 3, p=0.5),
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

    model = MSDANet().to(DEVICE)
    loss_fn = SoftIoULoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    FA_PD_metric = PD_FA_2(1)

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

    scaler = torch.cuda.amp.GradScaler()

    best_mIoU = 0
    best_nIoU = 0
    best_mIoU_nIoU = 0
    best_mIoU_FA = 0
    best_mIoU_PD = 0
    best_nIoU_mIoU = 0
    best_nIoU_FA = 0
    best_nIoU_PD = 0
    best_mIoU_epoch = 0
    best_nIoU_epoch = 0

    num_epoch = []
    num_train_loss = []
    num_test_loss = []
    num_mioU = []
    num_nioU = []

    if not os.path.exists('./work_dirs'):
        os.mkdir('./work_dirs')

    save_model_file_path = os.path.join(root_path, 'work_dirs', MODEL_NAME)
    make_dir(save_model_file_path)
    save_file_name = os.path.join(save_model_file_path, MODEL_NAME + '.txt')
    save_best_miou_file_name = os.path.join(save_model_file_path,
                                            'best_mIoU_checkpoint_' + MODEL_NAME + ".pth.tar")
    save_best_niou_file_name = os.path.join(save_model_file_path,
                                            'best_nIou_checkpoint_' + MODEL_NAME + ".pth.tar")

    save_file = open(save_file_name, 'a')
    save_file.write(
        '\n---------------------------------------start--------------------------------------------------\n')
    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    for epoch in range(NUM_EPOCHS):

        train_mioU, train_nioU, train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # 评价指标

        mioU, nioU, test_loss, fa, pd = val_fn(val_loader, model, loss_fn, epoch)

        # save model
        checkpoint = {
            'epoch': epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_mIoU': best_mIoU,
            'best_nIoU': best_nIoU,
            'mIoU': mioU,
            'nIoU': nioU,
            'FA': fa,
            'PD': pd,
        }
        save_checkpoint(checkpoint)

        num_epoch.append(epoch + 1)
        num_train_loss.append(train_loss)
        num_test_loss.append(test_loss)
        num_mioU.append(mioU)
        num_nioU.append(nioU)

        if best_mIoU < mioU:
            best_mIoU = mioU
            best_mIoU_nIoU = nioU
            best_mIoU_FA = fa
            best_mIoU_PD = pd
            best_mIoU_epoch = epoch
            if epoch + 1 > 100:
                torch.save(checkpoint, save_best_miou_file_name)
        if best_nIoU < nioU:
            best_nIoU = nioU
            best_nIoU_mIoU = mioU
            best_nIoU_FA = fa
            best_nIoU_PD = pd
            best_nIoU_epoch = epoch
            if epoch + 1 > 100:
                torch.save(checkpoint, save_best_niou_file_name)
        print(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n"
              f"当前epoch:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa * 1000000, 3)}  PD:{round(pd, 4)} \n"
              f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n"
              f"best_epoch:{best_nIoU_epoch + 1}  b_miou:{round(best_nIoU_mIoU, 4)}  best_niou:{round(best_nIoU, 4)}  FA:{round(best_nIoU_FA * 1000000, 3)}  PD:{round(best_nIoU_PD, 4)}")

        save_file.write(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n")
        save_file.write(
            f"epoch is:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa, 3)}  PD:{round(pd, 4)}\n")
        save_file.write(
            f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n")
        save_file.write(
            f"best_epoch:{best_nIoU_epoch + 1}  b_miou:{round(best_nIoU_mIoU, 4)}     best_niou:{round(best_nIoU, 4)}  FA:{round(best_nIoU_FA * 1000000, 3)}  PD:{round(best_nIoU_PD, 4)}\n")
        save_file.flush()
    drawing_loss(num_epoch, num_train_loss, num_test_loss)
    drawing_iou(num_epoch, num_mioU, num_nioU)
    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    save_file.write('\n---------------------------------------end--------------------------------------------------\n')


if __name__ == "__main__":
    main()
