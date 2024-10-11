
import matplotlib.pyplot as plt

def drawing_loss(num_epoch, train_loss, test_loss):
    plt.figure()
    plt.plot(num_epoch, train_loss, label='train_loss', color='r',linewidth=0.5, linestyle='-', marker='o', markersize=0.5)
    plt.plot(num_epoch, test_loss, label='test_loss', color='b',linewidth=0.5,linestyle='-', marker='^', markersize=0.5)
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("work_dirs/fig_loss.png")

def drawing_iou(num_epoch, mIoU, nIoU):
    plt.figure()
    plt.plot(num_epoch, mIoU, label='mIoU', color='r',linewidth=0.5, linestyle='-', marker='o', markersize=0.5)
    plt.plot(num_epoch, nIoU, label='nIoU', color='b',linewidth=0.5, linestyle='-', marker='^', markersize=0.5)
    plt.legend()
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.savefig("work_dirs/fig_IoU.png")

