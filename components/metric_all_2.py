
import torch
import numpy as np
from skimage import measure

TEST_BATCH_SIZE = 1
class SigmoidMetric():
    def __init__(self, score_thresh=0.5):
        self.score_thresh = score_thresh
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        predict = (output > self.score_thresh).astype('int64')  # P
        target = (target > self.score_thresh).astype('int64')

        pixel_labeled = np.sum(target > 0)  # T
        pixel_correct = np.sum((predict == target) * (target > 0))  # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1
        nbins = 1
        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')
        target = (target.cpu().detach().numpy() > self.score_thresh).astype('int64')
        intersection = predict * (predict == target)  # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target):
        """nIoU"""

        mini = 1
        maxi = 1
        nbins = 1

        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')
        target = (target.cpu().detach().numpy() > self.score_thresh).astype('int64')
        intersection = predict * (predict == target)

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr

class PD_FA_2():
    def __init__(self, nclass):
        super(PD_FA_2, self).__init__()
        self.nclass = nclass
        self.image_area_total = []
        self.image_area_match = []
        self.FA = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels):


        predits  = np.array((preds > 0.5).cpu()).astype('int64')
        predits = np.squeeze(predits)
        self.image_h =predits.shape[0]
        self.image_w = predits.shape[1]

        labelss = np.array((labels > 0.5).cpu()).astype('int64')
        labelss = np.squeeze(labelss)

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)
                    del coord_image[m]
                    break
        self.dismatch = np.sum(self.image_area_total)-np.sum(self.image_area_match)
        self.FA += self.dismatch
        self.PD+=len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / ((self.image_h * self.image_w) * img_num)
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = 0
        self.PD  = 0
        self.target = 0

if __name__ == '__main__':
    pred = torch.rand(8, 1, 512, 512)
    target = torch.rand(8, 1, 512, 512)
    m1 = SigmoidMetric()
    m2 = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
    m1.update(pred, target)
    m2.update(pred, target)
    pixAcc, mIoU = m1.get()
    _, nIoU = m2.get()
