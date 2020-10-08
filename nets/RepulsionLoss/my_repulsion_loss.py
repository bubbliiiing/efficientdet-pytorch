import math
import torch
from torch.autograd import Variable
from .bbox_transform import bbox_transform_inv, bbox_overlaps
from numba import jit

# https://github.com/dongdonghy/repulsion-loss-faster-rcnn-pytorch/blob/master/lib/model/faster_rcnn/repulsion_loss.py

def IoG(box_a, box_b):
    inter_xmin = torch.max(box_a[0], box_b[0])   #torch.max(input, dim, keepdim=False, out=None)
    inter_ymin = torch.max(box_a[1], box_b[1])
    inter_xmax = torch.min(box_a[2], box_b[2])
    inter_ymax = torch.min(box_a[3], box_b[3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return I / G

def IoG_batch(box_a, box_b):
    """
       param box:  (N, 4) ndarray of float
       """
    N = box_a.shape[0]
    xmin_cat = torch.cat([box_a[:, 0].view(N, 1), box_b[:, 0].view(N, 1)], dim=-1)
    ymin_cat = torch.cat([box_a[:, 1].view(N, 1), box_b[:, 1].view(N, 1)], dim=-1)
    xmax_cat = torch.cat([box_a[:, 2].view(N, 1), box_b[:, 2].view(N, 1)], dim=-1)
    ymax_cat = torch.cat([box_a[:, 3].view(N, 1), box_b[:, 3].view(N, 1)], dim=-1)
    print("xmin", xmin_cat)   #input inf??

    inter_xmin,indice = torch.max(xmin_cat, dim=-1)
    inter_ymin,indice = torch.max(ymin_cat, dim=-1)
    inter_xmax,indice = torch.min(xmax_cat, dim=-1)
    inter_ymax,indice = torch.min(ymax_cat, dim=-1)


    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)  #缩紧
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    output = torch.div(I.to(torch.float64), G.to(torch.float64))  #type

    return output


def smooth_ln(input, delta=0.9):
    output = torch.where(torch.ls(input, delta), -torch.log(1 - input),
                         (input - delta) / (1 - delta) - torch.log(1 - input))
    return output



def RepGT(pred_boxes, gt_boxes):  # B, G   #, rois_inside_ws

    sigma_repgt = 0.9
    loss_repgt = torch.zeros(len(pred_boxes)).cuda()

    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        gt_box = gt_boxes[i]  #one

        num_repgt = 0
        repgt_smoothln = 0


        overlaps = bbox_overlaps(pred_box, gt_box)

        if pred_box.shape[0] != 0 :
            max_overlaps, argmax_overlaps = torch.max(overlaps, dim=1)

            for j in range(max_overlaps.shape[0]):
                if max_overlaps[j] > 0:
                    num_repgt += 1
                    iog = IoG(pred_box[j], gt_box[argmax_overlaps[j]])  # G, P

                    ln = torch.where(torch.le(iog, sigma_repgt), -torch.log(1 - iog),
                                     (iog - sigma_repgt) / (1 - sigma_repgt) - math.log(1 - sigma_repgt))

                    repgt_smoothln += ln

            if num_repgt > 0:
                loss_repgt[i] = repgt_smoothln / num_repgt

    return loss_repgt


def RepBox(pred_boxes, gt_boxes):
    sigma_repbox = 0
    loss_repbox = torch.zeros(len(pred_boxes)).cuda()

    for i in range(len(pred_boxes)):

        pred_box = pred_boxes[i]
        gt_box = gt_boxes[i]


        num_repbox = 0
        repbox_smoothln = 0
        if pred_box.shape[0] > 0:

            overlaps = bbox_overlaps(pred_box, pred_box)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if z >= j:
                        overlaps[j, z] = 0
                    elif int(torch.sum(gt_box[j] == gt_box[z])) == 4:
                        overlaps[j, z] = 0

            iou = overlaps[overlaps > 0]
            for j in range(iou.shape[0]):
                num_repbox += 1
                if iou[j] <= sigma_repbox:
                    repbox_smoothln += -math.log(1 - iou[j])
                elif iou[j] > sigma_repbox:
                    repbox_smoothln += ((iou[j] - sigma_repbox) / (1 - sigma_repbox) - math.log(1 - sigma_repbox))

        if num_repbox > 0:
            loss_repbox[i] = repbox_smoothln / num_repbox

    return loss_repbox


def repulsion(gt_box, pred_box):
    # pred_box = pred_box[torch.arange(pred_box.size(0)) != 1]

    loss_RepGT = RepGT(pred_box, gt_box)
    # loss_RepBox = RepBox(pred_box, gt_box)

    return loss_RepGT

if __name__ == '__main__':
    a = torch.tensor([[3, 4, 5, 7], [3, 4, 5, 7]])
    b = torch.tensor([[4, 5, 7, 8], [4, 5, 7, 8]])
    IoG_batch(a, b)
