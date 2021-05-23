import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

def get_target(anchor, bbox_annotation, classification, cuda):
    #------------------------------------------------------#
    #   计算真实框和先验框的交并比
    #   anchor              num_anchors, 4
    #   bbox_annotation     num_true_boxes, 5
    #   Iou                 num_anchors, num_true_boxes
    #------------------------------------------------------#
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
    
    #------------------------------------------------------#
    #   计算与先验框重合度最大的真实框
    #   IoU_max             num_anchors,
    #   IoU_argmax          num_anchors,
    #------------------------------------------------------#
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)

    #------------------------------------------------------#
    #   寻找哪些先验框在计算loss的时候需要忽略
    #------------------------------------------------------#
    targets = torch.ones_like(classification) * -1
    if cuda:
        targets = targets.cuda()

    #------------------------------------------#
    #   重合度小于0.4需要参与训练
    #------------------------------------------#
    targets[torch.lt(IoU_max, 0.4), :] = 0

    #--------------------------------------------------#
    #   重合度大于0.5需要参与训练，还需要计算回归loss
    #--------------------------------------------------#
    positive_indices = torch.ge(IoU_max, 0.5)

    #--------------------------------------------------#
    #   取出每个先验框最对应的真实框
    #--------------------------------------------------#
    assigned_annotations = bbox_annotation[IoU_argmax, :]

    #--------------------------------------------------#
    #   将对应的种类置为1
    #--------------------------------------------------#
    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    #--------------------------------------------------#
    #   计算正样本数量
    #--------------------------------------------------#
    num_positive_anchors = positive_indices.sum()
    return targets, num_positive_anchors, positive_indices, assigned_annotations

def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    #--------------------------------------------------#
    #   取出作为正样本的先验框对应的真实框
    #--------------------------------------------------#
    assigned_annotations = assigned_annotations[positive_indices, :]

    #--------------------------------------------------#
    #   取出作为正样本的先验框
    #--------------------------------------------------#
    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    #--------------------------------------------------#
    #   计算真实框的宽高与中心
    #--------------------------------------------------#
    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    #---------------------------------------------------#
    #   利用真实框和先验框进行编码，获得应该有的预测结果
    #---------------------------------------------------#
    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.t()
    return targets

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, alpha = 0.25, gamma = 2.0, cuda = True):
        #---------------------------#
        #   获得batch_size的大小
        #---------------------------#
        batch_size = classifications.shape[0]

        #--------------------------------------------#
        #   获得先验框，将先验框转换成中心宽高的形式
        #--------------------------------------------#
        dtype = regressions.dtype
        anchor = anchors[0, :, :].to(dtype)
        #--------------------------------------------#
        #   将先验框转换成中心，宽高的形式
        #--------------------------------------------#
        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        regression_losses = []
        classification_losses = []
        for j in range(batch_size):
            #-------------------------------------------------------#
            #   取出每张图片对应的真实框、种类预测结果和回归预测结果
            #-------------------------------------------------------#
            bbox_annotation = annotations[j]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            if len(bbox_annotation) == 0:
                #-------------------------------------------------------#
                #   当图片中不存在真实框的时候，所有特征点均为负样本
                #-------------------------------------------------------#
                alpha_factor = torch.ones_like(classification) * alpha
                if cuda:
                    alpha_factor = alpha_factor.cuda()
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                
                #-------------------------------------------------------#
                #   计算特征点对应的交叉熵
                #-------------------------------------------------------#
                bce = - (torch.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                classification_losses.append(cls_loss.sum())
                #-------------------------------------------------------#
                #   回归损失此时为0
                #-------------------------------------------------------#
                if cuda:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                    
                continue

            #------------------------------------------------------#
            #   计算真实框和先验框的交并比
            #   targets                 num_anchors, num_classes
            #   num_positive_anchors    正样本的数量
            #   positive_indices        num_anchors, 
            #   assigned_annotations    num_anchors, 5
            #------------------------------------------------------#
            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor, 
                                                                                        bbox_annotation, classification, cuda)
            
            #------------------------------------------------------#
            #   首先计算交叉熵loss
            #------------------------------------------------------#
            alpha_factor = torch.ones_like(targets) * alpha
            if cuda:
                alpha_factor = alpha_factor.cuda()
            #------------------------------------------------------#
            #   这里使用的是Focal loss的思想，
            #   易分类样本权值小
            #   难分类样本权值大
            #------------------------------------------------------#
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            #------------------------------------------------------#
            #   把忽略的先验框的loss置为0
            #------------------------------------------------------#
            zeros = torch.zeros_like(cls_loss)
            if cuda:
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
            
            #------------------------------------------------------#
            #   如果存在先验框为正样本的话
            #------------------------------------------------------#
            if positive_indices.sum() > 0:
                targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
                #---------------------------------------------------#
                #   将网络应该有的预测结果和实际的预测结果进行比较
                #   计算smooth l1 loss
                #---------------------------------------------------#
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if cuda:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
        
        # 计算平均loss并返回
        c_loss = torch.stack(classification_losses).mean()
        r_loss = torch.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss
