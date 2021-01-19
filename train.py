#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import FocalLoss
from utils.dataloader import EfficientdetDataset, efficientdet_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def fit_one_epoch(net,focal_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            _, regression, classification, anchors = net(images)
            loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets, cuda=cuda)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_c_loss += c_loss.item()
            
            pbar.set_postfix(**{'Conf Loss'         : total_c_loss / (iteration+1), 
                                'Regression Loss'   : total_r_loss / (iteration+1), 
                                'lr'                : get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                _, regression, classification, anchors = net(images_val)
                loss, c_loss, r_loss = focal_loss(classification, regression, anchors, targets_val, cuda=cuda)
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    #-------------------------------------------#
    phi = 0
    #-------------------------------------------#
    #   根据phi的值选择输入图片的大小
    #-------------------------------------------#
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    input_shape = (input_sizes[phi], input_sizes[phi])
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #----------------------------------------------------#
    #   classes的路径非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'   
    #----------------------------------------------------#
    #   获取classes
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    #------------------------------------------------------#
    #   创建EfficientDet模型
    #   训练前一定要修改classes_path和对应的txt文件
    #------------------------------------------------------#
    model = EfficientDetBackbone(num_classes,phi)

    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = "model_data/efficientdet-d0.pth"
    print('Loading weights into state dict...')
    model_dict = model.state_dict() 
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    efficient_loss = FocalLoss()

    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 50
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)
        val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=efficientdet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=efficientdet_dataset_collate)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,efficient_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)

    if True:
        #--------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        #--------------------------------------------#
        lr = 1e-4
        Batch_size = 4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)
        val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=efficientdet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=efficientdet_dataset_collate)

                        
        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            val_loss = fit_one_epoch(net,efficient_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
