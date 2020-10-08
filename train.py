# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import efficientdet_dataset_collate, EfficientdetDataset
from nets.efficientdet import EfficientDetBackbone
from nets.efficientdet_training import Generator, FocalLoss
from nets.multibox_loss import MultiBoxLoss
from nets.repulsion_loss import RepulsionLoss
from tqdm import tqdm

from functools import wraps
from datetime import datetime

train_type = 0   #  1 for train



init_model_path = './logs/Epoch50-Total_Loss0.6522-Val_Loss0.6001.pth'
if not init_model_path:
    init_model_path = "./weights/efficientdet-d0.pth"

loss = 'F'

loss_type = {
    'F' : FocalLoss,
    'M': MultiBoxLoss,       #num_classes
    'R': RepulsionLoss
}

# test_loss = RepulsionLoss()

criteria = loss_type[loss]

def _curent_time():
    date = datetime.now()
    return date.strftime("%Y%m%d_%H-%M-%S")


def time_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        begin = datetime.now()
        res = func(*args, **kwargs)
        after = datetime.now()
        print('===time cost: {} costs {}'.format(func.__name__, after - begin))
        return res

    return wrapper


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


# @time_log
def fit_one_epoch(model, optimizer, net, criteria_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_r_loss = 0
    total_c_loss = 0
    total_repu_loss = 0
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
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

            loss, c_loss, r_loss, repu_loss = criteria_loss(classification, regression, anchors, targets, cuda=cuda)
            # rep_loss = test_loss(classification, regression, anchors, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_r_loss += r_loss.item()
            total_c_loss += c_loss.item()
            total_repu_loss += repu_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Total Loss' : total_loss / (iteration + 1),
                                'Conf Loss': total_c_loss / (iteration + 1),
                                'Regression Loss': total_r_loss / (iteration + 1),
                                'Repulsion Loss': total_repu_loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'time/s': waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in
                                   targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                _, regression, classification, anchors = net(images_val)
                loss, c_loss, r_loss = criteria_loss(classification, regression, anchors, targets_val, cuda=cuda)
                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    return val_loss / (epoch_size_val + 1)


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#


@time_log  # modified  #注释掉才可以运行成功
def train():
    # -------------------------------------------#
    #   训练前，请指定好phi和model_path
    #   二者所使用Efficientdet版本要相同
    # -------------------------------------------#
    lr = 1e-3
    phi = 0
    Cuda = True
    annotation_path = '2007_train.txt'
    classes_path = 'model_data/voc_classes.txt'
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True

    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536]
    input_shape = (input_sizes[phi], input_sizes[phi])  # TODO Input picture size need adjust
    # 4000*2250  ->  512*512
    # 500 * 2250/8
    # 创建模型
    model = EfficientDetBackbone(num_classes, phi)

    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#

    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(init_model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    efficient_loss = criteria()         # TODO loss: repulsive loss

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#

        Batch_size = 4
        Init_Epoch = 0
        Freeze_Epoch = 25

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)  # adam  SGD
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        if Use_Data_Loader:
            train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
            val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            val_loss = fit_one_epoch(model, optimizer, net, efficient_loss, epoch, epoch_size, epoch_size_val, gen,
                                     gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step(val_loss)
            # TODO every epoch: precision and recall

    if True:
        # --------------------------------------------#
        #   BATCH_SIZE不要太小，不然训练效果很差
        # --------------------------------------------#
        lr = lr/10
        Batch_size = 2  #
        Freeze_Epoch = 25
        Unfreeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

        if Use_Data_Loader:
            train_dataset = EfficientdetDataset(lines[:num_train], (input_shape[0], input_shape[1]))
            val_dataset = EfficientdetDataset(lines[num_train:], (input_shape[0], input_shape[1]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=efficientdet_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=efficientdet_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone_net.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            val_loss = fit_one_epoch(model, optimizer, net, efficient_loss, epoch, epoch_size, epoch_size_val, gen,
                                     gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step(val_loss)


def predict(model_path):
    from efficientdet import EfficientDet
    from PIL import Image
    efficientdet = EfficientDet(model_path)
    # img = "./VOCdevkit/VOC2007/JPEGImages/notag/bike{}.JPG"  # input('Input image filename:')   #随便
    img = "./VOCdevkit/VOC2007/JPEGImages/tagbike{}.JPG"
    while True:
        I = input("Input a number:\n")
        try:
            image = Image.open(img.format(I))
        except:
            if I == 'q':
                break
            print('Open Error! Try again!')

        else:
            r_image = efficientdet.detect_image(image)
            r_image.show()


if __name__ == '__main__':
    # model_path = './logs/Epoch50-Total_Loss0.6522-Val_Loss0.6001.pth'  # hardcore, assign to Durbin
    if train_type:
        train()
    else:
        #reducing learning rate of group 0 to 6.2500e-06.
        predict(init_model_path)
