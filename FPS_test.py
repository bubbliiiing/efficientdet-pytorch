import colorsys
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from tqdm import tqdm

from efficientdet import EfficientDet
from nets.efficientdet import EfficientDetBackbone
from utils.utils import (bbox_iou, decodebox, efficientdet_correct_boxes,
                         letterbox_image, non_max_suppression)

image_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

def preprocess_input(image):
    image /= 255
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image
'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''
class FPS_EfficientDet(EfficientDet):
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (image_sizes[self.phi], image_sizes[self.phi])))
        photo = np.array(crop_img,dtype = np.float32)
        photo = np.transpose(preprocess_input(photo), (2, 0, 1))

        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()
                
            _, regression, classification, anchors = self.net(images)
            
            regression = decodebox(regression, anchors, images)
            detection = torch.cat([regression,classification],axis=-1)
            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:,4] > self.confidence
                top_conf = batch_detections[top_index,4]
                top_label = np.array(batch_detections[top_index,-1], np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)
            except:
                pass 
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                _, regression, classification, anchors = self.net(images)
                
                regression = decodebox(regression, anchors, images)
                detection = torch.cat([regression,classification],axis=-1)
                batch_detections = non_max_suppression(detection, len(self.class_names),
                                                        conf_thres=self.confidence,
                                                        nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:,4] > self.confidence
                    top_conf = batch_detections[top_index,4]
                    top_label = np.array(batch_detections[top_index,-1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index,:4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                    boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)
                except:
                    pass 

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
efficientdet = FPS_EfficientDet()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = efficientdet.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
