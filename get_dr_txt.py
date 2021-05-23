#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from efficientdet import EfficientDet
from utils.utils import (decodebox, efficientdet_correct_boxes,
                         letterbox_image, non_max_suppression)

image_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
'''
这里设置的门限值较低是因为计算map需要用到不同门限条件下的Recall和Precision值。
所以只有保留的框足够多，计算的map才会更精确，详情可以了解map的原理。
计算map时输出的Recall和Precision值指的是门限为0.5时的Recall和Precision值。

此处获得的./input/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。

这里的self.iou指的是非极大抑制所用到的iou，具体的可以了解非极大抑制的原理，
如果低分框与高分框的iou大于这里设定的self.iou，那么该低分框将会被剔除。

可能有些同学知道有0.5和0.5:0.95的mAP，这里的self.iou=0.5不代表mAP0.5。
如果想要设定mAP0.x，比如设定mAP0.75，可以去get_map.py设定MINOVERLAP。
'''
def preprocess_input(image):
    image /= 255
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image
    
class mAP_EfficientDet(EfficientDet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        self.iou = 0.5
        f = open("./input/detection-results/"+image_id+".txt","w") 
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

            #---------------------------------------------------------#
            #   传入网络当中进行预测
            #---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)
            
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            regression = decodebox(regression, anchors, images)
            detection = torch.cat([regression,classification],axis=-1)
            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return 
                
            #-----------------------------------------------------------#
            #   筛选出其中得分高于confidence的框 
            #-----------------------------------------------------------#
            top_index = batch_detections[:,4] > self.confidence
            top_conf = batch_detections[top_index,4]
            top_label = np.array(batch_detections[top_index,-1], np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------#
            #   去掉灰条部分
            #-----------------------------------------------------------#
            boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

efficientdet = mAP_EfficientDet()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    efficientdet.detect_image(image_id,image)
    
print("Conversion completed!")
