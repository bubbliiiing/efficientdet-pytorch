## Efficientdet：Scalable and Efficient Object目标检测模型在Pytorch当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [注意事项 Attention](#注意事项)
4. [训练步骤 How2train](#训练步骤)
5. [参考资料 Reference](#Reference)

### 所需环境
torch==1.2.0

### 文件下载  
训练所需的pth可以在百度网盘下载。       
包括Efficient-d0到d7所有权重。    
链接: https://pan.baidu.com/s/1Kvv526YYSDJEf9BzWfIb3Q 提取码: f9g3  

### 注意事项
**1、训练前一定要注意权重文件与Efficientdet版本的对齐！**  
**2、注意修改训练用到的voc_classes.txt文件！**  
**3、注意修改预测用到的voc_classes.txt文件！**  

### 训练步骤
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4、在训练前利用voc2efficientdet.py文件生成对应的txt。  
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。  
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。  
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。  
8、修改train.py文件下的phi可以修改efficientdet的版本，训练前注意权重文件与Efficientdet版本的对齐。  
9、运行train.py即可开始训练。  

### mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

### Reference
https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch   
https://github.com/Cartucho/mAP
