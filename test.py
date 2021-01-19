#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import torch

from nets.efficientdet import EfficientDetBackbone
from nets.efficientnet import EfficientNet

if __name__ == '__main__':
    inputs = torch.randn(4, 3, 512, 512)
    model = EfficientDetBackbone(80,0)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
