#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
from nets.efficientdet import EfficientDetBackbone

if __name__ == '__main__':
    model   = EfficientDetBackbone(80,0)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
