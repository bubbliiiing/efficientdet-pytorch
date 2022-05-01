#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nets.efficientdet import EfficientDetBackbone
from utils.utils import image_sizes

if __name__ == '__main__':
    phi             = 0
    input_shape     = [image_sizes[phi], image_sizes[phi]]
    num_classes     = 80
    
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model           = EfficientDetBackbone(num_classes, phi).to(device)
    print(model)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print("Total GFLOPs: %s" %(flops))
    print("Total Parameters: %s" %(params))
