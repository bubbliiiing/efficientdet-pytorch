import torch
from nets.efficientnet import EfficientNet
from nets.efficientdet import EfficientDetBackbone

if __name__ == '__main__':

    inputs = torch.randn(5, 3, 512, 512)

    # Test EfficientNet
    model = EfficientNet.from_name('efficientnet-b0')
    inputs = torch.randn(4, 3, 512, 512)
    P = model(inputs)
    for idx, p in enumerate(P):
        print('P{}: {}'.format(idx, p.size()))
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))

    # # print('model: ', model)

    # Test inference
    model = EfficientDetBackbone(80,0)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    