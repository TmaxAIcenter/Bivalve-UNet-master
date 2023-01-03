import torch
from model import UNet as _UNet

def Unet_Bivalve(pretrained=False, dataset_type="FU"):
    
    net = _UNet(n_channels=3, n_classes=5, bilinear=False)
    if pretrained:
        if dataset_type == "FU":
            checkpoint = 'https://github.com/TmaxAIcenter/Bivalve-UNet-master/releases/download/untagged-cc8aa95d00d1d8e5e206/FU_Unet_Checkpoint_epoch200.pth'
        elif dataset_type == "MA":
            checkpoint = 'https://github.com/TmaxAIcenter/Bivalve-UNet-master/releases/download/untagged-cc8aa95d00d1d8e5e206/MA_Unet_Checkpoint_epoch200.pth'
        else:
            raise RuntimeError('Only FU and MA dataset_type are available')

        net.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))

    return net

