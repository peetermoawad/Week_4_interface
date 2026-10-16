"""
model.py  —  Water Segmentation Model Definition
=================================================
Best model from Weeks 1-2: SMP U-Net / ResNet-34 / weight_avg
Val IoU: 0.8253  |  Input: 12-channel Sentinel-2 TIF (128x128)
"""
import torch
import torch.nn as nn
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

INPUT_CHANNELS = 12
IMAGE_SIZE     = 128
GREEN_CH, NIR_CH, SWIR_CH = 2, 7, 10


def adapt_first_layer(model, in_channels, strategy='weight_avg'):
    first_conv = first_name = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            first_conv, first_name = mod, name
            break
    assert first_conv is not None
    old_w = first_conv.weight.data.clone()
    if old_w.shape[1] == in_channels:
        return model
    assert old_w.shape[1] == 3
    out_ch, _, kH, kW = old_w.shape
    if strategy == 'weight_avg':
        mean_w = old_w.mean(dim=1, keepdim=True)
        std_w  = old_w.std(dim=1, keepdim=True)
        noise  = torch.randn(out_ch, in_channels, kH, kW, dtype=old_w.dtype)
        new_w  = (mean_w + std_w * noise) * (3.0 / in_channels)
    elif strategy == 'zero_init':
        new_w = torch.zeros(out_ch, in_channels, kH, kW, dtype=old_w.dtype)
        new_w[:, :3] = old_w
    else:
        raise ValueError(f'Unknown strategy: {strategy}')
    new_conv = nn.Conv2d(in_channels, out_ch,
                         kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride,
                         padding=first_conv.padding,
                         bias=first_conv.bias is not None)
    new_conv.weight = nn.Parameter(new_w)
    if first_conv.bias is not None:
        new_conv.bias = nn.Parameter(first_conv.bias.data.clone())
    parts  = first_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)
    return model


def build_model(in_channels=INPUT_CHANNELS, strategy='weight_avg'):
    assert smp is not None, 'segmentation_models_pytorch not installed'
    model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                     in_channels=3, classes=1, activation=None)
    return adapt_first_layer(model, in_channels, strategy)


def load_weights(model, weights_path, device):
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
