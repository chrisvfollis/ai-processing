# standard dependencies
from collections import OrderedDict
import functools
import os

# 3rd-party dependencies
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.transforms as transforms

# internal dependencies
import utilities.general_utils as utils
from utilities import io_utils


class ClearFace:
    def __init__(
            self,
            device: torch.device = None,
            checkpoint: str = '90000_G.pth',
            in_nc: int = 3,
            out_nc: int = 3,
            nf: int = 64,
            nb: int = 16
        ):
        self.device = device or utils.get_default_device()

        project_root = io_utils.get_project_root()
        self.weights_path = os.path.join(
            project_root, 'models/weights/clearface/', checkpoint
        )

        self.netG = RRDBNet(
            in_nc=in_nc, out_nc=out_nc, nf=nf,nb=nb
        ).to(self.device)

        self.netG = DataParallel(self.netG)
        self.load()

        self.fwd_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        ])

    def load(self):        
        load_net = torch.load(self.weights_path, map_location=self.device)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        if (
            (isinstance(self.netG, nn.DataParallel)) or
            (isinstance(self.netG, DistributedDataParallel))
        ):
            self.netG = self.netG.module

        self.netG.load_state_dict(load_net_clean, strict=True)

    def forward(self, img, is_rgb=False):
        if len(img.shape) != 3:
            return None
        
        if not is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img_tensor = torch.unsqueeze(self.fwd_transform(img), 0).to(self.device)
        
        self.netG.eval()
        with torch.no_grad():
            output_tensor = self.netG(img_tensor)

        output_img = np.transpose(
            output_tensor.squeeze(0).cpu().numpy(), (1, 2, 0)   
        ) # (C, H, W) --> (H, W, C)

        output_img = np.clip(
            (output_img / 2.0 + 0.5) * 255.0, 0, 255
        ).astype(np.uint8)
        
        return output_img


# ----------------------------------------------------------------------------
# RRDBNet Model Architecture:


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        '''
        Args:
            in_nc (int): Number of input channels
            out_nc (int): Number of output channels
            nf (int): Number of feature maps (filters) in the convolutional
                layers (defines model width).
            nb (int): Number of RRDB blocks in the trunk (controls network
                depth).
            gc (int): Number of growth channels per convolution inside
                ResidualDenseBlock_5C.
        '''
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        # self.tanh = nn.Tanh()

    def make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        # out_tanh = self.tanh(out)

        return out


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.initialize_weights([self.conv1, self.conv2, self.conv3,
                                 self.conv4, self.conv5], 0.1)

    def initialize_weights(self, net_l, scale=1):
        if not isinstance(net_l, list):
            net_l = [net_l]
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale  # for residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x
