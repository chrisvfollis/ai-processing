# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) 2025 Ivakt Vision Inc. All rights reserved.

# standard dependencies
import math
import os
from typing import Optional

# 3rd-party dependencies
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision

# internal dependencies
from utilities import io_utils, log_utils
from utilities.log_utils import press_stopwatch
import utilities.general_utils as utils


logger = log_utils.get_logger(__name__)


class YoloX:
    def __init__(
        self,
        checkpoint: str = 'yolox_mot17.pth.tar',
        num_classes: int = 1,
        depth: float = 1.33,
        width: float = 1.25,
        input_size: tuple[int] = (800, 1440),
        conf_thresh: float = 0.1,
        nms_thresh: float = 0.7,
        device: torch.device = None,
        fp16: bool = True,
        use_trt: bool = False,
        decode: bool = True,
    ):
        def _configure_batchnorm(model):
            '''
            Adjust BatchNorm2d layers to use YOLOX-specific epsilon and
            momentum values instead of PyTorch defaults.

            YOLOX models are designed with:
            - eps = 1e-3 (pytorch default is 1e-5)
            - momentum = 0.03 (pytorch default is 0.1)
            '''
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        def _fuse_model():
            '''
            Fuses the convolutional and batchnorm layers for faster inference.
            '''
            def _fuse_conv_and_bn(conv, bn):
                # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
                fusedconv = (
                    nn.Conv2d(
                        conv.in_channels,
                        conv.out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        groups=conv.groups,
                        bias=True,
                    )
                    .requires_grad_(False)
                    .to(conv.weight.device)
                )
                # prepare filters
                w_conv = conv.weight.clone().view(conv.out_channels, -1)
                w_bn = torch.diag(bn.weight.div(
                    torch.sqrt(bn.eps + bn.running_var)
                ))
                fusedconv.weight.copy_(
                    torch.mm(w_bn, w_conv)
                    .view(fusedconv.weight.shape)
                )
                # prepare spatial bias
                b_conv = (
                    torch.zeros(conv.weight.size(0), device=conv.weight.device)
                    if conv.bias is None
                    else conv.bias
                )
                b_bn = bn.bias - (bn.weight.mul(bn.running_mean).div(
                    torch.sqrt(bn.running_var + bn.eps)
                ))
                fusedconv.bias.copy_(
                    torch.mm(w_bn, b_conv.reshape(-1, 1))
                    .reshape(-1) + b_bn
                )
                return fusedconv
            
            logger.info('Fusing model...')
            for m in self.model.modules():
                if type(m) is BaseConv and hasattr(m, "bn"):
                    m.conv = _fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.fuseforward  # update forward

        self.device = device or utils.get_default_device()
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # LOAD MODEL:
        self.use_trt = use_trt
        if self.use_trt:
            self.fp16 = True    # override fp16 arg
            from torch2trt import TRTModule
            self.model = TRTModule()
            self.model.to(self.device)

            trt_path = os.path.join(
                io_utils.get_project_root(), 'models/weights/yolox/', checkpoint
            )
            sd = torch.load(trt_path, map_location=self.device)
            self.model.load_state_dict(sd)

            self.decoder = YoloXTRTDecoder(
                input_size=self.input_size,
                num_classes=self.num_classes,
                strides=[8, 16, 32]
            )

            self.model.eval()
        else:
            backbone = YOLOPAFPN(depth, width, in_channels=[256, 512, 1024])
            head = YOLOXHead(num_classes, width, in_channels=[256, 512, 1024], input_size=input_size, decode=decode)
            self.model = YOLOX(backbone, head)

            self.model.apply(_configure_batchnorm)
            self.model.head.initialize_biases(1e-2)

            self.model.to(self.device)

            checkpoint_path = os.path.join(
                io_utils.get_project_root(), 'models/weights/yolox/', checkpoint
            )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

            self.model.eval()
            _fuse_model()

            if self.fp16:
                self.model = self.model.half()

        # TIMING ATTRS:
        self.preprocess_time = 0
        self.postprocess_time = 0
        self.inference_time = 0
        
    def inference(
            self,
            img_data: list[np.ndarray] | np.ndarray,
            conf_thresh: Optional[float] = None,
            nms_thresh: Optional[float] = None,
            num_classes: Optional[int] = None,
        ) -> list[torch.tensor | None]:
        '''
        Returns (list[torch.tensor or None]): List of tensors, one for each
            image (or None if there were no detections in that image). Each
            tensor is of shape [num_detections, 7], where each detection
            is as follows:
                [x1, y1, x2, y2, object_conf, class_conf, class_pred]
        '''
        conf_thresh = conf_thresh or self.conf_thresh
        nms_thresh = nms_thresh or self.nms_thresh
        num_classes = num_classes or self.num_classes

        if isinstance(img_data, np.ndarray):
            img_data = [img_data]

        img_data = [img.copy() for img in img_data]
        img_tensor = self.preprocess(
            img_data, self.input_size, self.rgb_means, self.std
        )
        if self.fp16:
            img_tensor = img_tensor.half()
        with torch.no_grad():
            press_stopwatch(self, 'inference_time')
            outputs = self.model(img_tensor)
            press_stopwatch(self, 'inference_time')

            if self.use_trt:
                outputs = self.decoder.decode_outputs(
                    outputs, dtype=outputs[0].dtype
                )
            outputs = self.postprocess(
                outputs,
                conf_thresh,
                nms_thresh,
                num_classes,
            )
            
        return outputs

    def preprocess(self, images, input_size, mean, std, swap=(2, 0, 1)):
        '''
        Rescales, pads, and normalizes input images. The image's aspect ratio
        is preserved through its rescaling, such that it fits within a canvas of
        shape `input_size` which is used to pad it.
        '''
        press_stopwatch(self, 'preprocess_time')
        preprocessed_images = []

        for image in images:
            if len(image.shape) == 3:
                padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
            else:
                padded_img = np.ones(input_size) * 114.0
            img = np.array(image)
            r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

            padded_img = padded_img[:, :, ::-1]
            padded_img /= 255.0
            if mean is not None:
                padded_img -= mean
            if std is not None:
                padded_img /= std
            padded_img = padded_img.transpose(swap)
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

            preprocessed_images.append(padded_img)

        preprocessed_images = np.stack(preprocessed_images, axis=0)
        preprocessed_images = torch.from_numpy(preprocessed_images).to(self.device)

        press_stopwatch(self, 'preprocess_time')
        return preprocessed_images

    def postprocess(self, prediction, conf_thresh, nms_thresh, num_classes):
        press_stopwatch(self, 'postprocess_time')
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                continue

            class_conf, class_pred = torch.max(
                image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
            )

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thresh).squeeze()

            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            
            if not detections.size(0):
                continue
            if detections.shape[1] == 1:
                detections = detections.squeeze(0)

            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thresh,
            )

            detections = detections[nms_out_index]

            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        press_stopwatch(self, 'postprocess_time')
        return output


# =============================================================================
#                      - OVERALL MODEL ARCHITECTURE -
# -----------------------------------------------------------------------------


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
        ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            input_size=(800, 1440),
            act="silu",
            depthwise=False,
            decode=True,
        ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.hw = [(input_size[0] // s, input_size[1] // s) for s in strides]
        self.decode = decode

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, xin)
            ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)


            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

            outputs.append(output)

        # [batch, n_anchors_all, 85]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)

        if self.decode:
            return self.decode_outputs(outputs, dtype=xin[0].dtype)
        else:
            return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        device = outputs.device

        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype).to(device)
        strides = torch.cat(strides, dim=1).type(dtype).to(device)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        return outputs


# =============================================================================
#                       - ARCHITECTURE COMPONENTS -
# -----------------------------------------------------------------------------


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


# =============================================================================
#                            - TRT DECODING -
# -----------------------------------------------------------------------------


class YoloXTRTDecoder:
    def __init__(self, input_size, num_classes, strides=[8, 16, 32]):
        self.num_classes = num_classes
        self.strides = strides
        self.hw = [(input_size[0] // s, input_size[1] // s) for s in strides]

    def decode_outputs(self, outputs, dtype):
        device = outputs.device
        grids = []
        strides = []

        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype).to(device)
        strides = torch.cat(strides, dim=1).type(dtype).to(device)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
