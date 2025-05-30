# standard dependencies
from itertools import product as product
from math import ceil
import os

# 3rd-party dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.models as models
from deepface.models.Detector import FacialAreaRegion

# internal dependencies
from utilities import io_utils


class RetinaFace:
    def __init__(
            self,
            checkpoint: str = 'retinaface.pth',
            device: torch.device = None,
            conf_thresh: float = 0.02,
            nms_thresh: float = 0.4,
            top_k: int = 5000,
            keep_top_k: int = 750,
            min_sizes: list = [[16, 32], [64, 128], [256, 512]],
            steps: list = [8, 16, 32],
            variance: list = [0.1, 0.2],
            clip: bool = False,
            fp16: bool = False,
        ):
        project_root = io_utils.get_project_root()
        self.checkpoint_path = os.path.join(
            project_root, 'models/weights/', checkpoint
        )
        self.fp16 = fp16
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model = RetinaFaceModel()
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        if self.fp16:
            self.model = self.model.half()

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.variance = variance
    
    def get_priors(self, image_size: tuple[int]):
        image_size = image_size
        feature_maps = [
            [ceil(image_size[0]/step), ceil(image_size[1]/step)]
            for step in self.steps
        ]

        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [
                        x * self.steps[k] / image_size[1]
                        for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / image_size[0]
                        for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)

        return output.to(self.device)

    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """

        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def decode_landm(self, pre, priors, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), dim=1)
        return landms

    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def preprocess(self, img):
        img = img.copy()

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        return img.to(self.device)

    def detect(self, img):
        img = np.float32(img)
        img = self.preprocess(img)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        ).to(self.device)
        resize = 1

        loc, conf, landms = self.model(img)

        priors = self.get_priors(image_size=(im_height, im_width))
        prior_data = priors.data

        boxes = self.decode(loc.data.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = self.decode_landm(landms.data.squeeze(0), prior_data, self.variance)
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, self.nms_thresh)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def detect_faces(self, img: np.ndarray) -> list[FacialAreaRegion]:
        '''
        Detect and align faces with RetinaFace.

        Args:
            img (np.ndarray): Pre-loaded image as numpy array.

        Returns:
            List[FacialAreaRegion]: A list of FacialAreaRegion objects.
        '''
        results = []
        detections = self.detect(img)

        for det in detections:
            x1, y1, x2, y2 = det[0:4]
            confidence = det[4]
            w = x2 - x1
            h = y2 - y1
            x, y = int(x1), int(y1)

            left_eye = tuple(map(int, det[5:7]))
            right_eye = tuple(map(int, det[7:9]))
            nose = tuple(map(int, det[9:11]))
            mouth_left = tuple(map(int, det[11:13]))
            mouth_right = tuple(map(int, det[13:15]))

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=int(w),
                h=int(h),
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                mouth_left=mouth_left,
                mouth_right=mouth_right,
                confidence=float(confidence)
            )
            results.append(facial_area)

        return results



# =============================================================================
#                      - OVERALL MODEL ARCHITECTURE -
# -----------------------------------------------------------------------------


class RetinaFaceModel(nn.Module):
    def __init__(
            self,      
            return_layers: dict = {'layer2': 1, 'layer3': 2, 'layer4': 3},
            in_channels: int = 256,
            out_channels: int = 256,
        ):
        super(RetinaFaceModel, self).__init__()
            
        backbone = models.resnet50(pretrained=True)

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = in_channels
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )

        output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            self.conv_bn(3, 8, 2, leaky = 0.1),    # 3
            self.conv_dw(8, 16, 1),   # 7
            self.conv_dw(16, 32, 2),  # 11
            self.conv_dw(32, 32, 1),  # 19
            self.conv_dw(32, 64, 2),  # 27
            self.conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            self.conv_dw(64, 128, 2),  # 43 + 16 = 59
            self.conv_dw(128, 128, 1), # 59 + 32 = 91
            self.conv_dw(128, 128, 1), # 91 + 32 = 123
            self.conv_dw(128, 128, 1), # 123 + 32 = 155
            self.conv_dw(128, 128, 1), # 155 + 32 = 187
            self.conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            self.conv_dw(128, 256, 2), # 219 +3 2 = 241
            self.conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def conv_bn(self, inp, oup, stride=1, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def conv_dw(self, inp, oup, stride, leaky=0.1):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.LeakyReLU(negative_slope= leaky,inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope= leaky,inplace=True),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


# =============================================================================
#                       - ARCHITECTURE COMPONENTS -
# -----------------------------------------------------------------------------


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        shared_args = {'oup': out_channels, 'stride': 1, 'leaky': leaky}

        self.output1 = self.conv_bn1X1(in_channels_list[0], **shared_args)
        self.output2 = self.conv_bn1X1(in_channels_list[1], **shared_args)
        self.output3 = self.conv_bn1X1(in_channels_list[2], **shared_args)

        self.merge1 = self.conv_bn(out_channels, **shared_args)
        self.merge2 = self.conv_bn(out_channels, **shared_args)

    def conv_bn1X1(self, inp, oup, stride, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def conv_bn(self, inp, oup, stride=1, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = self.conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = self.conv_bn(in_channel, out_channel//4, stride=1, leaky=leaky)
        self.conv5X5_2 = self.conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = self.conv_bn(out_channel//4, out_channel//4, stride=1, leaky=leaky)
        self.conv7x7_3 = self.conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def conv_bn(self, inp, oup, stride=1, leaky=0):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )

    def conv_bn_no_relu(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)
