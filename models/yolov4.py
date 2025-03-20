# standard dependencies
import sys
import time

# 3rd-party dependencies
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F

# internal dependencies
pass


class YOLOv4:
    def __init__(self, weights_path, device, nms_thresh=0.5, conf_thresh=0.65,
                 input_dims=(416, 416)):
        self.device = device

        self.model = Yolov4Model(inference=True)
        weights = torch.load(weights_path, map_location=device)

        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.input_dims = input_dims

        self.preprocess_time = 0
        self.detection_time = 0
        self.postprocess_time = 0
        
    def detect(self, img, class_id, nms_thresh=None, conf_thresh=None,
               input_dims=None):
        def _preprocess_img(img, input_dims):
            '''
            input_dims — the width and height to resize the image to. YOLOv4
            only accepts image dimensions that can be expressed using the
            formula (320 + (96 * n)), where n is a positive integer.
        
            Examples of valid dimensions include 320, 416, 512, 608, etc
            '''
            start_preprocess = time.perf_counter()

            self.input_dims = input_dims
            original_dims = img.shape[:2][::-1]
            w, h = input_dims

            img_resized = cv2.resize(img, (w, h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            img_tensor = (torch.from_numpy(img_rgb.transpose(2, 0, 1))
                          .float().div(255.0).unsqueeze(0))
    
            if self.device.type == 'cuda':
                img_tensor = img_tensor.cuda()
            
            end_preprocess = time.perf_counter()
            self.preprocess_time += (end_preprocess - start_preprocess)
    
            return img_tensor, original_dims
        
        def _postprocess_output(output, nms_thresh, conf_thresh, class_id,
                                original_dims):
            def _filter_output(output, nms_thresh, conf_thresh, class_id):
                def _apply_nms(boxes, confs, nms_thresh):
                    x1 = boxes[:, 0]
                    y1 = boxes[:, 1]
                    x2 = boxes[:, 2]
                    y2 = boxes[:, 3]

                    areas = (x2 - x1) * (y2 - y1)
                    order = confs.argsort()[::-1]

                    keep = []
                    while order.size > 0:
                        idx_self = order[0]
                        idx_other = order[1:]

                        keep.append(idx_self)

                        xx1 = np.maximum(x1[idx_self], x1[idx_other])
                        yy1 = np.maximum(y1[idx_self], y1[idx_other])
                        xx2 = np.minimum(x2[idx_self], x2[idx_other])
                        yy2 = np.minimum(y2[idx_self], y2[idx_other])

                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)

                        inter = w * h
                        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

                        inds = np.where(over <= nms_thresh)[0]
                        order = order[inds + 1]
                    
                    return np.array(keep)
                
                box_array = output[0][0]            # (n_detections, 1, 4)
                confs = output[1][0]                # (n_detections, n_classes)

                if type(box_array).__name__ != 'ndarray':
                    box_array = box_array.cpu().detach().numpy()
                    confs = confs.cpu().detach().numpy()

                box_array = box_array[:, 0]         # (n_detections, 4)
                
                max_conf = np.max(confs, axis=1)    # (n_detections,)
                max_id = np.argmax(confs, axis=1)   # (n_detections,)

                # Filter by confidence threshold and class ID:
                argwhere = (max_conf > conf_thresh) & (max_id == class_id)
                filtered_max_conf = max_conf[argwhere]
                filtered_boxes = box_array[argwhere, :]
                
                if filtered_boxes.shape[0] > 0:
                    keep = _apply_nms(filtered_boxes, filtered_max_conf,
                                      nms_thresh)
                    filtered_boxes = filtered_boxes[keep, :]
                    filtered_max_conf = filtered_max_conf[keep]

                bboxes = []
                for k in range(filtered_boxes.shape[0]):
                    bboxes.append([
                        filtered_boxes[k, 0], filtered_boxes[k, 1],
                        filtered_boxes[k, 2], filtered_boxes[k, 3],
                        filtered_max_conf[k]
                    ])
                
                return bboxes

            def _translate_detection(box, original_dims):
                x1, y1, x2, y2 = box
                img_w, img_h = original_dims

                scale_x = img_w
                scale_y = img_h

                x1 = int(round(x1 * scale_x))
                y1 = int(round(y1 * scale_y))
                x2 = int(round(x2 * scale_x))
                y2 = int(round(y2 * scale_y))

                x1 = int(max(0, min(x1, img_w)))
                y1 = int(max(0, min(y1, img_h)))
                x2 = int(max(0, min(x2, img_w)))
                y2 = int(max(0, min(y2, img_h)))

                w = x2 - x1
                h = y2 - y1

                return [x1, y1, w, h]
            
            start_postprocess = time.perf_counter()

            filtered_output = _filter_output(
                output, nms_thresh, conf_thresh, class_id
            )

            final_output = []
            for detection in filtered_output:
                bbox, conf = detection[:4], float(detection[4])

                x, y, w, h = _translate_detection(
                    bbox, original_dims
                )
                final_output.append([x, y, w, h, conf])

            end_postprocess = time.perf_counter()
            self.postprocess_time += (end_postprocess - start_postprocess)
            
            return final_output
        
        nms_thresh = nms_thresh or self.nms_thresh
        conf_thresh = conf_thresh or self.conf_thresh
        input_dims = input_dims or self.input_dims

        img, original_dims = _preprocess_img(img, input_dims)

        start_detect = time.perf_counter()
        with torch.no_grad():
            raw_output = self.model(img)
        end_detect = time.perf_counter()

        self.detection_time += (end_detect - start_detect)
        
        results = _postprocess_output(
            raw_output, nms_thresh, conf_thresh, class_id, original_dims
        )
        
        return results


# ----------------------------------------------------------------------------


# Overall Model Architecture:


class Yolov4Model(nn.Module):
    def __init__(self, n_classes=80, inference=True):
        super().__init__()
        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()

        # neck
        self.neck = Neck(inference=inference)
        
        # head
        self.head = Yolov4Head(output_ch, n_classes, inference=inference)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output


# ----------------------------------------------------------------------------


# Model Architecture Components:


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target_size, inference=False):
        target_h = target_size[2]
        target_w = target_size[3]

        input_h = x.size(2)
        input_w = x.size(3)

        batch_size = x.size(0)
        num_channels = x.size(1)

        if inference:
            reshaped_tensor = (
                x.view(batch_size, num_channels, input_h, 1, input_w, 1)
                .expand(
                    batch_size, num_channels,
                    input_h, (target_h // input_h),
                    input_w, (target_w // input_w)
                )
                .contiguous()
                .view(batch_size, num_channels, target_h, target_w)
            )
            return reshaped_tensor

        else:
            rescaled_tensor = F.interpolate(
                x, size=(target_h, target_w), mode='nearest'
            )
            return rescaled_tensor


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=True):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=True):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super().__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output):
        def _dynamic_flexible_fwd(output, anchors):
            # Output would be invalid if it does not satisfy this assert
            # assert (output.size(1) == (5 + num_classes) * num_anchors)

            # print(output.size())

            # Slice the second dimension (channel) of output into:
            # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
            # And then into
            # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
            # batch = output.size(0)
            # H = output.size(2)
            # W = output.size(3)

            bxy_list = []
            bwh_list = []
            det_confs_list = []
            cls_confs_list = []

            num_anchors = len(self.anchor_mask)

            for i in range(num_anchors):
                begin = i * (5 + self.num_classes)
                end = (i + 1) * (5 + self.num_classes)
                
                bxy_list.append(output[:, begin : begin + 2])
                bwh_list.append(output[:, begin + 2 : begin + 4])
                det_confs_list.append(output[:, begin + 4 : begin + 5])
                cls_confs_list.append(output[:, begin + 5 : end])

            # Shape: [batch, num_anchors * 2, H, W]
            bxy = torch.cat(bxy_list, dim=1)
            # Shape: [batch, num_anchors * 2, H, W]
            bwh = torch.cat(bwh_list, dim=1)

            # Shape: [batch, num_anchors, H, W]
            det_confs = torch.cat(det_confs_list, dim=1)
            # Shape: [batch, num_anchors * H * W]
            det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

            # Shape: [batch, num_anchors * num_classes, H, W]
            cls_confs = torch.cat(cls_confs_list, dim=1)
            # Shape: [batch, num_anchors, num_classes, H * W]
            cls_confs = cls_confs.view(output.size(0), num_anchors, self.num_classes, output.size(2) * output.size(3))
            # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
            cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(output.size(0), num_anchors * output.size(2) * output.size(3), self.num_classes)

            # Apply sigmoid(), exp() and softmax() to slices
            #
            bxy = torch.sigmoid(bxy) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)
            bwh = torch.exp(bwh)
            det_confs = torch.sigmoid(det_confs)
            cls_confs = torch.sigmoid(cls_confs)

            # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
            grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0)
            grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0)
            # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
            # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

            anchor_w = []
            anchor_h = []
            for i in range(num_anchors):
                anchor_w.append(anchors[i * 2])
                anchor_h.append(anchors[i * 2 + 1])

            device = None
            cuda_check = output.is_cuda
            if cuda_check:
                device = output.get_device()

            bx_list = []
            by_list = []
            bw_list = []
            bh_list = []

            # Apply C-x, C-y, P-w, P-h
            for i in range(num_anchors):
                ii = i * 2
                # Shape: [batch, 1, H, W]
                bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
                # Shape: [batch, 1, H, W]
                by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
                # Shape: [batch, 1, H, W]
                bw = bwh[:, ii : ii + 1] * anchor_w[i]
                # Shape: [batch, 1, H, W]
                bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

                bx_list.append(bx)
                by_list.append(by)
                bw_list.append(bw)
                bh_list.append(bh)


            ########################################
            #   Figure out bboxes from slices     #
            ########################################
            
            # Shape: [batch, num_anchors, H, W]
            bx = torch.cat(bx_list, dim=1)
            # Shape: [batch, num_anchors, H, W]
            by = torch.cat(by_list, dim=1)
            # Shape: [batch, num_anchors, H, W]
            bw = torch.cat(bw_list, dim=1)
            # Shape: [batch, num_anchors, H, W]
            bh = torch.cat(bh_list, dim=1)

            # Shape: [batch, 2 * num_anchors, H, W]
            bx_bw = torch.cat((bx, bw), dim=1)
            # Shape: [batch, 2 * num_anchors, H, W]
            by_bh = torch.cat((by, bh), dim=1)

            # normalize coordinates to [0, 1]
            bx_bw /= output.size(3)
            by_bh /= output.size(2)

            # Shape: [batch, num_anchors * H * W, 1]
            bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
            by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
            bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
            bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)

            bx1 = bx - bw * 0.5
            by1 = by - bh * 0.5
            bx2 = bx1 + bw
            by2 = by1 + bh

            # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
            boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4)
            # boxes = boxes.repeat(1, 1, num_classes, 1)

            # boxes:     [batch, num_anchors * H * W, 1, 4]
            # cls_confs: [batch, num_anchors * H * W, num_classes]
            # det_confs: [batch, num_anchors * H * W]

            det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
            confs = cls_confs * det_confs

            # boxes: [batch, num_anchors * H * W, 1, 4]
            # confs: [batch, num_anchors * H * W, num_classes]

            return  boxes, confs

        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return _dynamic_flexible_fwd(output, masked_anchors)


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=True):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
            anchor_mask=[0, 1, 2], num_classes=n_classes,
            anchors=[
                12, 16, 19, 36, 40, 28, 36, 75, 76, 55,
                72, 146, 142, 110, 192, 243, 459, 401
            ],
            num_anchors=9, stride=8
        )

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
            anchor_mask=[3, 4, 5], num_classes=n_classes,
            anchors=[
                12, 16, 19, 36, 40, 28, 36, 75, 76, 55,
                72, 146, 142, 110, 192, 243, 459, 401
            ],
            num_anchors=9, stride=16
        )

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
            anchor_mask=[6, 7, 8], num_classes=n_classes,
            anchors=[
                12, 16, 19, 36, 40, 28, 36, 75, 76, 55,
                72, 146, 142, 110, 192, 243, 459, 401
            ],
            num_anchors=9, stride=32
        )

    def forward(self, input1, input2, input3):
        def _get_region_boxes(boxes_and_confs):

            boxes_list = []
            confs_list = []

            for item in boxes_and_confs:
                boxes_list.append(item[0])
                confs_list.append(item[1])

            # boxes: [batch, num1 + num2 + num3, 1, 4]
            # confs: [batch, num1 + num2 + num3, num_classes]
            boxes = torch.cat(boxes_list, dim=1)
            confs = torch.cat(confs_list, dim=1)
                
            return [boxes, confs]

        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)
    
            return _get_region_boxes([y1, y2, y3])

        else:
            return [x2, x10, x18]
