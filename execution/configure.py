# standard dependencies
import argparse

# 3rd-party dependencies
pass

# internal dependencies
pass


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-level', type=int, default=0)

    parser.add_argument('--retain-footage', action='store_true', default=False)
    parser.add_argument('--save-all-data', action='store_true', default=False)

    parser.add_argument('--start-from', type=str, help='Comma-separated datetime')
    parser.add_argument('--priority-cam', type=str)
    parser.add_argument('--f-cutoff', type=int, default=None)

    parser.add_argument('--id-strategy', type=str, default='assess_presence')

    return parser


def package_model_cfgs():
    yolox_cfg = {
        'checkpoint'  : 'yolox_model_trt.pth',
        'num_classes' : 1,
        'depth'       : 1.33,
        'width'       : 1.25,
        'input_size'  : (800, 1440),
        'conf_thresh' : 0.05,
        'nms_thresh'  : 0.7,
        'fp16'        : True,
        'use_trt'     : True,
    }
    osnet_cfg = {}
    faces_cfg = {
        'facenet_cfg': {
            'embedding_size' : 512,
            'checkpoint'     : 'facenet512_model_trt.pth',
            'fp16'           : False,
            'use_trt'        : True,
        },
        'centerface_cfg': {
            # 'conf_thresh' : 0.40,
            'conf_thresh' : 0.35,
            'min_area'    : (16, 16),
        },
        # 'clearface_cfg': {
        #     'checkpoint': '90000_G.pth',
        # }
    }
    
    return {
        'yolox': yolox_cfg,
        'osnet': osnet_cfg,
        'faces': faces_cfg,
    }