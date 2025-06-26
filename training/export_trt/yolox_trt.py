# standard dependencies
import os

# 3rd-party dependencies
import torch
import tensorrt as trt
from torch2trt import torch2trt

# internal dependencies
from models import YoloX
from utilities import io_utils


def main(
        checkpoint='yolox_mot17.pth.tar',
        input_size=(800, 1440),
        num_classes=1,
        depth=1.33,
        width=1.25,
):
    output_dir = os.path.join(
        io_utils.get_project_root(), 'models/weights/yolox/'
    )

    model = YoloX(
        checkpoint=checkpoint,
        num_classes=num_classes,
        depth=depth,
        width=width,
        input_size=input_size,
        use_trt=False,
        fp16=True,
        decode=False,
    ).model

    model.eval().cuda()

    dummy_input = torch.ones(
        1, 3, input_size[0], input_size[1],
        dtype=torch.float16
    ).cuda()

    model_trt = torch2trt(
        model,
        [dummy_input],
        fp16_mode=True,
        max_batch_size=20,
        max_workspace_size=(1 << 33),
        log_level=trt.Logger.INFO,
        opt_shapes=[
            (1, 3, 800, 1440),
            (8, 3, 800, 1440),
            (16, 3, 800, 1440),
            (20, 3, 800, 1440),
        ],
    )

    del model
    torch.cuda.empty_cache()

    trt_pth = io_utils.get_unique_path(output_dir, 'yolox_model_trt.pth')
    torch.save(model_trt.state_dict(), trt_pth)
    print(f'Saved TensorRT .pth to {trt_pth}')

    engine_file = io_utils.get_unique_path(output_dir, 'yolox_model_trt.engine')
    with open(engine_file, 'wb') as f:
        f.write(model_trt.engine.serialize())
    print(f'Saved TensorRT engine to {engine_file}')


if __name__ == '__main__':
    main()
