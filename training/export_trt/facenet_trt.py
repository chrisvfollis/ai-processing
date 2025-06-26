import os
import torch
from torch import nn
from torch.nn import functional as F
import tensorrt as trt
from torch2trt import torch2trt

from models import FaceNet512
from utilities import io_utils


class InceptionResnetV1Export(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.conv2d_1a(x)
        x = self.base_model.conv2d_2a(x)
        x = self.base_model.conv2d_2b(x)
        x = self.base_model.maxpool_3a(x)
        x = self.base_model.conv2d_3b(x)
        x = self.base_model.conv2d_4a(x)
        x = self.base_model.conv2d_4b(x)
        x = self.base_model.repeat_1(x)
        x = self.base_model.mixed_6a(x)
        x = self.base_model.repeat_2(x)
        x = self.base_model.mixed_7a(x)
        x = self.base_model.repeat_3(x)
        x = self.base_model.block8(x)
        x = self.base_model.avgpool_1a(x)
        x = self.base_model.dropout(x)
        x = self.base_model.last_linear(x.view(x.shape[0], -1))
        # skip last_bn because TensorRT has no 1D BatchNorm
        x = F.normalize(x, p=2, dim=1)
        return x


def main(
        checkpoint='facenet512.pth',
        input_size=(160, 160),
        fp16=False,
):
    output_dir = os.path.join(
        io_utils.get_project_root(), 'models/weights/facenet/'
    )

    model = FaceNet512(
        checkpoint=checkpoint,
        fp16=fp16,
    ).model
    export_model = InceptionResnetV1Export(model)

    export_model.eval().cuda()

    dummy_input = torch.ones(
        1, 3, input_size[0], input_size[1],
        dtype=torch.float16 if fp16 else torch.float32
    ).cuda()

    model_trt = torch2trt(
        export_model,
        [dummy_input],
        fp16_mode=fp16,
        max_batch_size=16,
        max_workspace_size=(1 << 33),
        log_level=trt.Logger.INFO,
        min_shapes=[(1, 3, 160, 160)],
        opt_shapes=[(1, 3, 160, 160)],
        max_shapes=[(16, 3, 160, 160)],
    )

    del export_model
    torch.cuda.empty_cache()

    trt_pth = io_utils.get_unique_path(output_dir, 'facenet512_model_trt.pth')
    torch.save(model_trt.state_dict(), trt_pth)
    print(f'Saved TensorRT .pth to {trt_pth}')

    engine_file = io_utils.get_unique_path(output_dir, 'facenet512_model_trt.engine')
    with open(engine_file, 'wb') as f:
        f.write(model_trt.engine.serialize())
    print(f'Saved TensorRT engine to {engine_file}')


if __name__ == '__main__':
    main()
