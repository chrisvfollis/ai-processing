# standard dependencies
import os
from collections import deque
import gc
import warnings
from typing import Optional
import math

# 3rd-party dependencies
import numpy as np
import h5py
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

warnings.filterwarnings(
    "ignore", message="Cython evaluation",
    module="torchreid.reid.metrics.rank"
)
from torchreid import models as reid

# internal dependencies
from utilities.log_utils import press_stopwatch
from utilities import io_utils
import utilities.general_utils as utils


class OSNet:
    def __init__(
            self,
            weights_file: str = 'OSNet.pth.tar-250',
            device: torch.device = None,
            input_dims: tuple[int, int] = (128, 256),
            output_shape: tuple[int] = (512,),
            num_classes: int = None,
            loss: str = 'triplet',
            buffer_limit: int = 100,
            mode: str = 'eval'
        ):
        self.project_root = io_utils.get_project_root()
        self.weights_path = os.path.join(
            self.project_root, 'models/weights/', weights_file
        )
        self.device = device or utils.get_default_device()

        checkpoint = torch.load(self.weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']

        if not num_classes:
            for key in state_dict:
                if key.endswith('classifier.weight'):
                    classifier_weight_shape = state_dict[key].shape
                    num_classes = classifier_weight_shape[0]
                    break
            if num_classes is None:
                raise ValueError('classifier.weight not found in checkpoint')

        self.model = reid.osnet.osnet_x1_0(
            num_classes=num_classes,
            loss=loss,
            pretrained=False,
        )

        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')

            if (mode == 'train') and (
                ('classifier.weight' in new_key) or
                ('classifier.bias' in new_key)
            ):
                continue

            new_state_dict[new_key] = state_dict[key]
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)

        if mode == 'eval':
            self.eval_mode()
        elif mode == 'train':
            self.train_mode(num_classes)

        self.input_dims = input_dims
        self.output_shape = output_shape
        self.n_features = output_shape[0]

        self.buffer_limit = buffer_limit
        self.buffer_type = None
        self.embedding_idx = 0

    def eval_mode(self):
        self.model.eval()

        self.preprocess_time = 0
        self.embedding_time = 0
        self.flush_time = 0

    def train_mode(self, num_classes):
        self.model.train()

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # if num_classes < 100:   # freeze early layers
        #     for name, param in self.model.named_parameters():
        #         if 'layer3' not in name and 'layer4' not in name:
        #             param.requires_grad = False

    def preprocess(
            self, input_data: np.ndarray | list[np.ndarray]
        ) -> torch.Tensor:
        '''
        Args:
            input_data: A single image (HWC ndarray) or a list of such images.

        Returns:
            A 4D tensor of shape (B, C, H, W), where B is the batch size (which
            will be 1 if a single image is passed).
        '''
        def _preprocess_image(image: np.array) -> torch.Tensor:
            image = cv2.resize(image, self.input_dims)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = image / 255.0

            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
            )

            image_tensor = TF.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            return image_tensor

        if not input_data:
            raise ValueError('No images to process in batch')

        press_stopwatch(self, 'preprocess_time')

        if isinstance(input_data, list):
            preprocessed_imgs = [_preprocess_image(img) for img in input_data]
            image_tensor = torch.stack(preprocessed_imgs)
        else:
            image_tensor = _preprocess_image(input_data).unsqueeze(0)
        
        press_stopwatch(self, 'preprocess_time')

        return image_tensor.to(self.device)

    def postprocess(
            self, output_data, batched=False
        ) -> np.ndarray | list[np.ndarray]:
        if not batched:
            postprocessed = output_data.cpu().detach().numpy().flatten() 
        else:
            postprocessed = [
                output.cpu().detach().numpy().flatten()
                for output in output_data
            ]
        return postprocessed
        
    def extract_features(self, image):
        image_tensor = self.preprocess(image)

        press_stopwatch(self, 'embedding_time')
        with torch.no_grad():
            output = self.model(image_tensor)
        press_stopwatch(self, 'embedding_time')

        embedding = self.postprocess(output)

        if self.buffer_type:
            self.update_buffers(
                embedding,
                self.embedding_idx,
                structure=self.buffer_type
            )
        self.embedding_idx += 1
        return embedding

    @staticmethod
    def _safe_crop(img, box_xywh):
        """Clamp box to image; return None if it becomes empty."""
        x, y, w, h = [int(round(v.item() if torch.is_tensor(v) else v)) for v in box_xywh]
        H, W = img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def extraction_batch(self, img, detections, f_num):                
        batch_images, kept = [], []
        for i, box in enumerate(detections):
            if math.prod(box[2:4]) < 80**2:
                continue
            crop = self._safe_crop(img, box[:4])
            if crop is None:
                continue
            batch_images.append(crop)
            kept.append(i)
        
        if not batch_images:
            return

        batch_tensor = self.preprocess(batch_images)

        press_stopwatch(self, 'embedding_time')
        with torch.no_grad():
            batch_output = self.model(batch_tensor)
        press_stopwatch(self, 'embedding_time')

        embeddings = self.postprocess(batch_output, batched=True)
        self.update_buffers(
            embeddings,
            index=f_num,
            box_indices=kept,
            structure='video_data'
        )

    @property
    def active_buffers(self):
        buffer_attrs = [
            'embedding_buffer',
            'index_buffer',
            'frame_buffer',
            'box_index_buffer',
        ]
        return [getattr(self, b) for b in buffer_attrs if hasattr(self, b)]

    def activate_buffers(
            self,
            file_prefix: str,
            structure: str = 'standard',
            buffer_limit: int = None,
        ):
        '''
        Sets up the appropriate buffer attributes for the given output structure,
        and creates a corresponding HDF5 file for dumping the buffered data.
        '''
        self.buffer_type = structure
        self.buffer_limit = buffer_limit or self.buffer_limit

        # set up buffer output file:
        output_dir = os.path.join(self.project_root, 'files/output/')

        filename = file_prefix + '_embeddings.hdf5'
        unique_filename = io_utils.get_unique_filename(output_dir, filename)

        self.output_path = os.path.join(output_dir, unique_filename)
        self.hdf5_file = h5py.File(self.output_path, 'a')

        index_data_configs = {
            'standard': {
                'buffer_attrs': ['index_buffer'],
                'datasets': ['indices'],
            },
            'video_data': {
                'buffer_attrs': ['frame_buffer', 'box_index_buffer'],
                'datasets': ['frames', 'box_indices'],
            },
        }
        index_data_config = index_data_configs[structure]

        # set up buffer attributes:
        self.embedding_buffer = deque(maxlen=None)
        for index_buffer in index_data_config['buffer_attrs']:
            setattr(self, index_buffer, deque(maxlen=None))

        embeddings_dataset_kwargs = {
            'shape': (0, self.n_features),
            'maxshape': (None, self.n_features),
        }
        index_dataset_kwargs = {
            'shape': (0,),
            'dtype': 'i',
            'maxshape': (None,)
        }
        
        # structure buffer output file:
        self.hdf5_file.create_dataset('embeddings', **embeddings_dataset_kwargs)
        for index_dataset in index_data_config['datasets']:
            self.hdf5_file.create_dataset(index_dataset, **index_dataset_kwargs)

    def update_buffers(
            self, embedding_data: np.ndarray | list[np.ndarray],
            index: int, box_indices: list[int] = None, structure: str = None
        ) -> None:
        structure = structure or self.buffer_type

        if isinstance(embedding_data, np.ndarray):
            num_embeddings = 1
        elif isinstance(embedding_data, list):
            num_embeddings = len(embedding_data)

        if (len(self.embedding_buffer) + num_embeddings) >= self.buffer_limit:
            self.flush_buffers(structure=structure)

        if structure == 'standard':
            self.embedding_buffer.append(embedding_data)
            self.index_buffer.append(index)
        elif structure == 'video_data':
            self.embedding_buffer.extend(embedding_data)
            self.frame_buffer.extend([index] * num_embeddings)
            self.box_index_buffer.extend(box_indices)

    def flush_buffers(self, structure=None, release=False):
        structure = structure or self.buffer_type

        press_stopwatch(self, 'flush_time')

        if len(self.embedding_buffer) > 0:
            self.write_embeddings(structure=structure)
    
        for buffer in self.active_buffers:
            buffer.clear()
        if release:
            self.release_buffers()
            self.hdf5_file = h5py.File(self.output_path, 'a')

        torch.cuda.empty_cache()
        gc.collect()
        
        press_stopwatch(self, 'flush_time')
    
    def release_buffers(self):
        self.hdf5_file.flush()
        self.hdf5_file.close()
        del self.hdf5_file

        torch.cuda.empty_cache()
        gc.collect()

    def write_embeddings(
            self,
            structure: Optional[str] = None,
            hdf5_file: Optional[h5py.File] = None,
            embeddings: Optional[np.ndarray] = None,
            indices: Optional[np.ndarray] = None, 
        ):
        """
        Writes embedding data out to an HDF5 file.

        Args:
            structure (str): Indicates how the data should be organized
                for the HDF5 file. Options: 'standard', 'video_data'.

            hdf5_file (h5py.File): The file object to write to. If left
                unspecified, the self.hdf5_file attribute will be used instead.

            embeddings (np.ndarray): A numpy array of embeddings. If left
                unspecified, the self.embedding_buffer will be used instead.

            indices (np.ndarray): A numpy array of index values corresponding to
                the embeddings. If left unspecified, the self.index_buffer will
                be used instead.
        """
        structure = structure or self.buffer_type

        hdf5_file = hdf5_file or getattr(self, 'hdf5_file')
        if not hdf5_file:
            print('HDF5 file not found')
            return

        # organize data into appropriate structure:
        if structure == 'standard':
            new_embeddings = embeddings or np.array(self.embedding_buffer)
            new_idxs = indices or np.array(self.index_buffer)

            all_new_data = {
                'embeddings': new_embeddings,
                'indices': new_idxs,
            }
        elif structure == 'video_data':
            new_embeddings = np.stack(self.embedding_buffer)
            new_frames = np.array(self.frame_buffer)
            new_box_idxs = np.array(self.box_index_buffer)

            all_new_data = {
                'embeddings': new_embeddings,
                'frames': new_frames,
                'box_indices': new_box_idxs,
            }

        # write data to hdf5 file:    
        for dataset_name, new_data in all_new_data.items():
            dataset = hdf5_file[dataset_name]

            current_total, num_new = (
                dataset.shape[0],
                new_data.shape[0]
            )
            new_total = current_total + num_new

            dataset.resize(new_total, axis=0)
            dataset[-new_data.shape[0]:] = new_data
