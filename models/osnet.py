import torch
import numpy as np
import cv2
import time
import os
import h5py
from utilities import io_utils
import time
from collections import deque
import torch
import gc

import warnings
warnings.filterwarnings(
    "ignore",
    message="Cython evaluation",
    module="torchreid.reid.metrics.rank"
)
from torchreid import models as reid


class OSNet:
    def __init__(self, weights_path, device, input_dims=(128, 256),
                 output_shape=(512,), num_classes=751, loss='triplet',
                 buffer_limit=100):
        self.device = device

        self.model = reid.osnet.osnet_x1_0(num_classes=num_classes,
                                           pretrained=False, loss=loss)

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = state_dict[key]
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.input_dims = input_dims
        self.output_shape = output_shape

        self.buffer_limit = buffer_limit

        self.preprocess_time = 0
        self.embedding_time = 0
        self.flush_time = 0
    
    def extract_features(self, image):
        def _preprocess_img(image):
            start_preprocess = time.perf_counter()

            image = cv2.resize(image, self.input_dims)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            end_preprocess = time.perf_counter()
            self.preprocess_time += (end_preprocess - start_preprocess)

            return image_tensor
        
        def _postprocess_output(output):
            return output.cpu().detach().numpy().flatten()
        
        image_tensor = _preprocess_img(image)

        start_extract = time.perf_counter()
        with torch.no_grad():
            output = self.model(image_tensor)
        
        end_extract = time.perf_counter()
        self.embedding_time += (end_extract - start_extract)

        embedding = _postprocess_output(output)
        
        return embedding

    def extraction_batch(self, img, detections, f_num):
        def _preprocess(batch_images):
            start_preprocess = time.perf_counter()

            processed_imgs = []
            for image in batch_images:
                image = cv2.resize(image, self.input_dims)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32)

                image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                processed_imgs.append(image_tensor)
            
            batch_tensor = torch.stack(processed_imgs).to(self.device)

            end_preprocess = time.perf_counter()
            self.preprocess_time += (end_preprocess - start_preprocess)
            
            return batch_tensor
        
        def _postprocess(outputs):
            return [output.cpu().detach().numpy().flatten() for output in outputs]

        def _update_buffers(embeddings, f_num, num_detections):
            if len(self.embedding_buffer) >= self.buffer_limit:
                self.flush_buffers()

            self.embedding_buffer.extend(embeddings)
            self.frame_buffer.extend([f_num] * num_detections)
            self.box_index_buffer.extend(list(range(num_detections)))
        
        batch_images = []

        for box in detections:
            x, y, w, h, = box[:4]
            batch_images.append(img[y:y+h, x:x+w])

        batch_tensor = _preprocess(batch_images)

        start_extract = time.perf_counter()
        with torch.no_grad():
            batch_output = self.model(batch_tensor)
        end_extract = time.perf_counter()
        self.embedding_time += (end_extract - start_extract)

        embeddings = _postprocess(batch_output)
        _update_buffers(embeddings, f_num, len(embeddings))

    def activate_buffers(self, video_file, output_dir='../files/output',
                         buffer_limit=None):
        self.output_path = os.path.join(
            output_dir, video_file.split('.')[0] + '_embeddings.hdf5'
        )
        self.hdf5_file = h5py.File(self.output_path, 'a')

        buffer_limit = buffer_limit if buffer_limit else self.buffer_limit

        self.embedding_buffer = deque(maxlen=buffer_limit)
        self.frame_buffer = deque(maxlen=buffer_limit)
        self.box_index_buffer = deque(maxlen=buffer_limit)

        n_features = self.output_shape[0]
        idx_dataset_kwargs = {'shape': (0,), 'dtype': 'i', 'maxshape': (None,)}

        self.hdf5_file.create_dataset(
            'embeddings', (0, n_features), maxshape=(None, n_features)
        )
        self.hdf5_file.create_dataset('frames', **idx_dataset_kwargs)
        self.hdf5_file.create_dataset('box_indices', **idx_dataset_kwargs)

    def flush_buffers(self, release=False):
        start_flush = time.perf_counter()

        io_utils.write_embeddings(
            self.hdf5_file, self.embedding_buffer, self.frame_buffer,
            self.box_index_buffer
        )
        self.embedding_buffer.clear()
        self.frame_buffer.clear()
        self.box_index_buffer.clear()

        if release:
            self.release_buffers()
            self.hdf5_file = h5py.File(self.output_path, 'a')

        torch.cuda.empty_cache()
        gc.collect()
        
        end_flush = time.perf_counter()
        self.flush_time += (end_flush - start_flush)
    
    def release_buffers(self):
        self.hdf5_file.flush()
        self.hdf5_file.close()
        del self.hdf5_file

        torch.cuda.empty_cache()
        gc.collect()
