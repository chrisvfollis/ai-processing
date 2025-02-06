import torchreid
import torch
import numpy as np
import cv2
import time
import os
import h5py
from utilities import io_utils
import time

import warnings
warnings.filterwarnings(
    "ignore",
    message="Cython evaluation",
    module="torchreid.reid.metrics.rank"
)
import torchreid


class OSNet:
    def __init__(self, weights_path, device, input_shape=(128, 256),
                 output_shape=(512,), num_classes=751, loss='triplet'):
        self.device = device

        self.model = torchreid.models.osnet.osnet_x1_0(
            num_classes=num_classes, pretrained=False, loss='triplet'
        )
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.extraction_time = 0
        self.flush_time = 0

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = state_dict[key]
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, img):
        def _preprocess_img(img):
            w, h = self.input_shape

            img_resized = cv2.resize(img, (w, h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32)

            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            return img_tensor
        
        def _postprocess_output(output):
            return output.cpu().detach().numpy().flatten()
        
        start_extract = time.perf_counter()

        img = _preprocess_img(img)
        with torch.no_grad():
            output = self.model(img)

        embedding = _postprocess_output(output)
        
        end_extract = time.perf_counter()
        self.extraction_time += (end_extract - start_extract)

        return embedding

    def enable_buffers(self, video_file, output_dir='../intermediate_output',
                       buffer_limit=100):
        '''
        Sets up buffers and an output file so the OSNet instance can use
        bulk processing features like extraction batches.
        '''
        
        hdf5_file = video_file.split('.')[0] + '_embeddings.hdf5'
        output_path = os.path.join(output_dir, hdf5_file)
        if os.path.exists(output_path):
            os.remove(output_path)

        self.hdf5_file = h5py.File(output_path, 'a')

        self.embedding_buffer = []
        self.frame_buffer = []
        self.box_index_buffer = []

        self.buffer_limit = buffer_limit

        n_features = self.output_shape[0]
        idx_dataset_kwargs = {'shape': (0,), 'dtype': 'i', 'maxshape': (None,)}

        self.hdf5_file.create_dataset(
            'embeddings', (0, n_features), maxshape=(None, n_features)
        )
        self.hdf5_file.create_dataset('frames', **idx_dataset_kwargs)
        self.hdf5_file.create_dataset('box_indices', **idx_dataset_kwargs)

    def flush_buffers(self, close_file=False):
        start_flush = time.perf_counter()

        io_utils.write_embeddings(
            self.hdf5_file, self.embedding_buffer, self.frame_buffer,
            self.box_index_buffer
        )
        self.embedding_buffer.clear()
        self.frame_buffer.clear()
        self.box_index_buffer.clear()

        if close_file == True:
            self.hdf5_file.close()
        
        end_flush = time.perf_counter()
        self.flush_time += (end_flush - start_flush)

    def extraction_batch(self, img, detections, f_num):
        def _update_buffers(embedding, f_num, box_idx):
            self.embedding_buffer.append(embedding)
            self.frame_buffer.append(f_num)
            self.box_index_buffer.append(box_idx)

        for i, box in enumerate(detections):
            x, y, w, h, = box[:4]
            img_cropped = img[y:y+h, x:x+w]
            try:
                embedding = self.extract_features(img_cropped)
            except Exception as e:
                print(f"Error processing bounding box: {e}")
                embedding = []

            _update_buffers(embedding, f_num, i)
        
        if self.buffer_limit <= len(self.embedding_buffer):
            self.flush_buffers()
