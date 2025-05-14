# standard dependencies
import os
from typing import Union, Optional
import re

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import h5py

# internal dependencies
from models import OSNet
from utilities import io_utils
from utilities import admin_utils


# =============================================================================
#                          - EVENT IMAGE TESTS -
# -----------------------------------------------------------------------------

def event_img_extraction(
        shop_id: Optional[str] = None,
        weights_file: str = 'OSNet.pth.tar-250'
    ):
    project_root = io_utils.get_project_root()

    img_dir_path = os.path.join(project_root, 'files/output/event_imgs/')
    weights_path = os.path.join(project_root, 'models/weights/', weights_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    osnet = OSNet(weights_path, device)
    osnet.activate_buffers(
        'event_imgs',
        structure='standard',
        output_dir=os.path.join(project_root, 'files/output/')
    )
    embeddings_filepath = osnet.output_path

    img_data_df = admin_utils.save_approved_img_data(shop_id=shop_id)

    try:
        for image in img_data_df['image']:
            image_path = os.path.join(img_dir_path, image)
            image = cv2.imread(image_path)

            osnet.extract_features(image)
    finally:
        if len(osnet.embedding_buffer) > 0:
            osnet.flush_buffers(structure='standard', release=True)
        else:
            osnet.release_buffers()

    return embeddings_filepath, img_data_df


def event_img_embedding_distances(
        embeddings_filepath: str,
        img_data_df: Union[pd.DataFrame, str],
        chunk_size: int = 100
    ) -> pd.DataFrame:
    def _structure_entry(img_1, img_2, distance):
        entry_data = {
            'image1': img_1['image'],
            'image2': img_2['image'],
            'person_id1': img_1['person_id'],
            'person_id2': img_2['person_id'],
            'shop_id1': img_1['shop_id'],
            'shop_id2': img_2['shop_id'],
            'first_name1': img_1['first_name'],
            'last_name1': img_1['last_name'],
            'first_name2': img_2['first_name'],
            'last_name2': img_2['last_name'],
            'start_time1': img_1['start_time'],
            'start_time2': img_2['start_time'],
            'distance': distance,
        }
        return entry_data

    project_root = io_utils.get_project_root()
    distances_spreadsheet_path = os.path.join(
        project_root, 'files/output/', 'cos_distances_data.xlsx'
    )

    if isinstance(img_data_df, str):
        img_data_df = pd.read_excel(img_data_df)

    distance_data = []
    
    with h5py.File(embeddings_filepath, 'r') as f:
        num_embeddings = f['embeddings'].shape[0]
        indices = f['indices'][:]

        metadata = []
        for idx in indices:
            row = img_data_df.iloc[idx]

            metadata.append({
                k: row[k] for k in [
                    'image',
                    'person_id',
                    'shop_id',
                    'first_name',
                    'last_name',
                    'start_time',
                ]
            })

        for start_idx in range(0, num_embeddings, chunk_size):
            end_idx = min(start_idx + chunk_size, num_embeddings)

            current_chunk_np = f['embeddings'][start_idx:end_idx]
            current_chunk = torch.from_numpy(current_chunk_np).float()

            for i in range(len(current_chunk)):
                for j in range(i + 1, len(current_chunk)):
                    sim = F.cosine_similarity(
                        current_chunk[i].unsqueeze(0),
                        current_chunk[j].unsqueeze(0)
                    ).item()
                    distance = 1 - sim

                    img_1 = metadata[start_idx + i]
                    img_2 = metadata[start_idx + j]

                    entry_values = _structure_entry(img_1, img_2, distance)
                    distance_data.append(entry_values)

            for next_idx in range(end_idx, num_embeddings):
                next_embedding_np = f['embeddings'][next_idx]

                next_embedding = (
                    torch.from_numpy(next_embedding_np).float()
                    .unsqueeze(0)
                )

                sims = F.cosine_similarity(current_chunk, next_embedding, dim=1)
                distances = 1 - sims

                for i in range(len(current_chunk)):
                    distance = distances[i].item()

                    img_1 = metadata[start_idx + i]
                    img_2 = metadata[next_idx]

                    entry_values = _structure_entry(img_1, img_2, distance)
                    distance_data.append(entry_values)

    distances_df = pd.DataFrame(distance_data)

    with pd.ExcelWriter(distances_spreadsheet_path, engine='xlsxwriter') as writer:
        distances_df.to_excel(
            writer, sheet_name='Cosine Distances', index=False
        )

    return distances_df


# =============================================================================
#                          - MARKET 1501 TESTS -
# -----------------------------------------------------------------------------

def load_market1501_metadata(dataset_dir: str = 'market-1501'):
    project_root = io_utils.get_project_root()
    dataset_dir_path = os.path.join(project_root, 'files/datasets/', dataset_dir)

    all_images = []
    image_dirs = ['bounding_box_test', 'query']
    
    for subdir in image_dirs:
        full_dir = os.path.join(dataset_dir_path, subdir)
        for filename in os.listdir(full_dir):
            if not filename.endswith('.jpg'):
                continue
            match = re.match(r'([-\d]+)_c(\d)s\d+_\d+_\d+\.jpg', filename)
            if not match:
                continue
            pid, camid = match.groups()
            all_images.append({
                'image': filename,
                'person_id': int(pid),
                'cam_id': int(camid),
                'path': os.path.join(full_dir, filename),
            })

    return pd.DataFrame(all_images)


def market1501_extraction(df, weights_file='OSNet.pth.tar-250'):
    project_root = io_utils.get_project_root()

    weights_path = os.path.join(project_root, 'models/weights/', weights_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    osnet = OSNet(weights_path, device)
    osnet.activate_buffers(
        'market-1501',
        structure='standard',
        output_dir=os.path.join(project_root, 'files/output/')
    )

    try:
        for image_path in df['path']:
            image = cv2.imread(image_path)
            osnet.extract_features(image)
    finally:
        if len(osnet.embedding_buffer) > 0:
            osnet.flush_buffers(structure='standard', release=True)
        else:
            osnet.release_buffers()

    return osnet.output_path


def market1501_embedding_distances(embeddings_filepath, img_data_df, chunk_size=100):
    def _structure_entry(img_1, img_2, distance):
        return {
            'image1': img_1['image'],
            'image2': img_2['image'],
            'person_id1': img_1['person_id'],
            'person_id2': img_2['person_id'],
            'cam_id1': img_1['cam_id'],
            'cam_id2': img_2['cam_id'],
            'distance': distance,
        }

    distance_data = []

    with h5py.File(embeddings_filepath, 'r') as f:
        num_embeddings = f['embeddings'].shape[0]
        indices = f['indices'][:]

        metadata = [img_data_df.iloc[idx] for idx in indices]

        for start_idx in range(0, num_embeddings, chunk_size):
            end_idx = min(start_idx + chunk_size, num_embeddings)
            current_chunk_np = f['embeddings'][start_idx:end_idx]
            current_chunk = torch.from_numpy(current_chunk_np).float()

            for i in range(len(current_chunk)):
                for j in range(i + 1, len(current_chunk)):
                    sim = F.cosine_similarity(
                        current_chunk[i].unsqueeze(0),
                        current_chunk[j].unsqueeze(0)
                    ).item()
                    distance = 1 - sim

                    img_1 = metadata[start_idx + i]
                    img_2 = metadata[start_idx + j]
                    distance_data.append(_structure_entry(img_1, img_2, distance))

            for next_idx in range(end_idx, num_embeddings):
                next_embedding_np = f['embeddings'][next_idx]
                next_embedding = torch.from_numpy(next_embedding_np).float().unsqueeze(0)

                sims = F.cosine_similarity(current_chunk, next_embedding, dim=1)
                distances = 1 - sims

                for i in range(len(current_chunk)):
                    distance = distances[i].item()
                    img_1 = metadata[start_idx + i]
                    img_2 = metadata[next_idx]
                    distance_data.append(_structure_entry(img_1, img_2, distance))

    return pd.DataFrame(distance_data)
