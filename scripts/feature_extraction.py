import os
import pandas as pd
import polars as pl
import numpy as np
from sklearn.utils import shuffle
from collections import Counter

import torchreid
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype
from torchvision.io import read_image
import torchvision.models as models
import h5py
import cv2
from utilities import read_detection_csv


def write_embeddings_hdf5(embeddings, video, detection_data, stride=1):
    frames = []
    box_indices = []
    for frame in sorted(detection_data.keys()):
        detections = detection_data[frame]
        for i in range(len(detections)):
            frames.append(frame)
            box_indices.append(i)
    frames = np.array(frames)
    box_indices = np.array(box_indices)

    embeddings_array = torch.stack([emb.detach().cpu() for emb in embeddings]).numpy()
    with h5py.File(f'../intermediate_output/s{stride}_{video}_embeddings.hdf5', 'w') as file:
        file.create_dataset('embeddings', data=embeddings_array)
        file.create_dataset('frames', data=frames)
        file.create_dataset('box_indices', data=box_indices)


# state_dict = torch.load('finetuned_osnet.pth')
checkpoint = torch.load('model.pth.tar-250')
state_dict = checkpoint['state_dict']

new_state_dict = {}
for key in state_dict.keys():
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = state_dict[key]

finetuned = torchreid.models.osnet.osnet_x1_0(num_classes=751, pretrained=False,
                                              loss='triplet')
finetuned.load_state_dict(new_state_dict)
finetuned.to(device)
finetuned.eval()
print("Done")

#@title Class Similarities


def cos_sim(embedding1, embedding2):
    embedding1 = embedding1.unsqueeze(0) if embedding1.dim() == 1 else embedding1
    embedding2 = embedding2.unsqueeze(0) if embedding2.dim() == 1 else embedding2
    sim_tensor = F.cosine_similarity(embedding1, embedding2, dim=1)
    return sim_tensor.item()


location = 'CP_Sacramento'
timestamp = '2024-08-12_08:35:57'
video = f'{location}_{timestamp}_2'

detection_data = read_detection_csv(f'../intermediate_output/s{stride}_{video}_detections.csv')
frames = sorted(detection_data.keys())
start = frames[0]
end = frames[-1]

cap = cv2.VideoCapture(f'../input_files/{video}.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
fh, fw = 1080, 1920

frame_number = start

embeddings = []
with torch.no_grad():
    while frame_number <= end:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number in detection_data.keys():
            for box in detection_data[frame_number]:
                x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
                cropped = frame[y1:y2, x1:x2]
                try:
                    image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (128, 256))
                    image = image.astype(np.float32)
                    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                    image = image.to(device)
                    embeddings.append(finetuned(image))
                except:
                    print(x1, y1, x2, y2)


        if ((frame_number - start) % 450) == 0:
            print(f'{(frame_number - start) // 30} seconds processed')
        frame_number += 1
cap.release()

write_embeddings_hdf5(embeddings, video, detection_data, stride=stride)

def get_embeddings_by_frame_and_box(file_name, target_frame):
    with h5py.File(file_name, 'r') as file:
        frames = file['frames']
        indices = np.where(frames[:] == target_frame)[0]
        print(indices)
        target_embeddings = file['embeddings'][sorted(indices)]
        return target_embeddings

# Example usage
video = 'cam0'
file_name = f'../output_files/{video}_embeddings.hdf5'
target_frame = 9852
f_emb = get_embeddings_by_frame_and_box(file_name, target_frame)

cos_sim(torch.Tensor(embeddings[0]).to(device), torch.Tensor(f_emb).to(device))

cos_sim(torch.Tensor(f_emb[0]), torch.Tensor(f_emb[1]))