# standard dependencies
import os
import sys
from typing import Union, Optional
import uuid

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import h5py

# internal dependencies
from models import YOLOv4, OSNet, FaceIq, CenterFace, ClearFace
from utilities import general_utils as utils
from utilities import io_utils
from utilities import admin_utils


# =============================================================================
#                           - IMAGE PROCESSING -
# -----------------------------------------------------------------------------

# DETECTION:

def detect_people_in_image():
    pass


def detect_faces_in_image(image: Union[str, np.ndarray], image_name: str = None):
    detector = CenterFace()

    if isinstance(image, str):
        image_name = image.split('/')[-1]
        image = cv2.imread(image)

    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)
    
    face_detections = detector.detect_faces(image)
    print(f'{len(face_detections)} faces detected')
    detector.visualize_detections(image, face_detections, output_path=output_path)


# FEATURE EXTRACTION:

def extract_event_img_embeddings(
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
        output_dir=img_dir_path
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


def calculate_embedding_distances(
        embeddings_filepath: str,
        img_data_df: Union[pd.DataFrame, str],
        chunk_size: int = 100
    ) -> pd.DataFrame:
    def _structure_entry(img_1, img_2, distance):
        entry_data = {
            'image1': img_1['image'],
            'image2': img_2['image'],
            'employee_id1': img_1['employee_id'],
            'employee_id2': img_2['employee_id'],
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
                    'employee_id',
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


# SUPER-RESOLUTION:

def enhance_face(image: Union[str, np.ndarray], image_name: str = None):
    clearface = ClearFace(weights_path='../models/weights/clearface/90000_G.pth')

    if isinstance(image, str):
        image_name = image.split('/')[-1]
        image = cv2.imread(image)

    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)

    enhanced_face = clearface.forward(image, is_rgb=True)

    cv2.imwrite(output_path, enhanced_face)


# RECOGNITION:

def recognize_faces_in_image(
        image: Union[str, np.ndarray],
        image_name: str = None,
        focus='global',
        face_iq=None
    ):
    
    if not face_iq:
        face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)
    
    if isinstance(image, str):
        image_name = image.split('/')[-1]
        image = cv2.imread(image)
    
    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)

    resolution = image.shape[:2][::-1]

    if focus == 'face':
        face_df = face_iq.recognize(image, id_cutoff=0.999)
        if not face_df.empty:
            for name, distance in (
                face_df[['name', 'distance']].itertuples(
                    index=False, name=None
                )
            ):
                print(f'Name: {name} | Distance: {distance}')

            face_iq.visualize_identifications(
                image, [face_df], output_path=output_path
            )
        return

    elif focus == 'global':
        regions = None

    elif focus == 'local':
        bboxes = yolov4.detect(image, 0)

        if not bboxes:
            return

        regions = utils.cluster_bboxes_into_regions(
            bboxes, *resolution
        )

    face_dfs = face_iq.identify_faces(image, id_cutoff=0.999, regions=regions)

    print(f'{len(face_dfs)} faces found...')
    for i, face_df in enumerate(face_dfs):
        if face_df.empty:
            continue

        for identity, distance in (
            face_df[['identity', 'distance']].itertuples(index=False, name=None)
        ):

            first_name, _ = io_utils.lookup_name(identity)

            print(f'Detection: {i} | Name: {first_name} | Distance: {distance}')

    face_iq.visualize_identifications(image, face_dfs, output_path=output_path)


# =============================================================================
#                           - VIDEO PROCESSING -
# -----------------------------------------------------------------------------

# DETECTION:

def detect_people_in_video():
    pass


def detect_faces_in_video(
        video_file: str,
        focus: str = 'global',
        input_dir: str = '../files/input',
        output_dir: str = '../files/output'
    ):

    detector = CenterFace(save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

        detector.regions = {}

    video_path = os.path.join(input_dir, video_file)

    output_filename = io_utils.get_unique_filename(
        output_dir, f'{video_file.split(".")[0]}_face_detections.mp4'
    )
    output_path = os.path.join(output_dir, output_filename)

    cap = cv2.VideoCapture(video_path)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)

    print(f'Resolution: {resolution}')
    print(f'FPS: {fps}')
    print(f'Total Frames: {total_frames}')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    
    f_num = -1
    detector.i = f_num

    while f_num < total_frames:
        f_num += 1
        detector.i = f_num
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if (f_num % fps) == 0:
            face_detections = []
    
            if focus == 'global':
                face_detections = detector.detect_faces(frame)
            
            elif focus == 'local':
                bboxes = yolov4.detect(frame, 0)

                if not bboxes:
                    continue
        
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )

                for region in regions:
                    x1, y1, x2, y2 = utils.xywh_xyxy(region)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    if detector.save_data:
                        detector.regions.setdefault(f_num, []).append(region)

                    frame_crop = utils.crop_region(frame, region)

                    region_face_detections = detector.detect_faces(
                        frame_crop, offset=region
                    )

                    face_detections += region_face_detections

            frame = detector.visualize_detections(frame, face_detections)

        cv2.putText(
            frame, f'frame {f_num}',
            (int(resolution[0]/2), int(resolution[1]/2)),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2
        )

        frame = cv2.resize(frame, (1920, 1080))
        out.write(frame)

        f_num += 1
        detector.i = f_num

    out.release()
    cap.release()

    detector.save_runtime_data()


# SUPER-RESOLUTION:

def test_enhanced_face_detections(
        video: str, focus: str = 'global',
        input_dir: str = '../files/input',
        output_dir: str = '../files/output'
    ):

    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

    cap = cv2.VideoCapture(video)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)

    f_num = -1
    face_iq.i = f_num

    while f_num < total_frames:
        f_num += 1
        face_iq.i = f_num
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if (f_num % fps) == 0:
            if focus == 'global':
                face_objects = face_iq.detection_pipeline(
                    frame, align=False, enhance=True, normalize_face=False
                )
    
            elif focus == 'local':
                bboxes = yolov4.detect(frame, 0)

                if not bboxes:
                    continue
    
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )
                for region in regions:
                    img_crop = utils.crop_region(frame, region)
                    local_face_objects = face_iq.detection_pipeline(img_crop, normalize_face=False)
    
    cap.release()
    face_iq.save_runtime_data()


# RECOGNITION:

def recognize_faces_in_video(
        video_file: str,
        focus: str = 'global',
        enhance: bool = False,
        input_dir: str = '../files/input',
        output_dir: str = '../files/output'
    ):

    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

    video_path = os.path.join(input_dir, video_file)

    output_filename = io_utils.get_unique_filename(
        output_dir, f'{video_file.split(".")[0]}_face_identifications.mp4'
    )
    output_path = os.path.join(output_dir, output_filename)

    cap = cv2.VideoCapture(video_path)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)

    print(f'Resolution: {resolution}')
    print(f'FPS: {fps}')
    print(f'Total Frames: {total_frames}')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1, (1920, 1080))

    f_num = -1
    face_iq.i = f_num

    while f_num < total_frames:
        f_num += 1
        face_iq.i = f_num
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if (f_num % fps) == 0:
            if focus == 'global':
                regions = []
    
            elif focus == 'local':
                bboxes = yolov4.detect(frame, 0)

                if not bboxes:
                    continue

                for box in bboxes:
                    x1, y1, x2, y2 = utils.xywh_xyxy(box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )
                for region in regions:
                    x1, y1, x2, y2 = utils.xywh_xyxy(region)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
            all_face_dfs = face_iq.identify_faces(
                frame, id_cutoff=0.999, regions=regions, enhance=enhance
            )
            heatmaps = face_iq.face_detector.heatmaps

            frame = face_iq.visualize_identifications(frame, all_face_dfs)
            frame = face_iq.face_detector.visualize_heatmaps(frame, heatmaps, regions)

            face_iq.face_detector.heatmaps = []

            cv2.putText(
                frame, f'frame {f_num}',
                (int(resolution[0]/2), int(resolution[1]/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2
            )

            frame = cv2.resize(frame, (1920, 1080))
            out.write(frame)
    
    cap.release()
    out.release()

    face_iq.save_runtime_data()


if __name__ == '__main__':
    if '--help' in sys.argv:
        print('========== Required Args: ==========')
        print('1. <fx_nickname>')
        print('     a. --detect-people')
        print('     b. --detect-faces')
        print('     c. --recognize')
        print('     d. --enhance-face')
        print('2. <input_data_path>\n')

        print('========== Optional Args: ==========')
        print('+ <focus>')
        print('     a. --local')
        print('     b. --global')
        print('+ --enhance\n')
        sys.exit(0)

    fx_nickname, input_path = sys.argv[1], sys.argv[2]

    focus = 'local' if ('--local' in sys.argv) else 'global'
    enhance = '--enhance' in sys.argv
    
    file_extension = input_path.split('.')[-1]
    print(f'File extension: {file_extension}')

    if fx_nickname == '--detect-people':
        print('Detecting: people')
        if file_extension in ['png', 'jpg', 'jpeg']:
            print('Input: image')
            detect_people_in_image()
        elif file_extension == 'mp4':
            print('Input: video')
            detect_people_in_video()

    elif fx_nickname == '--detect-faces':
        print('Detecting: faces')
        if file_extension in ['png', 'jpg', 'jpeg']:
            print('Input: image')
            detect_faces_in_image(input_path)
        elif file_extension == 'mp4':
            print('Input: video')
            focus = sys.argv[3]
            detect_faces_in_video(input_path, focus=focus)

    elif fx_nickname == '--recognize':
        if file_extension in ['png', 'jpg', 'jpeg']:
            print('Input: image')
            recognize_faces_in_image(input_path, focus=focus)
        elif file_extension == 'mp4':
            print('Input: video')
            
            recognize_faces_in_video(input_path, focus=focus, enhance=enhance)

    elif fx_nickname == '--enhance-face':
        if file_extension in ['png', 'jpg', 'jpeg']:
            enhance_face(input_path)
        elif file_extension == 'mp4':
            focus = sys.argv[3]
            test_enhanced_face_detections(input_path, focus=focus)
