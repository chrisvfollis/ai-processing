# standard dependencies
from typing import Optional
import os

# 3rd-party dependencies
import av
import cv2
import pandas as pd

# internal dependencies
import utilities.general_utils as utils
from utilities import io_utils


def global_id_output(video_file, person_df, face_df, trk_df, region_df, f_cutoff: Optional[int] = None):
    project_root = io_utils.get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/')

    time_prefix, cam_id = utils.decode_vid_filename(video_file)

    input_path = os.path.join(input_dir, video_file)
    output_path = os.path.join(
        output_dir, 'videos/', f'{time_prefix}_{cam_id}_annotated.mp4'
    )

    container = av.open(input_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)

    target_width = 960
    target_height = 540

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    f_num = 0
    for frame in container.decode(stream):
        if f_cutoff and (f_num >= f_cutoff):
            break
        img = frame.to_ndarray(format='bgr24')

        people = person_df[person_df['f'] == f_num]
        for _, row in people.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        tracks = trk_df[trk_df['f'] == f_num]
        for _, row in tracks.iterrows():
            x1, y1, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 255, 0), 2)

        faces = face_df[face_df['f'] == f_num]
        for _, row in faces.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            ident = row.name if pd.notna(row.name) else '?'
            label = str(ident)[:12]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

        regions = region_df[region_df['f'] == f_num]
        for _, row in regions.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.putText(img, f'Frame {f_num}', (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

        img = cv2.resize(img, (target_width, target_height))
        writer.write(img)
        f_num += 1

    writer.release()
    container.close()
