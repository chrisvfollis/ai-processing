# standard dependencies
from typing import Optional
import os

# 3rd-party dependencies
import av
import cv2

# internal dependencies
from utilities import utils, io_utils
from modules.results.render import annotate


def global_id_output(video_file, person_df, face_df, trk_df, region_df, f_cutoff: Optional[int] = None):
    project_root = io_utils.get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/')

    time_segment, cam_id = utils.decode_vid_filename(video_file)

    input_path = os.path.join(input_dir, video_file)
    output_path = os.path.join(
        output_dir, 'videos/', f'{time_segment}_{cam_id}_annotated.mp4'
    )

    container = av.open(input_path)
    stream = container.streams.video[0]

    fps = float(stream.average_rate)
    img_w = stream.codec_context.width
    img_h = stream.codec_context.height

    target_width = 1280
    target_height = 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    fontscale_small = 1
    fontscale_large = 3
    thickness = 2

    f_num = 0
    for frame in container.decode(stream):
        if f_cutoff and (f_num >= f_cutoff):
            break
        img = frame.to_ndarray(format='bgr24')

        people = person_df[person_df['f'] == f_num]
        for _, row in people.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            det_conf = str(row.c)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            annotate.text_with_shadow(
                img,
                text=det_conf,
                xy_org=(int(x + w/2), int(y + h/2)),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                fontscale=fontscale_small,
                color=(255, 255, 255),
                thickness=thickness,
                shadow_color=(0, 0, 0),
                offset=(-2, -2),
            )

        tracks = trk_df[trk_df['f'] == f_num]
        for _, row in tracks.iterrows():
            x1, y1, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 255, 0), thickness)

        faces = face_df[face_df['f'] == f_num]
        for _, row in faces.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)

        regions = region_df[region_df['f'] == f_num]
        for _, row in regions.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), thickness)

        cv2.putText(
            img,
            str(f_num),
            (int(img_w/2), int(img_h/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale_large,
            (75, 25, 255),
            thickness
        )

        img = cv2.resize(img, (target_width, target_height))
        writer.write(img)
        f_num += 1

    writer.release()
    container.close()
