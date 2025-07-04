# standard dependencies
pass

# 3rd-party dependencies
import av
import cv2
import pandas as pd

# internal dependencies
pass


def visualize_global_id_output(input_path, output_path, face_df, trk_df):
    container = av.open(input_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    width = stream.codec_context.width
    height = stream.codec_context.height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    for frame in container.decode(stream):
        img = frame.to_ndarray(format='bgr24')

        tracks = trk_df[trk_df['f'] == frame_num]
        for _, row in tracks.iterrows():
            x1, y1, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 255, 0), 2)

        faces = face_df[face_df['f'] == frame_num]
        for _, row in faces.iterrows():
            x, y, w, h = int(row.x), int(row.y), int(row.w), int(row.h)
            ident = row.name if pd.notna(row.name) else '?'
            label = str(ident)[:12]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(img, f'Frame {frame_num}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        writer.write(img)
        frame_num += 1

    writer.release()
    container.close()
