# standard dependencies
import os
import argparse
from typing import Union
import uuid
import sys
from datetime import datetime

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils.dataframe import dataframe_to_rows

# internal dependencies
from utilities import test_utils
from models.face_iq import FaceIq
from utilities import general_utils as utils
from utilities import io_utils


def identify_event_imgs(img_dir='../files/output/event_imgs/'):
    
    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    all_images = [img for img in os.listdir(img_dir)
                  if not img.endswith('.gitkeep')]
    all_face_dfs = []
    
    for image_name in all_images:
        image = cv2.imread(os.path.join(img_dir, image_name))
        output_path = os.path.join('../files/output', image_name)
        
        best_detection = pd.DataFrame()
        face_dfs = face_iq.identify_faces(image, id_cutoff=0.8)
        for face_df in face_dfs:
            best_match = face_df.loc[[face_df['distance'].idxmin()]]

            distance = best_match['distance'].iloc[0]
            if (best_detection.empty) or (distance > best_detection['distance'].iloc[0]):
                best_detection = best_match

        if not best_detection.empty:
            best_detection['img_path'] = output_path
            all_face_dfs.append(best_detection)
        
            face_iq.visualize_identifications(image, [best_detection], output_path=output_path)
    
    full_face_df = pd.concat(all_face_dfs)

    return full_face_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-imgs', type=int)
    parser.add_argument('--start-from', type=str, help='Comma-separated datetime')

    args = parser.parse_args()

    max_imgs = args.max_imgs or 1000
    start_from = args.start_from

    if start_from:
        try:
            parts = [int(x) for x in args.start_from.split(',')]
            start_from = datetime(*parts)
        except Exception as e:
            print(f'Invalid --start-from value: {args.start_from} ({e})')
            sys.exit(1)

    test_utils.download_event_imgs(max_imgs=max_imgs, start_from=start_from)

    full_face_df = identify_event_imgs()
    test_utils.export_face_df_with_images(full_face_df)
