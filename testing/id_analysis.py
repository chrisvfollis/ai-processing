# standard dependencies
import os
import argparse
from typing import Union
import sys
from datetime import datetime

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# internal dependencies
from utilities import test_utils
from models.face_iq import FaceIq


def generate_id_data(id_cutoff=0.6, img_dir='../files/output/event_imgs/'):
    
    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=False)

    all_images = [img for img in os.listdir(img_dir)
                  if not img.endswith('.gitkeep')]
    all_face_dfs = []
    no_face_events = 0
    
    for image_name in all_images:
        image = cv2.imread(os.path.join(img_dir, image_name))
        output_path = os.path.join('../files/output', image_name)
        
        best_detection = pd.DataFrame()
        face_dfs = face_iq.identify_faces(image, id_cutoff=id_cutoff)
        for face_df in face_dfs:
            if face_df.empty:
                continue

            best_match = face_df.loc[[face_df['distance'].idxmin()]]

            distance = best_match['distance'].iloc[0]
            if (best_detection.empty) or (distance > best_detection['distance'].iloc[0]):
                best_detection = best_match

        if not best_detection.empty:
            best_detection['img_path'] = output_path
            all_face_dfs.append(best_detection)
        
            face_iq.visualize_identifications(image, [best_detection], output_path=output_path)
        else:
            no_face_events += 1

    full_face_df = pd.concat(all_face_dfs)

    print(f'{no_face_events} event images with no detected faces')
    return full_face_df, no_face_events


def analyze_id_data(df, output_dir='../files/output'):
    os.makedirs(output_dir, exist_ok=True)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(df['correct_id'], -df['distance'])  # flip distance, since lower is better
    roc_auc = auc(fpr, tpr)

    # find optimal cutoff (maximizes TPR - FPR)
    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[j_idx]
    optimal_fpr = fpr[j_idx]
    optimal_tpr = tpr[j_idx]

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

    ax.plot(optimal_fpr, optimal_tpr, 'ro', label=f'Optimal threshold = {-1 * optimal_threshold:.3f}')
    ax.annotate(
        f'Thresh = {-1 * optimal_threshold:.3f}',
        (optimal_fpr, optimal_tpr),
        textcoords="offset points",
        xytext=(10, -15),
        ha='left',
        fontsize=9,
        color='red',
        arrowprops=dict(arrowstyle='->', lw=1, color='red')
    )

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str)

    parser.add_argument('--max-imgs', type=int)
    parser.add_argument('--start-from', type=str, help='Comma-separated datetime')
    parser.add_argument('--min-kb', type=int)
    parser.add_argument('--id-cutoff', type=float)

    args = parser.parse_args()

    mode = args.mode or 'analyze'

    if mode == 'generate':
        max_imgs = args.max_imgs or 1000
        start_from = args.start_from
        id_cutoff = args.id_cutoff or 0.6

        min_kb = args.min_kb or 0
        min_bytes = (min_kb * 1000)

        if start_from:
            try:
                parts = [int(x) for x in args.start_from.split(',')]
                start_from = datetime(*parts).replace(tzinfo=None)
            except Exception as e:
                print(f'Invalid --start-from value: {args.start_from} ({e})')
                sys.exit(1)

        test_utils.download_event_imgs(max_imgs=max_imgs, start_from=start_from,
                                       min_bytes=min_bytes)

        full_face_df, no_face_events = generate_id_data(id_cutoff=id_cutoff)
        test_utils.export_face_df_with_images(full_face_df)
    
    elif mode == 'analyze':
        output_dir = '../files/output/'
        filename = 'event_img_face_data.xlsx'

        file_path = os.path.join(output_dir, filename)

        df = pd.read_excel(file_path)
        analyze_id_data(df, output_dir=output_dir)
