import os
from pathlib import Path

import pandas as pd

from utilities import log_utils, conn_utils, io_utils


logger = log_utils.get_logger()


if __name__ == '__main__':
    log_utils.configure_logging()
    credentials = conn_utils.get_aws_credentials()

    project_root = io_utils.get_project_root()
    output_dir = os.path.join(project_root, 'files/output/')

    time_segment = '2025-06-10_09-00-00'

    trk_files = sorted(Path(output_dir).glob(f'{time_segment}_*_trk_dets.parquet'))
    trk_dets  = pd.concat([pd.read_parquet(f) for f in trk_files],  ignore_index=True)

    presence_df = pd.read_csv(os.path.join(
        output_dir, f'{time_segment}_presence_summary.csv'
    ))
    filtered_faces = pd.read_csv(os.path.join(
        output_dir, f'{time_segment}_filtered_faces.csv'
    ))

    io_utils.save_global_id_event_imgs(
        time_segment, presence_df, filtered_faces, trk_dets, credentials
    )
