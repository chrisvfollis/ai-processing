# standard dependencies
from typing import Optional

# 3rd-party dependencies
import pandas as pd
import numpy as np

# internal dependencies
from utilities import conn_utils, io_utils
from utilities.conn_utils import APIClient


def clean_dataset(
        dataset: pd.DataFrame,
        class_col: str = 'person_id',
        split: str | int = 'test',
) -> pd.DataFrame:
    '''
    Removes classes from the dataset with fewer samples than the minimum
    required for the given split.  
    '''
    if isinstance(split, str):
        if split == 'validate':
            min_samples = 2
        elif split == 'test':
            min_samples = 3
        else:
            raise ValueError(f'Unrecognized split type: {split}')
    else:
        min_samples = int(split)
    
    sample_counts = dataset[class_col].value_counts()
    valid_classes = sample_counts[sample_counts >= min_samples].index

    dataset = dataset[dataset[class_col].isin(valid_classes)]

    return dataset


def create_dataset_manifest(
        training: pd.DataFrame,
        validation: pd.DataFrame,
        testing: Optional[pd.DataFrame] = None,
        output_path: str = None,
) -> pd.DataFrame:
    '''
    Saves pertinent information about each sample in the dataset, such as
    whether it was in the training, validation, or (optional) testing set. 
    '''
    training = training.copy()
    training['split'] = 'train'

    validation = validation.copy()
    validation['split'] = 'val'

    all_datasets = [training, validation]

    if testing is not None:
        testing = testing.copy()
        testing['split'] = 'test'
        all_datasets.append(testing)

    manifest_df = pd.concat(all_datasets, ignore_index=True)

    if output_path:
        manifest_df.to_csv(output_path, index=False)

    return manifest_df


def log_training_run(
    checkpoint_id: str,
    model_name: str,
    source_checkpoint: str,
    output_checkpoint: str,
    dataset_name: str,
    manifest_file: str,
    num_classes: int,
    num_train_samples: int,
    num_val_samples: int,
    triplet_loss_margin: float,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    final_train_loss: float,
    final_val_score: float,
) -> bool:
    webapp_api = APIClient(var_prefix='INTERNAL_API')

    model_info = {
        'checkpoint_id': checkpoint_id,
        'model_name': model_name,
        'source_checkpoint': source_checkpoint,
        'output_checkpoint': output_checkpoint,
    }
    dataset_info = {
        'dataset_name': dataset_name,
        'manifest_file': manifest_file,
        'num_classes': num_classes,
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
    }
    hyperparameters = {
        'triplet_loss_margin': triplet_loss_margin,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
    }
    results = {
        'final_train_loss': final_train_loss,
        'final_val_score': final_val_score,
    }

    payload = model_info | dataset_info | hyperparameters | results

    response = webapp_api.post('log_training/', json=payload)
    if response.status_code == 201:
        print('Logged training run:', response.json())
        return True
    else:
        print('Failed to log training run:', response.status_code, response.text)
        return False


def log_epoch_data(checkpoint_id: str, epoch_data: list[dict]):
    webapp_api = APIClient(var_prefix='INTERNAL_API')

    for data in epoch_data:
        payload = data | {'checkpoint_id': checkpoint_id}

        for k, v in payload.items():
            if isinstance(v, (np.integer, np.floating)):
                payload[k] = v.item()

        response = webapp_api.post('log_epoch/', json=payload)
        if response.status_code == 201:
            print(f'Logged epoch {data["epoch"]}')
        else:
            print(
                f'Failed to log epoch {data["epoch"]}:',
                response.status_code,
                response.text,
            )


def upload_training_files(
        checkpoint_path: str = None,
        manifest_path: str = None,
        epoch_data_path: str = None,
        region: str = 'us-west-1',
        bucket_name: str = 'ivakt-training-files',
):
    s3_client = conn_utils.s3_connect(region=region)

    for path in [checkpoint_path, manifest_path, epoch_data_path]:
        filename = path.split('/')[-1]
        io_utils.upload_file(
            s3_client,
            bucket_name,
            file_path=path,
            object_key=filename,
        )


def download_best_checkpoint():
    pass
