# standard dependencies
from typing import Optional, Union

# 3rd-party dependencies
import pandas as pd

# internal dependencies
from utilities import conn_utils
from utilities.conn_utils import APIClient


def clean_dataset(
        dataset: pd.DataFrame,
        class_col: str = 'person_id',
        split: Union[str, int] = 'test',
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
    run_id: str,
    model_name: str,
    starting_weights: str,
    output_weights: str,
    dataset_name: str,
    train_manifest: str,
    num_classes: int,
    num_train_samples: int,
    num_val_samples: int,
    epoch_count: int,
    final_train_loss: float,
    final_val_loss: float,
    additional_metadata: dict = None,
    ) -> bool:
    webapp_api = APIClient(var_prefix='WEBAPP_API')

    payload = {
        'run_id': run_id,
        'model_name': model_name,
        'starting_weights': starting_weights,
        'output_weights': output_weights,
        'dataset_name': dataset_name,
        'train_manifest': train_manifest,
        'num_classes': num_classes,
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'epoch_count': epoch_count,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'additional_metadata': additional_metadata or {},
    }

    response = webapp_api.post('log_training/', json=payload)
    if response.status_code == 201:
        print('Logged training run:', response.json())
        return True
    else:
        print('Failed to log training run:', response.status_code, response.text)
        return False
