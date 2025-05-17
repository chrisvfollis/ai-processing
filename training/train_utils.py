# standard dependencies
from typing import Optional, Union

# 3rd-party dependencies
import pandas as pd

# internal dependencies
pass


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


def save_dataset_manifest(
        training: pd.DataFrame,
        validation: pd.DataFrame,
        testing: Optional[pd.DataFrame] = None,
        output_path: str = None
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
