from botocore.exceptions import ClientError
import os
from utilities import initial_s3_setup
from typing import Union
from datetime import datetime, timezone


def list_download(object_keys: Union[list, str], output_dir='resources/downloads',
                  config: Union[dict, str] = None, existing_setup: list = None):
    
    if existing_setup is None:
        items = initial_s3_setup(config, object_keys, output_dir)
        s3_client, bucket, object_keys = items
    else:
        s3_client, bucket = existing_setup

    results = {'downloaded': [], 'failed': {}}
    for obj_key in object_keys:
        try:
            local_path = os.path.join(output_dir, obj_key)
            s3_client.download_file(bucket, obj_key, local_path)
            results['downloaded'].append(obj_key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']

            results['failed'].setdefault(error_code, []).append(obj_key)
            print(f'Failed to download {obj_key}: {error_code}: {error_message}')
        except Exception as e:
            results['failed'].setdefault("UnknownError", []).append(obj_key)
            print(f"Failed to download {obj_key}: {e}")

    return results


def list_delete(object_keys: Union[list, str], config: Union[dict, str] = None,
                existing_setup: list = None):

    if existing_setup is None:
        s3_client, bucket, object_keys = initial_s3_setup(config, object_keys)
    else:
        s3_client, bucket = existing_setup

    results = {'deleted': [], 'failed': {}}
    for i in range(0, len(object_keys), 1000):
        delete_params = {
            'Objects': [{'Key': key} for key in object_keys[i:i + 1000]],
            'Quiet': True
        }

        response = s3_client.delete_objects(Bucket=bucket, Delete=delete_params)

        for deleted_obj in response.get('Deleted', []):
            results['deleted'].append(deleted_obj['Key'])
 
        for error in response.get('Errors', []):
            error_code = error['Code']
            error_key = error['Key']
            error_message = error['Message']

            results['failed'].setdefault(error_code, []).append(error_key)
            print(f"Failed to delete {error_key}: {error_code}: {error_message}")

    return results


def time_delete(
        start: Union[datetime, list] = None, end: Union[datetime, list] = None,
        config: Union[dict, str] = None, existing_setup: list = None
    ):

    if (start is None) and (end is None):
        raise ValueError(
            "Both arguments for 'start' and 'end' are None. At least\n" +
            "one of these parameters must reference a point in time."
        )

    if existing_setup is None:
        s3_client, bucket = initial_s3_setup(config)
    else:
        s3_client, bucket = existing_setup

    if isinstance(start, list):
        start = datetime(*start)
    if isinstance(end, list):
        end = datetime(*end)

    if start and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    object_keys = []
    results = {'deleted': [], 'failed': {}}
    
    try:
        print('Collecting object keys...')
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket):
            if 'Contents' not in page:
                continue

            for object in page['Contents']:
                obj_key = object['Key']
                last_modified = object['LastModified']

                if (
                    ((start is None) or (last_modified > start)) and
                    ((end is None) or (last_modified < end))
                ):
                    object_keys.append(obj_key)
        print(f'Matching object keys: {object_keys}')

    except ClientError as e:
        print(f"Error listing objects: {e}")

    results = list_delete(object_keys, existing_setup=[s3_client, bucket])

    return results
