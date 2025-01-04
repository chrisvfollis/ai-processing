import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import utilities
from typing import Union


def s3_bulk_download(object_keys: Union[list, str], config: Union[dict, str],
                     output_dir='downloads'):
    def _initial_setup(config, obj_keys, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(config, str):
            config = utilities.read_aws_config(config)

        region = config['region']
        bucket = config['bucket']

        load_dotenv()
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
            region_name=region
        )

        if isinstance(obj_keys, str):
            obj_keys = utilities.read_list_file(obj_keys)

        return (s3_client, bucket, obj_keys)
    
    results = {'downloaded': [], 'failed': {}}
    items = _initial_setup(config, object_keys, output_dir)
    s3_client, bucket, object_keys = items

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


def s3_bulk_delete(object_keys: Union[list, str], config: Union[dict, str]):
    def _initial_setup(config, obj_keys):
        if isinstance(config, str):
            config = utilities.read_aws_config(config)

        region = config['region']
        bucket = config['bucket']

        load_dotenv()
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
            region_name=region
        )

        if isinstance(obj_keys, str):
            obj_keys = utilities.read_list_file(obj_keys)

        return (s3_client, bucket, obj_keys)
    
    items = _initial_setup(config, object_keys)
    s3_client, bucket, object_keys = items

    delete_params = {
        'Objects': [{'Key': key} for key in object_keys],
        'Quiet': True
    }

    response = s3_client.delete_objects(Bucket=bucket, Delete=delete_params)
    print("Errors:", response.get('Errors', []))
