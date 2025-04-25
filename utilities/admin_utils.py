# standard dependencies
import os
import subprocess
from typing import Union
import re
from datetime import datetime, timezone

# 3rd-party dependencies
from dotenv import load_dotenv
import requests
import boto3
from botocore.exceptions import ClientError

# internal dependencies
from utilities import io_utils


def get_instance_info(nickname: str = None, shop_id: str = None):
    '''
    Fetches instance information using either its nickname or the ID of the
    associated shop.

    Args:
        nickname (str): The nickname of the instance.
        shop_id (str): The ID of the shop associated with the instance.

    Returns:
        dict or None: The instance info dictionary if found, None if not found,
                      or False if an error occurs.
    '''
    load_dotenv()

    base_url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/'
    endpoint = 'api/service/get_instance_info/'

    endpoint_url = base_url + endpoint

    headers = {
        'X-Custom-API-Key': os.environ.get('INTERNAL_API_KEY'),
        'Content-Type': 'application/json'
    }

    params = {}
    if nickname:
        params['nickname'] = nickname
    elif shop_id:
        params['shop_id'] = shop_id
    else:
        raise TypeError("no arguments supplied to get_instance_info()") 

    try:
        response = requests.get(endpoint_url, headers=headers, params=params)
        data = response.json()

        if 'error' in data:
            print(f"Error: {data['error']}")
            return False

        return data.get('results', None)

    except requests.exceptions.RequestException as e:
        print(f'Error making request: {e}')
        return False
    except Exception as e:
        print(f'Unexpected error: {e}')
        return False


def validate_filepath(file_name, dir_path):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(
            f'The specified directory path does not exist: {dir_path}.'
        )
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'{file_name} does not exist in the specified directory:' +
            dir_path
        )
    return file_path


def read_aws_config(file_name, dir_path='resource_mgmt/configs'):
    file_path = validate_filepath(file_name, dir_path)
    
    config_dict = {}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.split('=')
            config_dict[line[0].strip().lower()] = line[1].strip()

    return config_dict


def read_listfile(file_name, dir_path='resource_mgmt/lists'):
    file_path = validate_filepath(file_name, dir_path)
    items = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line:
                items.append(line)
                print(line)

    return items


def initial_s3_setup(config, obj_keys=None, output_dir=None):
    if isinstance(config, str):
        config = read_aws_config(config)

    region = config['region']
    bucket = config['bucket']

    load_dotenv()
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
        region_name=region
    )

    if obj_keys is None:
        return s3_client, bucket
    else:
        if isinstance(obj_keys, str):
            obj_keys = read_listfile(obj_keys)
    
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        return s3_client, bucket, obj_keys


def get_ec2_public_dns(instance_info: dict):
    
    instance_id = instance_info['instance_id']
    region = instance_info['region']

    ec2 = boto3.client(
        'ec2',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
        region_name=region
    )
    
    response = ec2.describe_instances(InstanceIds=[instance_id])
    reservations = response['Reservations']
    
    if reservations:
        instance = reservations[0]['Instances'][0]
        public_dns = instance.get('PublicDnsName')
        return public_dns
    else:
        return None


def auto_scp(
        action: str = 'download',
        remote_base_path: str = '/home/ubuntu/ai-processing/',
        remote_dir: str = 'files/',
        recursive: bool = False,
        local_base_path: str = 'user_home',
        local_dir: str = 'Downloads/',
        nickname: str = None,
        shop_id: str = None
    ):
    def _scp_download(remote_path, local_path, instance_info, recursive):
        key_path = os.path.join('files/keys/', instance_info['key_filename'])
        remote_user = instance_info['remote_user']
        public_dns = get_ec2_public_dns(instance_info)
        
        scp_command = ['scp', '-i', key_path]

        if recursive:
            scp_command.append('-r')

        scp_command.extend([
            f"{remote_user}@{public_dns}:{remote_path}",
            local_path
        ])

        try:
            subprocess.run(scp_command, check=True)
            print(f"Successfully copied {remote_base_path} to {local_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during SCP: {e}")

    def _scp_upload(local_path, remote_path, instance_info):

        key_path = os.path.join('files/keys/', instance_info['key_filename'])
        remote_user = instance_info['remote_user']
        public_dns = get_ec2_public_dns(instance_info)
        
        scp_command = [
            'scp',
            '-i', key_path,
            '-r',
            local_path,
            f"{remote_user}@{public_dns}:{remote_path}"
        ]

        scp_command = [arg for arg in scp_command if arg]

        try:
            subprocess.run(scp_command, check=True)
            print(f"Successfully copied {remote_base_path} to {local_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during SCP: {e}")
        pass

    if local_base_path == 'user_home':
        local_base_path = os.path.expanduser('~')

    remote_path = os.path.join(remote_base_path, remote_dir)
    local_path = os.path.join(local_base_path, local_dir)

    instance_info = get_instance_info(nickname=nickname, shop_id=shop_id)

    if action == 'download':
        _scp_download(remote_path, local_path, instance_info, recursive)
    elif action == 'upload':
        _scp_upload(local_path, remote_path, instance_info)


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
        shop_id: str = None, config: Union[dict, str] = None,
        existing_setup: list = None
    ):
    '''
    Args:
        start: date/time in UTC of the start of the time period. If only start is
               specified without end, then all files after the start are deleted.
        end: date/time in UTC of the end of the time period. If only end is
             specified without start, then all files prior to the end are deleted.
    '''
    def _parse_timestamp_from_key(obj_key):
        try:
            matches = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', obj_key)
            if matches:
                time_str = matches.group(1)
                return datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            pass
        return None
    
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
    timestamps_to_clear = set()
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
                    timestamp = _parse_timestamp_from_key(obj_key)
                    if timestamp:
                        timestamps_to_clear.add(timestamp)

        print(f'Matching object keys: {object_keys}')

    except ClientError as e:
        print(f"Error listing objects: {e}")

    results = list_delete(object_keys, existing_setup=[s3_client, bucket])
    for timestamp in timestamps_to_clear:
        io_utils.clear_queue_block(shop_id, timestamp)

    return results
