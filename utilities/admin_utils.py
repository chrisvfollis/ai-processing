# standard dependencies
import os
import subprocess
from typing import Optional
import re
from datetime import datetime, timezone
from io import BytesIO

# 3rd-party dependencies
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import pandas as pd

# internal dependencies
from utilities import conn_utils, io_utils
from utilities.conn_utils import APIClient


# =============================================================================
#                             - GENERAL EC2 -
# -----------------------------------------------------------------------------


def get_edge_computer_info(
        nickname: str = None, shop_id: str = None
) -> dict | None:
    '''
    Fetches edge computer information using either its nickname or the ID of the
    associated shop.

    Args:
        nickname (str): The nickname of the edge computer.
        shop_id (str): The ID of the shop associated with the edge computer.

    Returns:
        computer_info (dict or None): A dictionary of the instance information
            if any was found, otherwise None.
    '''
    if not nickname and not shop_id:
        raise ValueError(
            'At least one of `nickname` or `shop_id` must be provided'
        )

    internal_api = APIClient(var_prefix='INTERNAL_API')
    params = {
        key: value for key, value in [
            ('nickname', nickname), ('shop_id', shop_id)
        ] if value
    }
    computer_info = None
    try:
        r = internal_api.get('get_edge_computer_info/', params=params)
        data = r.json()
        if 'error' in data:
            raise RequestException(data['error'])
        
        computer_info = data.get('results', None)

    except RequestException as e:
        print(f'Error making request: {e}')
    except Exception as e:
        print(f'Unexpected error: {e}')
    
    return computer_info


def get_instance_info(
        nickname: str = None, shop_id: str = None
) -> dict | None:
    '''
    Fetches instance information using either its nickname or the ID of the
    associated shop.

    Args:
        nickname (str): The nickname of the instance.
        shop_id (str): The ID of the shop associated with the instance.

    Returns:
        instance_info (dict or None): A dictionary of the instance information
            if any was found, otherwise None.
    '''
    if not nickname and not shop_id:
        raise ValueError(
            'At least one of `nickname` or `shop_id` must be provided'
        )

    internal_api = APIClient(var_prefix='INTERNAL_API')
    params = {
        key: value for key, value in [
            ('nickname', nickname), ('shop_id', shop_id)
        ] if value
    }
    instance_info = None
    try:
        r = internal_api.get('get_instance_info/', params=params)
        data = r.json()
        if 'error' in data:
            raise RequestException(data['error'])
        
        instance_info = data.get('results', None)

    except RequestException as e:
        print(f'Error making request: {e}')
    except Exception as e:
        print(f'Unexpected error: {e}')
    
    return instance_info


def get_ec2_public_dns(instance_info: dict):
    ec2 = conn_utils.ec2_connect(region=instance_info['region'])

    response = ec2.describe_instances(InstanceIds=[instance_info['instance_id']])
    reservations = response['Reservations']

    if not reservations:
        return None
    
    instance = reservations[0]['Instances'][0]
    public_dns = instance.get('PublicDnsName')

    return public_dns


def auto_scp(
        action: str = 'download',
        remote_base_path: str = '/home/ubuntu/ai-processing/',
        remote_dir: str = 'files/',
        recursive: bool = False,
        local_base_path: str = 'user_home',
        local_dir: str = 'Downloads/',
        nickname: str = None,
        shop_id: str = None,
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
        destination = os.path.join(local_path, remote_dir)
        if os.path.exists(destination):
            print(f'Conflicting path: {destination}')
            conflict = destination
            destination = io_utils.get_unique_subdir(local_path, remote_dir)

            temp_conflict_rename = conflict[:-1] + '_z/'
            os.rename(conflict, temp_conflict_rename)
            print(f'Temporary path: {temp_conflict_rename}')

            _scp_download(remote_path, conflict, instance_info, recursive)
            os.rename(conflict, destination)

            os.rename(temp_conflict_rename, conflict)
        else:
            print(f'Destination clear: {destination}')
            _scp_download(remote_path, local_path, instance_info, recursive)

    elif action == 'upload':
        _scp_upload(local_path, remote_path, instance_info)


# =============================================================================
#                              - GENERAL S3 -
# -----------------------------------------------------------------------------


def s3_list_download(object_keys: list | str, output_dir='resources/downloads',
                  config: Optional[dict | str] = None, s3_client=None):
    
    s3_client = s3_client or conn_utils.s3_connect(region=config['region'])
    bucket = config['bucket']

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


def s3_list_delete(
        object_keys: list | str, region: str = 'us-west-1',
        bucket: str = None, s3_client=None,
):
    s3_client = s3_client or conn_utils.s3_connect(region=region)

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


def s3_time_delete(
    start: datetime | list = None,
    end: datetime | list = None,
    shop_id: str = None,
    region: str = 'us-west-1',
    bucket: str = None,
    s3_client: list = None,
    use_key_timestamp: bool = False,
    dequeue: bool = True,
):
    '''
    Args:
        start (datetime or list): The date/time of the start of the time period.
            It should be UTC if `use_key_timestamp` is False, otherwise match
            the object key. If only `start` is specified, then all subsequent
            files are deleted.
        end (datetime or list): The date/time of the end of the time period. It
            should be UTC if `use_key_timestamp` is False, otherwise match the
            object key. If only end is specified, then all prior files are
            deleted.
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
            'Both arguments for `start` and `end` are None. At least\n' +
            'one of these parameters must reference a point in time.'
        )

    s3_client = s3_client or conn_utils.s3_connect(region=region)

    if isinstance(start, list):
        start = datetime(*start)
    if isinstance(end, list):
        end = datetime(*end)

    if use_key_timestamp:
        if start and start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if end and end.tzinfo is not None:
            end = end.replace(tzinfo=None)
    else:
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

                if use_key_timestamp:
                    timestamp = _parse_timestamp_from_key(obj_key)
                    if timestamp and (
                        ((start is None) or (timestamp > start)) and
                        ((end is None) or (timestamp < end))
                    ):
                        object_keys.append(obj_key)
                        timestamps_to_clear.add(timestamp)
                else:
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

    results = s3_list_delete(
        object_keys,
        region,
        bucket,
        s3_client,
    )
    if dequeue == True:
        for timestamp in timestamps_to_clear:
            io_utils.dequeue_segment(shop_id, timestamp)

    return results


# =============================================================================
#                         - API/REMOTE DATABASE -
# -----------------------------------------------------------------------------


def clear_queue_range(
        start: datetime | list = None,
        end: datetime | list = None,
        shop_id: Optional[str] = None,
) -> None:
    if (start is None) and (end is None):
        raise ValueError(
            'At least one of `start` or `end` must be provided.'
        )

    internal_api = APIClient(var_prefix='INTERNAL_API')

    payload = {'directive': 'delete_range'}
    if shop_id:
        payload['shop_id'] = shop_id
    if start:
        if isinstance(start, list):
            start = datetime(*start)
        payload['start'] = start.isoformat()
    if end:
        if isinstance(end, list):
            end = datetime(*end)
        payload['end'] = end.isoformat()

    response = internal_api.post('update_queue/', json=payload)

    if response.status_code == 200:
        print('Successfully cleared queue range')
    else:
        print(f'Failed posting to internal API: {response.text}')
        print(response.status_code)


# =============================================================================
#                            - OUTPUT DATA -
# -----------------------------------------------------------------------------


def get_approved_records(shop_id=None) -> list[tuple]:
    ''''
    Retrieve approved employee event records to get accurate labeled image data
    for training and/or fine tuning.
    '''
    webapp_api = APIClient(var_prefix='WEBAPP_API')
    params = {}
    if shop_id:
        params['shop_id'] = shop_id

    try:
        response = webapp_api.get(
            'approved_employee_event_records/', params=params
        )
        response.raise_for_status()

        data = response.json()
        formatted_records = []
        for record in data.get('results', []):
            formatted_records.append((
                record['employee_id'],
                record['shop_id'],
                record['image'],
                record['start_time'],
                record['first_name'],
                record['last_name'],
            ))
        return formatted_records

    except requests.RequestException as e:
        print(f'Request failed: {e}')
        return []
    except Exception as e:
        print(f'Unexpected error: {e}')
        return []


def save_approved_img_data(
        approved_records: list[tuple] = None,
        bucket_name: str = 'timemanager-event-imgs',
        shop_id: str = None,
) -> pd.DataFrame:
    approved_records = approved_records or get_approved_records(shop_id)

    project_root = io_utils.get_project_root()
    output_dir = os.path.join(project_root, 'files/output/')
    img_dir = os.path.join(output_dir, 'event_imgs/')

    s3_client = conn_utils.s3_connect(region='us-west-1')

    saved_img_data = []

    cols = [
        'person_id',
        'shop_id',
        'image',
        'start_time',
        'first_name',
        'last_name',
    ]

    print('Saving approved event image data...')
    for row in approved_records:
        row_data = dict(zip(cols, row))

        img_save_path = os.path.join(img_dir, row_data['image'])
        row_data['path'] = img_save_path
        try:
            if not os.path.exists(img_save_path):
                s3_obj = s3_client.get_object(
                    Bucket=bucket_name,
                    Key=row_data['image']
                )
                img = Image.open(BytesIO(s3_obj['Body'].read()))

                img.save(img_save_path)
            saved_img_data.append(row_data)

        except Exception as e:
            print(f"Error retrieving or saving {row_data['image']}: {e}")
    
    saved_img_data_df = pd.DataFrame(saved_img_data)

    for col in saved_img_data_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(saved_img_data_df[col]):
            continue
        standardized = saved_img_data_df[col].dt.tz_convert('UTC')
        saved_img_data_df[col] = standardized.dt.tz_localize(None)
    
    spreadsheet_path = os.path.join(output_dir, 'event_imgs_data.xlsx')
    with pd.ExcelWriter(spreadsheet_path, engine='openpyxl', mode='w') as writer:
        saved_img_data_df.to_excel(writer, sheet_name='Metadata', index=False)

    return saved_img_data_df
