import os
from dotenv import load_dotenv
import boto3


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


def ec2_public_dns(config, dir_path=None):
    if isinstance(config, str):
        if not dir_path:
            config = read_aws_config(config)
        else:
            config = read_aws_config(
                config, dir_path=dir_path
            )
    
    region = config['region']
    instance_id = config['instance_id']
    pem_path = config['pem_path']
    remote_user = config['remote_user']

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
        return pem_path, remote_user, public_dns
    else:
        return None
