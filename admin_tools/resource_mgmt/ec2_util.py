import subprocess
import os
from admin_tools import admin_utils
import boto3


def scp_download(remote_dir, local_dir, config, cfg_dir=None,
                 remote_path='/home/ubuntu/ai-processing/'):
    if not cfg_dir:
        pem_path, remote_user, public_dns = admin_utils.ec2_public_dns(config)
    else:
        pem_path, remote_user, public_dns = admin_utils.ec2_public_dns(
            config, dir_path=cfg_dir
        )
    pem_path = os.path.join('../', pem_path)

    local_path = os.path.dirname(os.getcwd())
    print(local_path)

    codebase_dirs = ['files/input/', 'files/output/', 'admin_tools/testing/output/']
    
    if remote_dir in codebase_dirs:
        remote_path = os.path.join(remote_path, remote_dir)
    else:
        remote_path = remote_dir
    
    if local_dir in codebase_dirs:
        local_path = os.path.join(local_path, local_dir)
    else:
        local_path = local_dir
    
    if remote_dir == local_dir:
        remote_path = os.path.join(remote_path, '*')
    
    scp_command = [
        'scp',
        '-i', pem_path,
        '-r',
        f"{remote_user}@{public_dns}:{remote_path}",
        local_path
    ]

    scp_command = [arg for arg in scp_command if arg]

    try:
        subprocess.run(scp_command, check=True)
        print(f"Successfully copied {remote_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during SCP: {e}")
