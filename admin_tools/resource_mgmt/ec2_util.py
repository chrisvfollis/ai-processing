import subprocess
import os


def scp_download(remote_dir, local_dir, remote_host, remote_user='ubuntu',
                 remote_path='/home/ubuntu/timemanager/', pem='timemanager.pem'):
    local_path = os.path.dirname(os.getcwd())

    codebase_dirs = ['input_files/', 'output_files/',
                     'intermediate_output', 'admin/testing/output/']
    
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
        '-i', pem,
        '-r',
        f"{remote_user}@{remote_host}:{remote_path}",
        local_path
    ]

    scp_command = [arg for arg in scp_command if arg]

    try:
        subprocess.run(scp_command, check=True)
        print(f"Successfully copied {remote_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during SCP: {e}")
