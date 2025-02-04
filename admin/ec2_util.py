import subprocess
import os

def scp_download(remote_path, local_destination, pem_path, remote_host, remote_user='ubuntu'):

    mappings = {
        ''
    }


    recursive_flag = '-r' if remote_path.endswith('/') else ''

    scp_command = [
        'scp',
        '-i', pem_path,
        recursive_flag,
        f"{remote_user}@{remote_host}:{remote_path}",
        local_destination
    ]

    scp_command = [arg for arg in scp_command if arg]

    try:
        subprocess.run(scp_command, check=True)
        print(f"Successfully copied {remote_path} to {local_destination}")
    except subprocess.CalledProcessError as e:
        print(f"Error during SCP: {e}")

