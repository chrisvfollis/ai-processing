import os


def validate_file_path(file_name, dir_path):
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


def read_aws_config(file_name, dir_path='configs'):
    file_path = validate_file_path(file_name, dir_path)
    
    config_dict = {}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.split('=')
            config_dict[line[0].strip().lower()] = line[1].strip()

    return config_dict


def read_list_file(file_name, dir_path='lists'):
    file_path = validate_file_path(file_name, dir_path)
    items = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            items.append(line)

    return items
