from datetime import datetime
import os


def load_directories(path):
    file_names = os.listdir(path)
    file_path_list = []
    for file_name in file_names:
        file_path_list.append(os.path.join(path, file_name))
    file_path_list.sort()
    return file_path_list


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_root_dir():
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    return 'output_lv2/' + now