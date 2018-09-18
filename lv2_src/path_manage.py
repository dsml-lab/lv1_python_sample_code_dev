import os
from datetime import datetime

import git


def load_directories(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    file_path_list = []
    for dir_name in directories:
        file_path_list.append(os.path.join(path, dir_name))
    file_path_list.sort()
    return file_path_list


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_root_dir():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    commit_hash = repo.git.rev_parse(sha)
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    return 'output_lv2/' + commit_hash + '_' + now
