import os
from datetime import datetime

import git


def load_directories(path):
    file_name = os.listdir(path)
    file_path = []
    for i in file_name:
        extension = i.split('.')[-1]
        if extension in 'png':
            file_path.append(path + '/' + i)
    file_path.sort()
    return file_path


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_root_dir():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    commit_hash = repo.git.rev_parse(sha)
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    return 'output_lv1/' + commit_hash + '_' + now
