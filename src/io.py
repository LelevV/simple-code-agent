import os


def read_file(path):
    with open(path, 'r') as f:
        return f.read()
    

def list_files(project_dir):
    file_list = []
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            file_list.append(os.path.relpath(os.path.join(root, file), project_dir))
    return file_list


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)