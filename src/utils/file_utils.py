import os

def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path

def get_project_root():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.dirname(path) # get to parent, outside of project code path"
    return path

def read_resource_file(path):

    assert path.startswith("./res/") or path.startswith("res/") , 'Path should start with ./res/ or res/'

    with open(assemble_project_path(path), "r", encoding="utf-8") as fd:
        return fd.read()
    
def init_path(path):
    os.makedirs(path, exist_ok=True)
    return path