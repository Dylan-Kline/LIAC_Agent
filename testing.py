from src.utils.file_utils import *

template = read_resource_file("./res/prompts/templates/decision_template.yaml")
print(template)
print(type(template))