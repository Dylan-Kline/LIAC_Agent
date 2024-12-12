from src.utils.file_utils import *
from src.prompt.prompt import YamlPrompt
from src.provider.provider import OpenAIProvider

import os

from dotenv import load_dotenv
load_dotenv(verbose=True)

# template = read_resource_file("./res/prompts/templates/decision_template.yaml")
# print(template)
# print(type(template))

template_path = "res/prompts/templates/decision_template.yaml"
decision_prompt = YamlPrompt(template_path=template_path)
params = dict()
messages = decision_prompt.assemble_messages(params=params)
config_path = "configs/provider_configs/openai_config.json"
provider = OpenAIProvider(config_path)
decision_prompt.get_response(provider=provider,
                             messages=messages)

