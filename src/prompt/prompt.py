import json
import os
import backoff
import yaml
import abc
from copy import deepcopy
from typing import Dict, Any, List
from jinja2 import Environment, BaseLoader, Template, exceptions as jinja2_exceptions

from src.memory import MemoryInterface
from src.asset import ASSET
from src.provider.provider import encode_image
from src.utils.file_utils import read_resource_file

class YamlPrompt():
    '''General prompt class for all prompt templates in YAML format.'''
    
    def __init__(self, 
                 template_path: str) -> None:
        """
        Initializes the Prompt class with a YAML template.

        :param template_path: Path to the YAML template file.
        """
        self.template_path = template_path
        self.template = self._load_template()
        self.env = Environment(loader=BaseLoader())
        
    def _load_template(self) -> Dict[str, Any]:
        '''
        Loads the YAML template from the specified file.

        :return: Dictionary representation of the YAML template.
        '''
        if not os.path.exists(self.template_path):
             raise FileNotFoundError(f"Template file not found: {self.template_path}")
        
        raw_template = read_resource_file(self.template_path)
        print(raw_template)
        try:
            template = yaml.safe_load(raw_template)
            print(template)
            if 'messages' not in template:
                raise ValueError("Main template YAML must contain a 'messages' key.")
            return template
        except yaml.YAMLError as e:
            raise ValueError(f'Error parsing YAML file: {e}')
            
    def _get_placeholders(self) -> dict[set]:
        """
        Extracts all unique placeholders from the main template's messages.

        :return: Dict of placeholder names for each role.
        """
        import re
        placeholders = dict()
        pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
        for message in self.template['messages']:
            placeholds = list()
            role = message.get('role', '')
            content = message.get('content', '')
            matches = pattern.findall(content)
            placeholds.append(matches)
            placeholders[role] = placeholds
        return placeholders
    
    def render_template(self, template_str: str, params: Dict[str, Any]) -> str:
        """
        Renders a template string using Jinja2 with provided parameters.

        :param template_str: Template string with placeholders.
        :param params: Dictionary containing values to replace placeholders.
        :return: Rendered string.
        """
        try:
            template = self.env.from_string(template_str)
            rendered = template.render(**params).strip()
            print(rendered)
            return rendered
        except jinja2_exceptions.TemplateError as e:
            raise ValueError(f"Error rendering template: {e}")
        
    def _convert_to_params(self,
                           *args,
                           **kwargs) -> Dict:
        raise NotImplementedError

    def assemble_messages(
        self,
        *args,
        params: Dict = None,
        **kwargs,
    ) -> List[str]:
        
        # Get placeholders for asset and sub-prompt info
        placeholders = self._get_placeholders()
        
        # Create the system message
        system_message_content = ""
        for placeholder_list in placeholders['system']:
            for placeholder in placeholder_list:
                if ASSET.check_task_prompts(name=placeholder):
                    
                    # Fetch and render the sub-prompt
                    sub_prompt_content = ASSET.get_task_prompts(name=placeholder)
                    rendered_sub_prompt = self.render_template(template_str=sub_prompt_content,
                                                            params=params)
                    system_message_content += rendered_sub_prompt
        
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message_content
                }
            ]
        }
        
        # Create the user message text message
        user_message_content = ""
        user_messages = []
        for placeholder_list in placeholders['user']:
            for placeholder in placeholder_list:
                if ASSET.check_task_prompts(name=placeholder):
                    
                    # Fetch and render the sub-prompt
                    sub_prompt_content = ASSET.get_task_prompts(name=placeholder)
                    rendered_sub_prompt = self.render_template(template_str=sub_prompt_content,
                                                            params=params)
                    user_message_content += rendered_sub_prompt
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message_content
                }
            ]
        }
        user_messages.append(user_message)
        
        return [system_message] + user_messages
    
    def extract_yaml(self,
                     response: str) -> str:
        """
        Extracts the YAML content enclosed within ```yaml and ``` from the response string.

        :param response: The full response string containing YAML code block.
        :return: Extracted YAML string.
        :raises ValueError: If no YAML code block is found.
        """
        import re
        pattern = r'```yaml\s*\n(.*?)```'
        
        # Search for the pattern with DOTALL to include newlines
        match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
        
        if match:
            yaml_content = match.group(1).strip()
            print(yaml_content)
            return yaml_content
        else:
            raise ValueError("No YAML code block found in the response.")

    def get_response(self,
                     provider,
                     messages,
                     model = None,
                     check_keys=["action", "reasoning"]):
        
        response, info = provider.create_completion(messages=messages, 
                                                    model=model)
        print("response from llm model {}: \ninfo: {}\nresponse: \n{}".format(model, info, response))
        
        yaml_content = self.extract_yaml(response=response)
        response_dict = yaml.safe_load(yaml_content)
        print("response_dict: \n{}\n".format(response_dict))
        
        for key in check_keys:
            if key not in response_dict["output"]:
                raise KeyError(f"Key {key} not in response: {response_dict}")
        return response_dict
                
        
        
        
        

    