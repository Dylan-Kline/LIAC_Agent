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
        try:
            template = yaml.safe_load(raw_template)
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
    
    def _get_path_placeholder(self, text) -> str:
        '''
        Extracts the image path placeholder from the given text.
        :return: String containing the placeholder text
        '''
        import re
        pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
        image_path_placeholder = pattern.findall(text)
        return image_path_placeholder[0]
    
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
        system_message_content = self.template["messages"][0].get('content', '')
        for placeholder_list in placeholders['system']:
            for placeholder in placeholder_list:

                placeholder_replaced = "{{" + f"{placeholder}" + "}}"

                if ASSET.check_task_prompts(name=placeholder):
                    
                    # Fetch and render the sub-prompt
                    sub_prompt_content = ASSET.get_task_prompts(name=placeholder)
                    rendered_sub_prompt = self.render_template(template_str=sub_prompt_content,
                                                            params=params)
                    system_message_content = system_message_content.replace(placeholder_replaced, 
                                                   rendered_sub_prompt)
                    
                elif placeholder in params:
                    system_message_content = system_message_content.replace(placeholder_replaced,
                                                                            params[placeholder])
                    # TODO may need to convert the params value to a str if its a price
        
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
        user_message_content = self.template['messages'][1].get('content', '')
        user_messages = []
        image_message = None
        for placeholder_list in placeholders['user']:
            for placeholder in placeholder_list:

                placeholder_replaced = "{{" + f"{placeholder}" + "}}"

                if ASSET.check_task_prompts(name=placeholder):
                    
                    # Fetch and render the sub-prompt
                    sub_prompt_content = ASSET.get_task_prompts(name=placeholder)
                    rendered_sub_prompt = self.render_template(template_str=sub_prompt_content,
                                                            params=params)
                    user_message_content = user_message_content.replace(placeholder_replaced,
                                                                        rendered_sub_prompt)
                elif placeholder in params:
                    user_message_content = user_message_content.replace(placeholder_replaced,
                                                                        params[placeholder])

                potential_image_name = placeholder + "image"
                if ASSET.check_task_prompts(name=potential_image_name):
                    
                    # Fetch and encode the image
                    image_content = ASSET.get_task_prompts(name=potential_image_name)
                    image_placeholder = self._get_path_placeholder(image_content)
                    str_to_replace = "{{" + f"{image_placeholder}" + "}}"
                    
                    if image_placeholder in params:
                        image_content = image_content.replace(str_to_replace,
                                                            params[image_placeholder]).strip()
                    else:
                        print("No image path found for kline chart.")
                        
                    image_base64 = encode_image(image_path=image_content)
                    image_message = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message_content
                },
            ]
        }
        if image_message:
            user_message["content"].append(image_message)
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
        
        for key in check_keys:
            if key not in response_dict["output"]:
                raise KeyError(f"Key {key} not in response: {response_dict}")
        return response_dict
                
        
        
        
        

    