import json
import os
import backoff
import yaml
import abc
from typing import Dict, Any, List
from jinja2 import Environment, BaseLoader, Template, exceptions as jinja2_exceptions

from src.provider.provider import encode_image

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
         
        with open(self.template_path, 'r') as file:
            try:
                template = yaml.safe_load(file)
                if 'messages' not in self.template:
                    raise ValueError("Main template YAML must contain a 'messages' key.")
                return template
            except yaml.YAMLError as e:
                raise ValueError(f'Error parsing YAML file: {e}')
            
    def _get_placeholders(self) -> List[str]:
        """
        Extracts all unique placeholders from the main template's messages.

        :return: List of placeholder names.
        """
        import re
        placeholders = set()
        pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
        for message in self.template['messages']:
            content = message.get('content', '')
            matches = pattern.findall(content)
            placeholders.update(matches)
        return list(placeholders)
    
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
    
    @abc.abstractmethod
    def assemble_prompt(
        self,
        *args,
        template: Any = None,
        params: Dict = None,
        **kwargs,
    ) -> List[str]:
        pass
        