import os
from typing import Dict, Any
from glob import glob
from copy import deepcopy
import json
import yaml

from src.utils.file_utils import assemble_project_path
from src.utils.singleton import Singleton

class Asset(metaclass=Singleton):
    def __init__(self):
        self.assets = self._load_assets()

    def _load_assets(self) -> Dict[str, Any]:
        assets = {}
        assets["asset_infos"] = self._load_asset_infos()
        assets["traders"] = self._load_traders()
        assets["task_prompts"] = self._load_task_prompts()
        return assets

    def _load_traders(self) -> Dict[str, str]:
        traders_dir_path = assemble_project_path("res/prompts/traders")
        traders_paths = glob(os.path.join(traders_dir_path, "**", "*.txt"), recursive=True)
        traders = {}
        
        for trader_path in traders_paths:
            name = os.path.basename(trader_path).replace(".txt", "")
            with open(trader_path, "r") as f:
                text = f.read().strip()
                traders[name] = text
                
        return traders

    def _load_asset_infos(self)->Dict[str, str]:
        asset_infos_dir_path = assemble_project_path("res/asset_infos")
        asset_infos_paths = glob(os.path.join(asset_infos_dir_path, "**", "*.json"), recursive=True)
        asset_infos = {}
        
        for asset_info_path in asset_infos_paths:
            with open(asset_info_path, "r") as f:
                asset_info = json.load(f)
                for k, v in asset_info.items():
                    if k not in asset_infos:
                        asset_infos[k] = v

        return asset_infos

    def _load_task_prompts(self) -> Dict[str, str]:
        task_prompts_dir_path = assemble_project_path("res/prompts/task_prompts")
        task_prompts_paths = glob(os.path.join(task_prompts_dir_path, "*.yaml"), recursive=True)
        task_prompts = {}
        
        for task_prompts_path in task_prompts_paths:
            with open(task_prompts_path, "r") as f:
                task_prompt = yaml.safe_load(f)
                name = os.path.basename(task_prompts_path).replace(".yaml", "")
                for _, value in task_prompt.items():
                    if 'text' in value:
                        task_prompts[name] = value['text']
                    if 'image' in value:
                        task_prompts[name + "image"] = value['image']
                        
        return task_prompts

    def check_asset_info(self, symbol: str = None) -> bool:
        return symbol in self.assets["asset_infos"]

    def get_asset_info(self, symbol: str = None) -> Dict[str, Any]:
        return deepcopy(self.assets["asset_infos"][symbol])

    def check_trader(self, name: str = None) -> bool:
        return name in self.assets["traders"]

    def get_trader(self, name: str = None) -> str:
        return deepcopy(self.assets["traders"][name])

    def check_task_prompts(self, name: str = None) -> bool:
        return name in self.assets["task_prompts"]

    def get_task_prompts(self, name: str = None) -> str:
        return deepcopy(self.assets["task_prompts"][name])

ASSET = Asset()

if __name__ == "__main__":
    print(json.dumps(ASSET.assets, indent=4))