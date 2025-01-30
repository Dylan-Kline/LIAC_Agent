import math
import os
import backoff
from typing import Dict, List, Any
from copy import deepcopy

from src.prompt import YamlPrompt
from src.asset import ASSET
from src.memory import MemoryInterface
from src.provider import EmbeddingProvider
from src.query import DiverseQuery
from src.registry import PROMPT

@PROMPT.register_module(force=True)
class DecisionPrompt(YamlPrompt):

    def __init__(self,
                 model,
                 template_path):
        self.model = model
        super(DecisionPrompt, self).__init__(template_path=template_path)

    def _convert_to_params(self,
                           state: Dict,
                            info: Dict,
                            params: Dict,
                            memory: MemoryInterface = None,
                            provider: EmbeddingProvider = None,
                            diverse_query: DiverseQuery=None) -> Dict:
        
        res_params = deepcopy(params)

        def convert_to_text(ret):
            '''Converts the given asset return into a text string saying what percent it has changed.'''
            absolute_percentage = abs(ret*100)
            if ret >= 0:
                return "an increase of {:.2f}%".format(absolute_percentage)
            elif ret < 0:
                return "a decrease of {:.2f}%".format(absolute_percentage)
            
        asset_price = info["price"]
        asset_cash = info["cash"]
        asset_position = info["position"]
        asset_profit = info["total_profit"]
        asset_return = info["total_return"]

        asset_price = "{:.2f}".format(asset_price)
        asset_cash = "{:.2f}".format(asset_cash)
        asset_position = "{}".format(int(asset_position))
        asset_profit = "{:.2f}%".format(asset_profit)
        asset_return = convert_to_text(asset_return)

        trader_preference = params["trader_preference"]

        res_params.update({
            "asset_price": asset_price,
            "asset_cash": asset_cash,
            "asset_position": asset_position,
            "asset_profit": asset_profit,
            "asset_return": asset_return,
            "trader_preference": trader_preference,
        })

        return res_params
    
    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):
        
        check_keys = ["action",
                      "reasoning"]
        
        response = super(DecisionPrompt, self).get_response(provider=provider,
                                                            messages=messages,
                                                            model=model,
                                                            check_keys=check_keys)
        response["output"]["action"] = response["output"]["action"].replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        return response
    
    def run(self,
            state: Dict,
            info: Dict,
            params: Dict,
            memory: MemoryInterface = None,
            provider: EmbeddingProvider = None,
            diverse_query: DiverseQuery = None,
            **kwargs):
        
        print(">" * 50 + f"{info['date']} - Running Decision Prompt" + ">" * 50)
        
        task_params = self._convert_to_params(state=state,
                                             info=info,
                                             params=params,
                                             memory=memory,
                                             provider=provider,
                                             diverse_query=diverse_query)
        message = self.assemble_messages(params=task_params)
        response = self.get_response(provider=provider,
                                               model=self.model,
                                               messages=message)
        
        response = response['output']
        action = response["action"]
        reasoning = response["reasoning"]
        
        result = {
            "params": task_params,
            "message": message,
            "response_dict": response,
        }
        
        params.update(task_params)

        params.update({
            "decision_action": action,
            "decision_reasoning": reasoning,
        })
        
        print("<" * 50 + f"{info['date']} - Finish Running Decision Prompt" + "<" * 50)
        return result