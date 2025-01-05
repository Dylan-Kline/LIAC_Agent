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

class PastMarketIntelligenceSummaryPrompt(YamlPrompt):

    def __init__(self,
                 model,
                 template_path: str):
        self.model = model
        super(PastMarketIntelligenceSummaryPrompt, self).__init__(template_path=template_path)
    
    def _convert_to_params(self,
                            state: Dict,
                            info: Dict,
                            params: Dict,
                            memory: MemoryInterface,
                            provider: EmbeddingProvider,
                            diverse_query: DiverseQuery = None) -> Dict:
        res_params = deepcopy(params)
        return res_params
    
    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response(self,
                    provider,
                    model,
                    messages,
                    check_keys: List[str] = None):

        check_keys = [
            "summary"
        ]

        response_dict = super(PastMarketIntelligenceSummaryPrompt, self).get_response(provider=provider,
                                                                                        model=model,
                                                                                        messages=messages,
                                                                                        check_keys=check_keys)

        return response_dict
    
    def add_to_memory(self,
                      state: Dict,
                      info: Dict,
                      params: Dict,
                      memory: MemoryInterface = None,
                      provider: EmbeddingProvider = None) -> None:
        raise NotImplementedError("PastMarketIntelligenceSummaryPrompt does not support add_to_memory")
    
    def run(self,
            state: Dict,
            info: Dict,
            params: Dict,
            memory: MemoryInterface = None,
            provider: EmbeddingProvider = None,
            diverse_query: DiverseQuery = None,
            **kwargs):
        
        print(">" * 50 + f"{info['date']} - Running Past Market Intelligence Summary Prompt" + ">" * 50)
        
        task_params = self.convert_to_params(state=state,
                                             info=info,
                                             params=params,
                                             memory=memory,
                                             provider=provider,
                                             diverse_query=diverse_query)
        message = self.assemble_messages(params=task_params)
        print(message)
        exit()
        response_dict = self.get_response(provider=provider,
                                               model=self.model,
                                               messages=message)
        response_dict = response_dict['output']
        summary = response_dict["summary"]
        
        result = {
            "params": task_params,
            "message": message,
            "response_dict": response_dict,
        }
        
        params.update(task_params)

        params.update({
            "past_market_intelligence_summary": summary,
        })
        
        print("<" * 50 + f"{info['date']} - Finish Running Past Market Intelligence Summary Prompt" + "<" * 50)
        return result