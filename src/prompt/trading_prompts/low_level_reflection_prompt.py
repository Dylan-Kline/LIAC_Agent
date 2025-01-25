import math
import os
import backoff
import pandas as pd
from typing import Dict, List, Any
from copy import deepcopy

from src.prompt import YamlPrompt
from src.asset import ASSET
from src.memory import MemoryInterface
from src.provider import EmbeddingProvider
from src.query import DiverseQuery
from src.registry import PROMPT

@PROMPT.register_module(force=True)
class LowLevelReflectionPrompt(YamlPrompt):

    def __init__(self,
                 *args,
                 model: Any = None,
                 template_path,
                 short_term_past_date_range: int = 1,
                 medium_term_past_date_range: int = 7,
                 long_term_past_date_range: int = 14,
                 short_term_next_date_range: int = 1,
                 medium_term_next_date_range: int = 7,
                 long_term_next_date_range: int = 14,
                 look_back_days: int = 14,
                 look_forward_days: int = 14,
                 **kwargs):

        self.model = model
        self.short_term_past_date_range = short_term_past_date_range
        self.medium_term_past_date_range = medium_term_past_date_range
        self.long_term_past_date_range = long_term_past_date_range
        self.short_term_next_date_range = short_term_next_date_range
        self.medium_term_next_date_range = medium_term_next_date_range
        self.long_term_next_date_range = long_term_next_date_range
        self.look_back_days = look_back_days
        self.look_forward_days = look_forward_days

        super(LowLevelReflectionPrompt, self).__init__(template_path=template_path)

    def _convert_to_price_movement(self, state: Dict, current_date: str = None):

        def price_movement_to_text(x):
            if x > 0:
                return "an increase of {:.2f}%".format(abs(x * 100))
            elif x < 0:
                return "a decrease of {:.2f}%".format(abs(x * 100))
            elif x == 0:
                return "no change"
            else:
                return "unknown"
            
        price = state["price"]
        price = deepcopy(price)
        price = price.reset_index(drop=False)
        price = price[["timestamp", "close"]]
        price = price.dropna(axis=0, how="any")
        price = price.drop_duplicates(subset=["timestamp"], keep="first")

        past_price = price[price["timestamp"] <= current_date]
        next_price = price[price["timestamp"] >= current_date]

        short_term_past_price_movement = past_price["close"].pct_change(periods=self.short_term_past_date_range).iloc[-1]
        medium_term_past_price_movement = past_price["close"].pct_change(periods=self.medium_term_past_date_range).iloc[-1]
        long_term_past_price_movement = past_price["close"].pct_change(periods=self.long_term_past_date_range).iloc[-1]
        short_term_past_price_movement_text = price_movement_to_text(short_term_past_price_movement)
        medium_term_past_price_movement_text = price_movement_to_text(medium_term_past_price_movement)
        long_term_past_price_movement_text = price_movement_to_text(long_term_past_price_movement)

        short_term_next_price_movement = next_price["close"].pct_change(periods=self.short_term_next_date_range).shift(-self.short_term_next_date_range).iloc[0]
        medium_term_next_price_movement = next_price["close"].pct_change(periods=self.medium_term_next_date_range).shift(-self.medium_term_next_date_range).iloc[0]
        long_term_next_price_movement = next_price["close"].pct_change(periods=self.long_term_next_date_range).shift(-self.long_term_next_date_range).iloc[0]
        short_term_next_price_movement_text = price_movement_to_text(short_term_next_price_movement)
        medium_term_next_price_movement_text = price_movement_to_text(medium_term_next_price_movement)
        long_term_next_price_movement_text = price_movement_to_text(long_term_next_price_movement)

        res = {
            "short_term_past_price_movement": str(short_term_past_price_movement_text),
            "medium_term_past_price_movement": str(medium_term_past_price_movement_text),
            "long_term_past_price_movement": str(long_term_past_price_movement_text),
            "short_term_past_date_range": int(self.short_term_past_date_range),
            "medium_term_past_date_range": int(self.medium_term_past_date_range),
            "long_term_past_date_range": int(self.long_term_past_date_range),
            "short_term_next_price_movement": str(short_term_next_price_movement_text),
            "medium_term_next_price_movement": str(medium_term_next_price_movement_text),
            "long_term_next_price_movement": str(long_term_next_price_movement_text),
            "short_term_next_date_range": int(self.short_term_next_date_range),
            "medium_term_next_date_range": int(self.medium_term_next_date_range),
            "long_term_next_date_range": int(self.long_term_next_date_range),
        }

        return res

    def _convert_to_params(self,
                         state: Dict,
                         info: Dict,
                         params: Dict,
                         memory: MemoryInterface = None,
                         provider: EmbeddingProvider = None,
                         diverse_query: DiverseQuery = None) -> Dict:

        res_params = deepcopy(params)

        current_date = info["date"]
        price_movement = self._convert_to_price_movement(state, current_date=current_date)

        res_params.update(price_movement)

        return res_params

    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):

        check_keys = [
            "reasoning",
            "query"
        ]

        response = super(LowLevelReflectionPrompt, self).get_response(provider=provider,
                                                            messages=messages,
                                                            model=model,
                                                            check_keys=check_keys)
        return response
    
    def add_to_memory(self,
                     state: pd.DataFrame = None,
                     info: Dict = None,
                     res: Dict = None,
                     memory: MemoryInterface = None,
                     provider: EmbeddingProvider = None) -> None:
        symbol = info["symbol"]

        data = deepcopy(res["params"])
        response_dict = deepcopy(res["response_dict"])

        embedding_text = response_dict["query"]
        embedding = provider.embed_query(embedding_text)

        data.update({
            "embedding_text": embedding_text,
            "embedding": embedding,
            "reasoning": response_dict["reasoning"],
            "query": response_dict["query"],
        })

        memory.add_memory(type="low_level_reflection",
                          symbol=symbol,
                          data=data,
                          embedding_key="embedding")

    def run(self,
            state: Dict,
            info: Dict,
            params: Dict,
            memory: MemoryInterface = None,
            provider: EmbeddingProvider = None,
            diverse_query: DiverseQuery = None,
            **kwargs):
        
        print(">" * 50 + f"{info['date']} - Running Low Level Reflection Prompt" + ">" * 50)
        
        task_params = self._convert_to_params(state=state,
                                             info=info,
                                             params=params,
                                             memory=memory,
                                             provider=provider,
                                             diverse_query=diverse_query)
        message = self.assemble_messages(params=task_params)
        response_dict = self.get_response(provider=provider,
                                               model=self.model,
                                               messages=message)
        response_dict = response_dict['output']
        reasoning = response_dict["reasoning"]
        query = response_dict["query"]
        
        result = {
            "params": task_params,
            "message": message,
            "response_dict": response_dict,
        }
        
        params.update(task_params)

        params.update({
            "low_level_reflection_reasoning": reasoning,
            "low_level_reflection_query": query,
        })
        
        print("<" * 50 + f"{info['date']} - Finished Running the Low Level Reflection Prompt" + "<" * 50)
        return result