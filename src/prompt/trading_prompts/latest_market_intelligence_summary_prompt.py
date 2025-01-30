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
class LatestMarketIntelligenceSummaryPrompt(YamlPrompt):
    def __init__(self, model, template_path: str) -> None:
        self.model = model
        super(LatestMarketIntelligenceSummaryPrompt, self).__init__(template_path)
        
    def _convert_to_params(self,
                          state: Dict,
                          info: Dict,
                          params: Dict,
                          memory: MemoryInterface,
                          provider: EmbeddingProvider,
                          diverse_query: DiverseQuery = None) -> Dict:
        result_params = deepcopy(params)
        asset_info = ASSET.get_asset_info(info["symbol"])
        
        asset_name = asset_info["company_name"]
        asset_symbol = asset_info["symbol"]
        asset_exchange = asset_info["exchange"]
        asset_sector = asset_info["sector"]
        asset_description = asset_info["description"]
        current_date = info["date"]
        
        price = deepcopy(state["price"])
        news = deepcopy(state["news"])
        
        price = price[price.index == current_date]
        news = news[news.index == current_date]
        
        if len(news) > 20:
            news = news.sample(n=20)
            
        latest_market_intelligence_text = f"Date: Today is {current_date}.\n"
        
        if len(price) > 0:
            open = price["open"].values[0]
            high = price["high"].values[0]
            low = price["low"].values[0]
            close = price["close"].values[0]
            volume = price["volume"].values[0]
            latest_market_intelligence_text += f"Prices: Open: ({open}), High: ({high}), Low: ({low}), Close: ({close}), Volume: ({volume})\n"
        else:
            latest_market_intelligence_text += f"Prices: No prices for today.\n"
            
        if len(news) > 0:
            latest_market_intelligence_list = []
            
            for row in news.iterrows():
                row = row[1]
                news_id = row["id"]
                title = row["title"]
                text = row["text"]
                
                latest_market_intelligence_item = f"ID: {news_id}\n" \
                                          f"Headline: {title}\n" \
                                          f"Content: {text}\n"   
                latest_market_intelligence_list.append(latest_market_intelligence_item)        

            if len(latest_market_intelligence_list) == 0:
                latest_market_intelligence_text += "There is no latest market intelligence.\n"
            else:
                news_text = "\n".join(latest_market_intelligence_list)
                latest_market_intelligence_text += news_text
        else:
            latest_market_intelligence_text += "There is no latest market intelligence.\n"
            
        result_params.update({
            "date": current_date,
            "asset_name": asset_name,
            "asset_symbol": asset_symbol,
            "asset_exchange": asset_exchange,
            "asset_sector": asset_sector,
            "asset_description": asset_description,
            "latest_market_intelligence": latest_market_intelligence_text,
        })
        
        return result_params
        
    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):

        check_keys = [
            "query",
            "summary"
        ]

        response_dict = super(LatestMarketIntelligenceSummaryPrompt, self).get_response(provider=provider,
                                                                                        model=model,
                                                                                        messages=messages,
                                                                                        check_keys=check_keys)

        return response_dict
    
    def add_to_memory(self,
                      state: Dict,
                      info: Dict,
                      result: Dict,
                      memory: MemoryInterface = None,
                      provider: EmbeddingProvider = None) -> None:
        response_dict = deepcopy(result["response_dict"])
        
        current_date = info["date"]
        symbol = info["symbol"]
        
        price = deepcopy(state["price"])
        news = deepcopy(state["news"])
        
        price = price[price.index == current_date]
        news = news[news.index == current_date]

        if len(price) > 0:
            open = price["open"].values[0]
            high = price["high"].values[0]
            low = price["low"].values[0]
            close = price["close"].values[0]
            volume = price["volume"].values[0]
        else:
            open = math.nan
            high = math.nan
            low = math.nan
            close = math.nan
            volume = math.nan
            
        for row in news.iterrows():
            date = row[0] if isinstance(row[0], str) else row[0].strftime("%Y-%m-%d")
            row = row[1]

            id = row["id"]
            title = row["title"]
            text = row["text"]

            embedding_text = f"Heading: {title}\n" + \
                             f"Content: {text}\n"

            embedding = provider.embed_query(embedding_text)

            data = {
                "date": date,
                "id": id,
                "title": title,
                "text": text,
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "query": response_dict["query"],
                "summary": response_dict["summary"],
                "embedding_text": embedding_text,
                "embedding": embedding,
            }

        memory.add_memory(memory_type="market_intelligence",
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
        
        print(">" * 50 + f"{info['date']} - Running Latest Market Intelligence Summary Prompt" + ">" * 50)
        
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
        query = response_dict["query"]
        summary = response_dict["summary"]
        
        result = {
            "params": task_params,
            "message": message,
            "response_dict": response_dict,
        }
        
        params.update(task_params)

        params.update({
            "latest_market_intelligence_query": query,
            "latest_market_intelligence_summary": summary,
        })
        
        print("<" * 50 + f"{info['date']} - Finished Running the Latest Market Intelligence Summary Prompt" + "<" * 50)
        return result