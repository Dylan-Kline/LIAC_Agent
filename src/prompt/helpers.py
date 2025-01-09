import os
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from query.query_types import extract_query_type
from query.diverse_query import DiverseQuery
from src.memory import MemoryInterface
from src.provider import EmbeddingProvider
ROOT = str(Path(__file__).resolve().parents[2])

def prepare_latest_market_intelligence_params(state: Dict,
                                            info: Dict,
                                            params: Dict,
                                            memory: MemoryInterface = None,
                                            provider: EmbeddingProvider = None,
                                            diverse_query: DiverseQuery = None
                                            ):

    res_params = deepcopy(params)

    latest_market_intelligence_query = params["latest_market_intelligence_query"]

    query_res = {}
    for query_type, quey_text in latest_market_intelligence_query.items():

        if len(quey_text) == 0 or len(quey_text.split(" ")) <= 5:
            continue

        query_params = {
            "type": "market_intelligence",
            "symbol": params["asset_symbol"],
            "query_text": quey_text,
        }

        query_type = extract_query_type(query_type)
        query_items = diverse_query.query(params=query_params,
                                          query_types=[query_type],
                                          top_k=3)[query_type]["query_items"]

        for item in query_items:
            id = item["id"]
            if id not in query_res:
                query_res[id] = item

    query_res = sorted(query_res.items(), key=lambda x: x[0], reverse=False)
    query_res = [item[1] for item in query_res]

    print(f"Number of queried past market intelligence: {len(query_res)}")

    past_market_intelligence_list = []
    for item in query_res:
        date = item["date"] if isinstance(item["date"], str) else item["date"].strftime("%Y-%m-%d")
        id = item["id"]
        title = item["title"]
        text = item["text"]
        open = item["open"]
        high = item["high"]
        low = item["low"]
        close = item["close"]
        volume = item["volume"]

        past_market_intelligence_query_item = f"Date: {date}.\n"

        past_market_intelligence_query_item += f"ID: {id}\n" + \
                                               f"Headline: {title}\n" + \
                                               f"Content: {text}\n"
        if math.isnan(open) == False:
            past_market_intelligence_query_item += f"Prices: Open: ({open}), High: ({high}), Low: ({low}), Close: ({close}), Volume: ({volume})\n"
        else:
            past_market_intelligence_query_item += f"Prices: Today is closed for trading.\n"

        past_market_intelligence_list.append(past_market_intelligence_query_item)

    if len(past_market_intelligence_list) == 0:
        past_market_intelligence_text = "There is no past market_intelligence.\n"
    else:
        past_market_intelligence_text = "\n".join(past_market_intelligence_list)

    res_params.update({
        "past_market_intelligence": past_market_intelligence_text,
    })

    return res_params