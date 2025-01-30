import os
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from src.query.query_types import extract_query_type
from src.query.diverse_query import DiverseQuery
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

def prepare_low_level_reflection_params(state: Dict,
                                        info: Dict,
                                        params: Dict,
                                        memory: MemoryInterface = None,
                                        provider: EmbeddingProvider = None,
                                        diverse_query: DiverseQuery = None
                                        ):
    
    llr_params = deepcopy(params)

    low_level_reflection_query = params["low_level_reflection_query"]
    low_level_reflection_reasoning = params["low_level_reflection_reasoning"]

    low_level_reflection_short_term_reasoning = low_level_reflection_reasoning["short_term_reasoning"]
    low_level_reflection_medium_term_reasoning = low_level_reflection_reasoning["medium_term_reasoning"]
    low_level_reflection_long_term_reasoning = low_level_reflection_reasoning["long_term_reasoning"]

    # Grab latest low level reflection data and format as string
    date = info["date"]
    latest_llr = f"""Date: {date}\n
                    Short-Term reasoning: {low_level_reflection_short_term_reasoning}.\n
                    Medium-Term reasoning: {low_level_reflection_medium_term_reasoning}.\n
                    Long-Term reasoning: {low_level_reflection_long_term_reasoning}.\n"""
    
    # Setup query params
    query_params = {
        "type": "low_level_reflection",
        "symbol": params["asset_symbol"],
        "query_text": low_level_reflection_query
    }

    # Query memory for similar low level reflections
    query_result = diverse_query.query(params=query_params,
                                       query_types=["plain"])
    
    past_llrs = list()
    for query_type, values in query_result.items():
        query_items = values["query_items"]

        # Aggregate all past low level reflections for the current query type
        query_type_reflections = list()
        for reflection in query_items:

            # Grab the reasonings for each time horizon (e.g. short, medium, and long term)
            reasoning = reflection["reasoning"]
            short_term_reasoning = reasoning["short_term_reasoning"]
            medium_term_reasoning = reasoning["medium_term_reasoning"]
            long_term_reasoning = reasoning["long_term_reasoning"]

            # Store all reflections in a corresponding string and append
            reflection_text = f"""Date: {reflection['date']}\n
                                  Short-Term reasoning: {short_term_reasoning}\n
                                  Medium-Term reasoning: {medium_term_reasoning}\n
                                  Long-Term reasoning: {long_term_reasoning}\n"""
            query_type_reflections.append(reflection_text)
        
        if len(query_type_reflections) != 0:
            type_text = "\n\n".join(query_type_reflections)
            type_text = "The past low level reflection for " + query_type + " is:\n" + type_text
            past_llrs.append(type_text)

    if len(past_llrs) != 0:
        past_low_level_reflection = "\n\n".join(past_llrs)
    else:
        past_low_level_reflection = "There is no past low level reflection as it is trading initialised."

    llr_params.update({
        "latest_low_level_reflection": latest_llr,
        "past_low_level_reflection": past_low_level_reflection,
    })

    return llr_params 

