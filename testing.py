from src.utils.file_utils import *
from prompt.trading_prompts.latest_market_intelligence_summary_prompt import LatestMarketIntelligenceSummaryPrompt
from prompt.trading_prompts.past_market_intelligence_summary_prompt import PastMarketIntelligenceSummaryPrompt
from prompt.trading_prompts.decision_prompt import DecisionPrompt
from src.provider.provider import OpenAIProvider

import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv(verbose=True)

# template = read_resource_file("./res/prompts/templates/decision_template.yaml")
# print(template)
# print(type(template))

#template_path = "res/prompts/templates/latest_market_intelligence_summary.yaml"
template_path = "res/prompts/templates/decision_template.yaml"
# prompt = LatestMarketIntelligenceSummaryPrompt(model="gpt-4o",
#                                                         template_path=template_path)
prompt = DecisionPrompt(model="gpt-4o",
                        template_path=template_path)

params = dict()
params["past_market_intelligence"] = "test past market intelligence string"
params["asset_symbol"] = "BTC-USD"
params["asset_name"] = "Bitcoin"
params["asset_exchange"] = "coinbase"
params["asset_sector"] = "cryptocurrency"
params["asset_description"] = "bitcoin is a coin"
params["trader_preference"] = "aggressive"
params["past_market_intelligence_summary"] = "No past market intelligence summary."
params["latest_market_intelligence_summary"] = "No latest market intelligence summary."
state = dict()
state["price"] = pd.DataFrame()
state["news"] = pd.DataFrame()
info = dict()
info['price'] = 100.123
info['position'] = 1
info['total_profit'] = 2.123
info['total_return'] = .1
info['date'] = "2024-12-13"
info['symbol'] = "BTC-USD"
config_path = "configs/provider_configs/openai_config.json"
provider = OpenAIProvider(config_path)
prompt.run(state=state,
                    info=info,
                    params=params,
                    provider=provider
                    )

