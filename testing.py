from src.utils.file_utils import *
from src.prompt.latest_market_intelligence_summary_prompt import LatestMarketIntelligenceSummaryPrompt
from src.prompt.past_market_intelligence_summary_prompt import PastMarketIntelligenceSummaryPrompt
from src.provider.provider import OpenAIProvider

import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv(verbose=True)

# template = read_resource_file("./res/prompts/templates/decision_template.yaml")
# print(template)
# print(type(template))

template_path = "res/prompts/templates/latest_market_intelligence_summary.yaml"
lmi_prompt = LatestMarketIntelligenceSummaryPrompt(model="gpt-4o",
                                                        template_path=template_path)
params = dict()
params["past_market_intelligence"] = "test past market intelligence string"
params["asset_symbol"] = "BTC-USD"
params["asset_name"] = "Bitcoin"
params["asset_exchange"] = "coinbase"
params["asset_sector"] = "cryptocurrency"
params["asset_description"] = "bitcoin is a coin"
state = dict()
state["price"] = pd.DataFrame()
state["news"] = pd.DataFrame()
info = dict()
info['date'] = "2024-12-13"
info['symbol'] = "BTC-USD"
config_path = "configs/provider_configs/openai_config.json"
provider = OpenAIProvider(config_path)
lmi_prompt.run(state=state,
                    info=info,
                    params=params,
                    provider=provider
                    )

