from src.utils.file_utils import *
from src.prompt.trading_prompts.latest_market_intelligence_summary_prompt import LatestMarketIntelligenceSummaryPrompt
from src.prompt.trading_prompts.past_market_intelligence_summary_prompt import PastMarketIntelligenceSummaryPrompt
from src.prompt.trading_prompts.decision_prompt import DecisionPrompt
from src.prompt.trading_prompts.low_level_reflection_prompt import LowLevelReflectionPrompt
from src.provider.provider import OpenAIProvider

import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv(verbose=True)

# template = read_resource_file("./res/prompts/templates/decision_template.yaml")
# print(template)
# print(type(template))

#template_path = "res/prompts/templates/latest_market_intelligence_summary.yaml"
template_path = "res/prompts/templates/train/train-mi-w-low-w-decision/low_level_reflection.yaml"
# prompt = LatestMarketIntelligenceSummaryPrompt(model="gpt-4o",
#                                                         template_path=template_path)
prompt = LowLevelReflectionPrompt(model="gpt-4o",
                                  template_path=template_path)

params = dict()
params["past_market_intelligence"] = "test past market intelligence string"
params["kline_path"] = "C:\projects\LIAC\LIAC_Agent\kline_chart.png"
params["asset_symbol"] = "BTC-USD"
params["asset_name"] = "Bitcoin"
params["asset_exchange"] = "coinbase"
params["asset_sector"] = "cryptocurrency"
params["asset_description"] = "bitcoin is a coin"
params["trader_preference"] = "aggressive"
params["past_market_intelligence_summary"] = "No past market intelligence summary."
params["latest_market_intelligence_summary"] = "No latest market intelligence summary."
state = dict()

np.random.seed(42)  # For reproducibility
days = 50
date_range = pd.date_range(start="2023-01-01", periods=days, freq="D")

# Generate random prices for open, high, low, and close
open_prices = np.random.uniform(100, 200, size=days)
high_prices = open_prices + np.random.uniform(0, 20, size=days)
low_prices = open_prices - np.random.uniform(0, 20, size=days)
close_prices = low_prices + np.random.uniform(0, high_prices - low_prices)

# Generate random volume data
volume = np.random.randint(1000, 10000, size=days)

# Create the DataFrame
kline_data = pd.DataFrame({
    "timestamp": date_range,
    "open": open_prices,
    "high": high_prices,
    "low": low_prices,
    "close": close_prices,
    "volume": volume
})
df = pd.DataFrame(kline_data).set_index("timestamp")

state["price"] = df
state["news"] = pd.DataFrame()
info = dict()
info['price'] = 100.123
info['cash'] = 10000.00
info['position'] = 1
info['total_profit'] = 2.123
info['total_return'] = .1
info['date'] = "2023-01-29"
info['symbol'] = "BTC-USD"
config_path = "configs/provider_configs/openai_config.json"
provider = OpenAIProvider(config_path)
prompt.run(state=state,
                    info=info,
                    params=params,
                    provider=provider
                    )

