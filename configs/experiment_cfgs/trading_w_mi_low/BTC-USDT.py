root = None
selected_asset = "BTC-USDT"
asset_type = "crypto"
workdir = "workdir/"
memory_path = "memory"
tag = f"{selected_asset}_1st_train"

initial_amount = 1e5
transaction_cost_pct = 1.5e-4

# adjust the following parameters mainly
trader_preference = "moderate_trader"
train_start_date = "2023-07-14"
train_end_date = "2024-06-09"
valid_start_date = "2024-06-10"
valid_end_date = "2024-12-12"

short_term_past_date_range = 1
medium_term_past_date_range = 7
long_term_past_date_range = 14
short_term_next_date_range = 1
medium_term_next_date_range = 7
long_term_next_date_range = 14
look_forward_days = long_term_next_date_range
look_back_days = long_term_past_date_range
previous_action_look_back_days = 7
top_k = 5

train_latest_market_intelligence_summary_template_path = "res/prompts/templates/train/train-mi-w-low-w-decision/latest_market_intelligence_summary.yaml"
train_past_market_intelligence_summary_template_path = "res/prompts/templates/train/train-mi-w-low-w-decision/past_market_intelligence_summary.yaml"
train_low_level_reflection_template_path = "res/prompts/templates/train/train-mi-w-low-w-decision/low_level_reflection.yaml"
train_decision_template_path = "res/prompts/templates/train/train-mi-w-low-w-decision/decision_template.yaml"

valid_latest_market_intelligence_summary_template_path = "res/prompts/templates/valid/train-mi-w-low-w-decision/latest_market_intelligence_summary.yaml"
valid_past_market_intelligence_summary_template_path = "res/prompts/templates/valid/train-mi-w-low-w-decision/past_market_intelligence_summary.yaml"
valid_low_level_reflection_template_path = "res/prompts/templates/valid/train-mi-w-low-w-decision/low_level_reflection.yaml"
valid_decision_template_path = "res/prompts/templates/valid/train-mi-w-low-w-decision/decision_template.yaml"

dataset = dict(
    type="Dataset",
    root=root,
    price_path="datasets/exp_cryptos/price",
    news_path="datasets/exp_cryptos/news",
    interval="1d",
    assets_path="configs/_asset_lists_/exp_cryptos.txt",
    workdir=workdir,
    tag=tag
)

train_environment = dict(
    type="TradingEnvironment",
    mode="train",
    dataset=None,
    selected_asset=selected_asset,
    asset_type=asset_type,
    start_date=train_start_date,
    end_date=train_end_date,
    look_back_days=look_back_days,
    look_forward_days=look_forward_days,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    discount=1.0,
)

valid_environment = dict(
type="TradingEnvironment",
    mode="valid",
    dataset=None,
    selected_asset=selected_asset,
    asset_type=asset_type,
    start_date=valid_start_date,
    end_date=valid_end_date,
    look_back_days=look_back_days,
    look_forward_days=look_forward_days,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    discount=1.0,
)

plots = dict(
    type = "PlotsInterface",
    root = root,
    workdir = workdir,
    tag = tag,
)

memory = dict(
    type="MemoryInterface",
    root=root,
    symbols=None,
    memory_path=memory_path,
    embedding_dim=None,
    max_recent_steps=5,
    workdir=workdir,
    tag=tag
)

latest_market_intelligence_summary = dict(
    type="LatestMarketIntelligenceSummaryPrompt",
    model = "gpt-4o"
)

past_market_intelligence_summary = dict(
    type="PastMarketIntelligenceSummaryPrompt",
    model = "gpt-4o"
)

low_level_reflection = dict(
    type="LowLevelReflectionPrompt",
    model = "gpt-4o",
    short_term_past_date_range=short_term_past_date_range,
    medium_term_past_date_range=medium_term_past_date_range,
    long_term_past_date_range=long_term_past_date_range,
    short_term_next_date_range=short_term_next_date_range,
    medium_term_next_date_range=medium_term_next_date_range,
    long_term_next_date_range=long_term_next_date_range,
    look_back_days=long_term_past_date_range,
    look_forward_days=long_term_next_date_range
)

decision = dict(
    type="DecisionPrompt",
    model = "gpt-4o",
)

provider = dict(
    type="OpenAIProvider",
    provider_cfg_path="configs/provider_configs/openai_config.json",
)
