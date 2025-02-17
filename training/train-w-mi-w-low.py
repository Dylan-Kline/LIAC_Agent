import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(verbose=True)

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

from src.asset import ASSET
from src.utils import read_resource_file, save_json, load_json
from src.utils.misc import update_data_root
from src.query.diverse_query import DiverseQuery
from src.prompt.helpers import (prepare_latest_market_intelligence_params,
                             prepare_low_level_reflection_params)
from src.registry import *

from mmengine.config import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description="Train with mi and low.")
    parser.add_argument("--config", 
                        default=os.path.join(ROOT, "configs", "experiment_cfgs", "trading_w_mi_low", "BTC-USDT.py"),
                        help="config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument("--root", type=str, default=ROOT)
    parser.add_argument("--if_remove", action="store_true", default=False)

    parser.add_argument("--checkpoint_start_date", type=str, default=None)
    parser.add_argument("--if_load_memory", action="store_true", default=False)
    parser.add_argument("--memory_path", type=str, default="memory")
    parser.add_argument("--if_load_trading_record", action="store_true", default=False)
    parser.add_argument("--trading_record_path", type=str, default=None)
    parser.add_argument("--if_train", action="store_true", default=True)
    parser.add_argument("--if_valid", action="store_true", default=False)

    args = parser.parse_args()
    return args

def setup_environment():   
    # args.cfg_options is used to override any config parameters in the config file read by the mmengine config class

    # Parse config arguements
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Set up the override settings in the cfg_options dict
    if args.cfg_options is None:
        args.cfg_options = dict()
    if args.root is not None:
        args.cfg_options["root"] = args.root

    args.cfg_options["checkpoint_start_date"] = args.checkpoint_start_date
    args.cfg_options["if_load_memory"] = args.if_load_memory
    args.cfg_options["memory_path"] = args.memory_path
    args.cfg_options["if_load_trading_record"] = args.if_load_trading_record
    args.cfg_options["trading_record_path"] = args.trading_record_path

    if args.if_train is not None:
        args.cfg_options["if_train"] = args.if_train
    if args.if_valid is not None:
        args.cfg_options["if_valid"] = args.if_valid
        
    # Override any config parameters with the provided cfg_options in the parsed args
    cfg.merge_from_dict(args.cfg_options)

    update_data_root(cfg, root=args.root)

    # Set up experiment pathing
    experiment_path = os.path.join(cfg.root, 
                                   cfg.workdir,
                                   cfg.tag)
    
    # Determine if the experiment should be deleted or not
    if args.if_remove:
        import shutil
        shutil.rmtree(experiment_path,
                      ignore_errors=True)
        print(f"Arguments provide removed work_dir: {experiment_path}")
    else:
        print(f"Arguments provided keep work_dir: {experiment_path}")

    # Create the experiment directory and dump current experiment config to dir
    os.makedirs(experiment_path, exist_ok=True)
    cfg.dump(os.path.join(experiment_path, 'config.py'))

    return cfg, experiment_path

def main():

    cfg, experiment_path = setup_environment()

    # Initialize provider and training dataset
    provider = PROVIDER.build(cfg.provider)
    dataset = DATASET.build(cfg.dataset)
    
    # Initialize trading environment
    cfg.train_environment['dataset'] = dataset
    train_env = ENVIRONMENT.build(cfg.train_environment)
    cfg.valid_environment['dataset'] = dataset
    valid_env = ENVIRONMENT.build(cfg.valid_environment)
    
    # Init plots and memory
    plots = PLOTS.build(cfg.plots)
    cfg.memory["symbols"] = dataset.assets
    cfg.memory["embedding_dim"] = provider.get_embedding_dim()
    memory = MEMORY.build(cfg.memory)
    
    if cfg.memory_path is not None:
        memory_path = os.path.join(cfg.root, cfg.workdir, cfg.memory_path)
    
    # Load local memory
    if cfg.if_load_memory and cfg.memory_path is not None:
        print("Loading local memory...")
        memory.load_local(memory_path=memory_path)
    
    # Setup diverse query system and strategy agents if need be
    diverse_query = DiverseQuery(memory=memory, 
                                 provider=provider, 
                                 top_k=cfg.top_k)
    
    # Train
    if cfg.if_train:
        train_records = run(cfg,
                            train_env,
                            plots,
                            memory,
                            provider,
                            diverse_query,
                            experiment_path,
                            mode = "train")
        train_save_path = os.path.join(experiment_path, "train_records.json")

        memory.save_local(memory_path=memory_path)
        save_json(train_records, train_save_path)
    
    # Validate
    if cfg.if_valid:
        valid_records = run(cfg,
                            valid_env,
                            plots,
                            memory,
                            provider,
                            diverse_query,
                            experiment_path,
                            mode = "valid")
        valid_save_path = os.path.join(experiment_path, "valid_records.json")
        save_json(valid_records, valid_save_path)
    
def run(cfg, 
        env, 
        plots, 
        memory, 
        provider, 
        diverse_query,
        experiment_path,
        mode = "train"):
    
    # Grab or make trading records directory and memory records
    trading_records_path = os.path.join(experiment_path, "trading_records")
    os.makedirs(trading_records_path, exist_ok=True)
    memory_path = os.path.join(experiment_path, "memory")
    os.makedirs(memory_path, exist_ok=True)
    
    if cfg.if_load_trading_record and cfg.trading_record_path is not None:
        print("Loading trading records...")
        record_path = os.path.join(cfg.root, cfg.trading_record_path)
        trading_records = load_json(record_path)
    else:
        # Trading from scratch
        trading_records = {
            "symbol": [],
            "day": [],
            "value": [],
            "cash": [],
            "position": [],
            "ret": [],
            "date": [],
            "price": [],
            "discount": [],
            "kline_path": [],
            "trading_path": [],
            "total_profit": [],
            "total_return": [],
            "action": [],
            "reasoning": [],
        }
        
    state, info = env.reset()
    
    # Optional start from checkpoint (loops through each previous action taken until a new date is reached)
    if cfg.checkpoint_start_date is not None:
        for action, date in zip(trading_records["action"], trading_records["date"]):
            if date <= cfg.checkpoint_start_date:
                action = env.action_map[action]
                state, reward, done, truncated, info = env.step(action)
            else:
                break
    
    while True:
        action = run_step(cfg,
                          state,
                          info,
                          plots,
                          memory,
                          provider,
                          diverse_query,
                          experiment_path,
                          trading_records,
                          mode)
        
        assert action in env.action_map.keys(), f"Action {action} is not in the action map {env.action_map.keys()}"

        action = env.action_map[action]
        state, reward, done, truncated, info = env.step(action)
        
        if len(trading_records["action"]) > 0:
            if trading_records["action"][-1] != info["action"]:
                trading_records["action"][-1] = info["action"]
                
        # Save memories
        memory.save_local(memory_path=memory_path)
        
        if done:
            trading_records["total_profit"].append(info["total_profit"])
            trading_records["total_return"].append(info["total_return"])
            trading_records["date"].append(info["date"])
            trading_records["price"].append(info["price"])
            break
        
        # Save trading records
        save_json(trading_records, os.path.join(trading_records_path, f"trading_records_{str(info['date'])}.json"))

    return trading_records

def run_step(cfg,
             state,
             info,
             plots,
             memory,
             provider,
             diverse_query,
             experiment_path,
             trading_records,
             mode):
    
    # TODO
    # 1) issues with updating trading records during training
    
    params = dict()
    save_dir = "train" if mode == "train" else "valid"
    
    # plot kline chart
    kline_path = plots.plot_kline(state=state,
                                  info=info,
                                  save_dir=save_dir,
                                  mode=mode)
    params.update({
        "kline_path":kline_path
    })
    
    # Latest market intelligence
    lmi_summary_template_path = (cfg.train_latest_market_intelligence_summary_template_path 
                                 if mode == "train" 
                                 else cfg.valid_latest_market_intelligence_summary_template_path)
    cfg.latest_market_intelligence_summary["template_path"] = lmi_summary_template_path
    lmi_summary = PROMPT.build(cfg.latest_market_intelligence_summary)
    lmi_result = lmi_summary.run(state=state,
                                 info=info,
                                 params=params,
                                 memory=memory,
                                 provider=provider,
                                 diverse_query=diverse_query,
                                 exp_path=experiment_path,
                                 save_dir=save_dir,)
    
    # Past latest market intelligence
    prepared_latest_market_intelligence_params = prepare_latest_market_intelligence_params(state=state,
                                                                                           info=info,
                                                                                           params=params,
                                                                                           memory=memory,
                                                                                           provider=provider,
                                                                                           diverse_query=diverse_query)
    params.update(prepared_latest_market_intelligence_params)
    
    # Past market intelligence
    pmi_summary_template_path = (cfg.train_past_market_intelligence_summary_template_path 
                                 if mode == "train" 
                                 else cfg.valid_past_market_intelligence_summary_template_path)
    cfg.past_market_intelligence_summary["template_path"] = pmi_summary_template_path
    pmi_summary = PROMPT.build(cfg.past_market_intelligence_summary)
    pmi_result = pmi_summary.run(state=state,
                                 info=info,
                                 params=params,
                                 memory=memory,
                                 provider=provider,
                                 diverse_query=diverse_query,
                                 exp_path=experiment_path,
                                 save_dir=save_dir,)
    
    # Save latest market intelligence to memory
    lmi_summary.add_to_memory(state=state,
                              info=info,
                              result=lmi_result,
                              memory=memory,
                              provider=provider)
    
    # Low Level Reflection 
    llr_template_path = (cfg.train_low_level_reflection_template_path 
                                 if mode == "train" 
                                 else cfg.valid_low_level_reflection_template_path)
    cfg.low_level_reflection["template_path"] = llr_template_path
    low_level_reflection = PROMPT.build(cfg.low_level_reflection)
    low_level_reflection_result = low_level_reflection.run(state=state,
                                                           info=info,
                                                           params=params,
                                                           memory=memory,
                                                           provider=provider,
                                                           diverse_query=diverse_query,
                                                           exp_path=experiment_path,
                                                           save_dir=save_dir)
    
    # Prepare past low level reflection params
    prepared_low_level_reflection_params = prepare_low_level_reflection_params(state=state,
                                                                               info=info,
                                                                               params=params,
                                                                               memory=memory,
                                                                               provider=provider,
                                                                               diverse_query=diverse_query)
    params.update(prepared_low_level_reflection_params)
    
    # Store low level reflection in memory system
    low_level_reflection.add_to_memory(state=state,
                                       info=info,
                                       res=low_level_reflection_result,
                                       memory=memory,
                                       provider=provider)
    
    # Plot trading chart
    #if len(trading_records["date"]) <= 0:
    trading_path = None
    # else:
    #     trading_path = plots.plot_trading(records=trading_records,
    #                                       info=info,
    #                                       save_dir=save_dir)
    params.update({
        "trading_path": trading_path
    })
    
    # Grab trader preference
    if ASSET.check_trader(cfg.trader_preference):
        trader_preference = {
            "trader_preference": ASSET.get_trader(cfg.trader_preference)
        } 
    else:
        print("Trader preference in config is invalid, default will be used.")
        trader_preference = {
            "trader_preference": ASSET.get_trader("moderate_trader")
        }
    params.update(trader_preference)
    
    # Decision Making
    decision_template_path = (cfg.train_decision_template_path 
                              if mode == "train"
                              else cfg.valid_decision_template_path)
    cfg.decision["template_path"] = decision_template_path
    decision_prompt = PROMPT.build(cfg.decision)
    decision_result = decision_prompt.run(state=state,
                                            info=info,
                                            params=params,
                                            memory=memory,
                                            provider=provider,
                                            diverse_query=diverse_query,
                                            exp_path=experiment_path,
                                            save_dir=save_dir)

    # add records
    trading_records["symbol"].append(info["symbol"])
    trading_records["day"].append(info["day"])
    trading_records["value"].append(info["value"])
    trading_records["cash"].append(info["cash"])
    trading_records["position"].append(info["position"])
    trading_records["ret"].append(info["ret"])
    trading_records["date"].append(info["date"])
    trading_records["price"].append(info["price"])
    trading_records["discount"].append(info["discount"])
    trading_records["kline_path"].append(kline_path)
    trading_records["trading_path"].append(trading_path)
    trading_records["total_profit"].append(info["total_profit"])
    trading_records["total_return"].append(info["total_return"])
    trading_records["action"].append(decision_result["response_dict"]["action"])
    trading_records["reasoning"].append(decision_result["response_dict"]["reasoning"])
    
    return params["decision_action"]

if __name__ == "__main__":
    main()