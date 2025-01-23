import argparse
import os
import sys
from pathlib import Path

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
from src.registery import *

from mmengine.config import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description="Train with mi and low.")
    parser.add_argument("--config", 
                        default=os.path.join(ROOT, "configs", "experiment_cfgs", "trading_w_mi_low", "BTCUSD.py"),
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
    parser.add_argument("--if_load_memory", action="store_true", default=True)
    parser.add_argument("--memory_path", type=str, default=None)
    parser.add_argument("--if_load_trading_record", action="store_true", default=True)
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

    return cfg


def main():

    cfg = setup_environment()

    # Initialize environments, memory, provider, and querys
    provider = PROVIDER.build(cfg.provider)
    
    


if __name__ == "__main__":
    main()