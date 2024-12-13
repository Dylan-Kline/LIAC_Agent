import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import ccxt

# Make sure ccxt_price_fetcher.py (with CCXTPriceFetcher class) is accessible
# If it's in the same directory, this should work:
from src.fetchers.prices import CCXTPriceFetcher

root = str(Path(__file__).resolve().parents[0])
sys.path.append(root)

def main():
    # Configuration
    exchange_name = "okx"     
    start_date = "2023-06-09"
    end_date = "2024-12-12"
    interval = "1d"
    workdir = "workdir"
    tag = "okx_data"
    cryptos_path = "configs/_asset_lists_/okx_cryptos.txt"

    # Initialize the price fetcher
    fetcher = CCXTPriceFetcher(
        root=root,
        exchange_name=exchange_name,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        cryptos_path=cryptos_path,
        workdir=workdir,
        tag=tag,
        limit=150000,
        delay=1.0
    )

    # Fetch the data for all cryptos (here just BTC/USDT)
    fetcher.fetch_all()

if __name__ == '__main__':
    main()