import os 
import pandas as pd
from tqdm.auto import tqdm
import time
from urllib.request import urlopen
import certifi
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
from ratelimit import limits, sleep_and_retry
import ccxt
import numpy as np

class CCXTPriceFetcher:
    
    def __init__(
        self,
        root: str = "",
        exchange_name: str = "okx",
        api_key: str = None,
        start_date: str = "2023-04-01",
        end_date: str = "2023-04-01",
        interval: str = "1d",
        cryptos_path: str = None,
        workdir: str = "",
        tag: str = "",
        limit: int = None,
        delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize the price fetcher.
        
        Parameters:
            root (str): Root directory for relative paths.
            exchange_name (str): Name of the exchange (e.g., 'okx', 'coinbase').
            api_key (str): Optional API key (not always required for public OHLCV).
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            interval (str): Timeframe for OHLCV (e.g. '1d', '1h', '15m').
            cryptos_path (str): Path to a file containing a list of crypto symbols (one per line).
            workdir (str): Directory to store output data.
            tag (str): Additional tag for output directory.
            limit (int): Max candles per fetch_ohlcv call.
            delay (float): Delay between requests in seconds.
        """
        self.root = root
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.limit = limit
        self.delay = delay

        if cryptos_path is not None:
            self.cryptos_path = os.path.join(root, cryptos_path)
        else:
            self.cryptos_path = None

        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)
        os.makedirs(self.workdir, exist_ok=True)

        self.cryptos = self._init_cryptos()
        
        # Dynamically create exchange instance
        if not hasattr(ccxt, exchange_name):
            raise ValueError(f"Exchange {exchange_name} is not supported by ccxt.")
        exchange_class = getattr(ccxt, exchange_name)
        print(str(exchange_class))
        self.exchange = exchange_class({'enableRateLimit': True})
        
    def _init_cryptos(self):
        if self.cryptos_path is None or not os.path.exists(self.cryptos_path):
            raise FileNotFoundError("cryptos_path not provided or does not exist.")
        with open(self.cryptos_path) as op:
            cryptos = [line.strip() for line in op.readlines() if line.strip()]
        return cryptos
    
    def _parse_date(self, date_str: str):
        return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)

    def fetch_all(self):
        """
        Fetch OHLCV data for all cryptos in self.cryptos and save to CSV.
        """
        for crypto in self.cryptos:
            self.fetch_symbol(crypto)
            
    def fetch_symbol(self, symbol: str):
        """
        Fetch OHLCV data for a single symbol between start_date and end_date.
        """
        
        start_ts = self._parse_date(self.start_date)
        end_ts = self._parse_date(self.end_date)
        
        all_data = []

        # We will keep fetching data until we reach end_date or run out of data
        since = start_ts
        end = end_ts
        
        pbar = tqdm(desc=f"Fetching {symbol} OHLCV", unit="req")
        while True:
            pbar.update(1)
            data = self.exchange.fetchOHLCV(symbol,
                                             timeframe=self.interval,
                                             since=since,
                                             limit=self.limit)
            
            if not data:
                break
            
            data = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            data = data.sort_values("timestamp").reset_index(drop=True)
            data = data[data["timestamp"] <= end]
            
            if data.empty:
                break
            
            all_data.append(data)
            
            last_ts = data["timestamp"].iloc[-1]
            if last_ts >= end:
                break
            
            since = last_ts + 1
            time.sleep(self.delay)
        
        pbar.close()
        
        if not all_data:
            # No data fetched
            print(f"No data fetched for {symbol} in the given date range.")
            return []
        
        results = pd.concat(all_data, ignore_index=True)
        results["timestamp"] = pd.to_datetime(results["timestamp"], unit='ms').apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        results = results[["timestamp", "open", "high", "low", "close", "volume"]]
        
        save_path = os.path.join(self.workdir, "{}_{}.csv".format(symbol, self.interval))
        results.to_csv(save_path, index=False)
        print(f"Saved {symbol} data to {save_path}")