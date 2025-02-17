import os

import pandas as pd

from src.registry import PLOTS
from src.plotting.charts import plot_kline, plot_trading
from src.utils.file_utils import init_path
import shutil

@PLOTS.register_module(force=True)
class PlotsInterface():
    def __init__(self,
                 root = None,
                 workdir = None,
                 tag = None,
                 suffix = 'png') -> None:
        super(PlotsInterface, self).__init__()
        self.root = root
        self.workdir = workdir
        self.tag = tag
        self.suffix = suffix

        self.exp_path = init_path(os.path.join(self.root, self.workdir, self.tag))
        self.plot_path = init_path(os.path.join(self.exp_path, "plots"))
        self.kline_plot_path = init_path(os.path.join(self.plot_path, "kline"))
        self.trading_plot_path = init_path(os.path.join(self.plot_path, "trading"))

    def plot_kline(self, state, info, save_dir, mode = "train"):

        try:
            price = state["price"]

            kline_dir = init_path(os.path.join(self.kline_plot_path, save_dir))

            price = price[["open", "high", "low", "close", "volume"]]
            price = price.reset_index(drop=False)
            price = price.dropna(axis=0, how="any")
            price = price.drop_duplicates(subset=["timestamp"], keep="first")
            price = price.set_index("timestamp")

            title = "{} kline of {}".format(info["date"], info["symbol"])
            kline_path = os.path.join(kline_dir, "kline_{}.{}".format(info["date"], self.suffix))

            now_date = pd.to_datetime(info["date"])
            now_date = min(price.index, key=lambda x: abs(x - now_date)) # find the nearest date before now_date
            now_date = now_date.strftime("%Y-%m-%d")

            plot_kline(price,
                       title,
                       kline_path,
                       now_date=now_date,
                       mode=mode)

        except Exception as e:
            print(e)
            kline_path = None
        return kline_path
    
    def plot_trading(self, records, info, save_dir):
        """
        Generates a trading plot using matplotlib and saves it to a file.
        
        Args:
            records (dict): A dictionary of trading records.
            info (dict): Dictionary with at least a 'date' key.
            save_dir (str): Subdirectory name within self.trading_plot_path to save the plot.
        
        Returns:
            str or None: The full file path to the saved plot, or None if an error occurred.
        """
        try:
            # Create (or get) the trading directory.
            trading_dir = init_path(os.path.join(self.trading_plot_path, save_dir))

            # Construct the filename, e.g. "trading_2020-01-01.png"
            trading_path = os.path.join(trading_dir, "trading_{}.{}".format(info['date'], self.suffix))

            # Plot trading chart
            plot_trading(records, trading_path, now_date=info.get('date'))
            
        except Exception as e:
            print(f"Error in plot_trading: {e}")
            trading_path = None

        return trading_path