import os
import pathlib
import sys
import pandas as pd

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100)

class Dataset:
    def __init__(self,
                 root: str = None,
                 price_path: str = None,
                 news_path: str = None,
                 assets_path: str = None,
                 interval: str = "1d",
                 workdir: str = None,
                 tag: str = None,
                 ):
        self.root = root
        self.price_path = os.path.join(root, price_path)
        self.news_path = os.path.join(root, news_path)
        self.assets_path = os.path.join(root, assets_path)
        self.interval = interval
        self.workdir = workdir
        self.tag = tag

        self.exp_path = os.path.join(self.root, self.workdir, self.tag)
        os.makedirs(self.exp_path, exist_ok=True)

        self.assets = self._init_assets()
        self.prices = self._load_prices()
        self.news = self._load_news()
        
    def _init_assets(self):
        with open(self.assets_path) as op:
            assets = [line.strip() for line in op.readlines()]
        return assets

    def _load_prices(self):

        prices = {}

        for asset in self.assets:
            path = os.path.join(self.price_path, "{}_{}.csv".format(asset, self.interval))
            df = pd.read_csv(path)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = df.sort_values(by="timestamp")
            df = df.reset_index(drop=True)

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            prices[asset] = df

        return prices

    def _load_news(self):

        news = {}

        global_id = 0

        for asset in self.assets:
            
            news_path = os.path.join(self.news_path, "{}.csv".format(asset))
            if os.path.exists(news_path):
                path = news_path
            else:
                print(f"Path does not exist for {asset}, {news_path}")
                continue
            
            df = pd.read_csv(path)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = df.dropna(axis=0, how="any")
            df = df.sort_values(by="timestamp")
            df = df.reset_index(drop=True)

            df["id"] = df.index + global_id
            df["id"] = df["id"].apply(lambda x: "{:06d}".format(x))
            global_id += len(df)

            df = df[["timestamp", "id", "title", "text"]]

            news[asset] = df

        return news

if __name__ == '__main__':

    dataset = Dataset(
        root = ROOT,
        price_path = "workdir/okx_data/",
        news_path = "workdir/fmp_news_exp_cryptos/",
        interval = "1d",
        assets_path = "configs/_asset_lists_/okx_cryptos.txt",
        workdir = os.path.join(ROOT, "workdir"),
        tag = "exp"
    )

    selected_asset = "BTC-USDT"

    print(len(dataset.prices[selected_asset]))
    print(len(dataset.news[selected_asset]))
    print(dataset.prices[selected_asset])
    print(dataset.news[selected_asset])