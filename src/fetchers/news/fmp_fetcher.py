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

load_dotenv(verbose=True)

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 200

class FMPCryptoNewsFetcher:
    def __init__(self,
                 root: str = "",
                 api_key: str = None,
                 delay: int = 1,
                 max_pages: int = None,
                 exchange_name: str = 'ok',
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 interval: str = "1d",
                 cryptos_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 shared_counter = None,
                 shared_lock = None,
                 shared_start_time = None,
                 **kwargs):
        self.root = root
        self.api_key = api_key if api_key is not None else os.environ.get("OA_FMP_KEY")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.cryptos_path = os.path.join(root, cryptos_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)

        self.max_pages = max_pages
        self.shared_counter = shared_counter
        self.shared_lock = shared_lock
        self.shared_start_time = shared_start_time

        os.makedirs(self.workdir, exist_ok=True)
        self.log_path = os.path.join(self.workdir, "log_{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.cryptos = self._init_cryptos()
        self.request_url = "https://financialmodelingprep.com/api/v4/crypto_news?symbol={}&page={}&from={}&to={}&apikey={}"
        
    def _init_cryptos(self):
        with open(self.cryptos_path) as op:
            cryptos = [line.strip() for line in op.readlines()]
        return cryptos
    
    def _get_jsonparsed_data(self, url):
        with self.shared_lock:
            current_time = datetime.now()
            elapsed = (current_time - self.shared_start_time.value).total_seconds()

            # If elapsed time has exceeded rate-limit timer, reset 
            if elapsed > ONE_MINUTE:
                self.shared_counter.value = 0
                self.shared_start_time.value = datetime.now()

            # If number of requests across processes is at the rate-limit, wait
            if self.shared_counter.value >= MAX_CALLS_PER_MINUTE:
                wait_time = ONE_MINUTE - elapsed
                print(f"Rate limit reached. Sleeping for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                self.shared_counter.value = 0
                self.shared_start_time.value = datetime.now()

            self.shared_counter.value += 1

        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        return json.loads(data)

    def check_status(self):
        failed_cryptos = self.cryptos
        return failed_cryptos
    
    def download(self,
                 cryptos = None,
                 start_date = None,
                 end_date = None,) -> None:
        
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if cryptos is None:
            cryptos = self.cryptos
            
        for crypto in cryptos:
            os.makedirs(os.path.join(self.workdir, crypto), exist_ok=True)
            crypto_news = pd.DataFrame()
            
            for page in tqdm(range(self.max_pages), 
                            bar_format="Download {} News:".format(crypto) + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
                
                # Check if page number has already been loaded and if so dont request it again
                page_path = os.path.join(self.workdir, crypto, "page{:06d}.csv".format(page))
                if os.path.exists(page_path):
                    chunk_news = pd.read_csv(page_path)
                else:
                    chunk_news = {
                        "timestamp": [],
                        "title": [],
                        "image": [],
                        "site": [],
                        "text": [],
                        "url": []
                    }
                    
                    # Format request url and contact API for news data
                    request_url = self.request_url.format(crypto, page, start_date, end_date, self.api_key)
                    try:
                        requested_data = self._get_jsonparsed_data(request_url)
                    except TimeoutError:
                        print("Time out.")
                        requested_data = []
                        
                    if len(requested_data) == 0:
                        with open(self.log_path, "a") as op:
                            op.write("{},{}\n".format(crypto, page))
                        continue
                    
                    for article in requested_data:
                        chunk_news["timestamp"].append(article["publishedDate"])
                        chunk_news["title"].append(article["title"])
                        chunk_news["image"].append(article["image"])
                        chunk_news["site"].append(article["site"])
                        chunk_news["text"].append(article["text"])
                        chunk_news["url"].append(article["url"])
                    
                    chunk_news = pd.DataFrame(chunk_news, index=range(len(chunk_news["timestamp"])))
                    chunk_news["timestamp"] = pd.to_datetime(chunk_news["timestamp"]).apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    chunk_news.to_csv(page_path, index=False)
                    
                crypto_news = pd.concat([crypto_news, chunk_news], axis=0)
            crypto_news.to_csv(os.path.join(self.workdir, "{}.csv".format(crypto)), index=False)
               
                    
                