import os
import sys
from pathlib import Path
import argparse
import multiprocessing
from src.fetchers.news import FMPCryptoNewsFetcher

root = str(Path(__file__).resolve().parents[0])
print(root)
sys.path.append(root)

class CryptoDownloaderProcess(multiprocessing.Process):
    def __init__(self, cryptos, downloader):
        super().__init__()
        self.cryptos = cryptos
        self.downloader = downloader
    def run(self):
        self.downloader.download(self.cryptos)
        
def main():
    workdir = "workdir"
    tag = "fmp_news_exp_cryptos"
    batch_size = 1

    type = "FMPCryptoNewsDownloader"
    token = None
    start_date = "2023-06-09"
    end_date = "2025-01-03"
    interval = "1d"
    delay = 1
    cryptos_path = "configs/_asset_lists_/cryptos.txt"
    
    downloader = FMPCryptoNewsFetcher(
        root = root,
        api_key=None,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        max_pages=3891,
        cryptos_path=cryptos_path,
        workdir=workdir,
        tag=tag,
    )

    print(f"| Check Downloading {tag}...")
    cryptos = downloader.check_status()
    print(f"| Check Downloading {tag} Done! Failed: {len(cryptos)} / {len(downloader.cryptos)}")

    batch_size = 1
    batch_size = min(len(cryptos), batch_size)

    processes = []
    remaining_cryptos = downloader.check_status()

    while remaining_cryptos:
        batch = remaining_cryptos[:batch_size]
        remaining_cryptos = remaining_cryptos[batch_size:]

        process = CryptoDownloaderProcess(batch, downloader)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()