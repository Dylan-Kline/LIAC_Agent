# LIAC_Agent

## Getting Started

1. Clone the git repo to a local folder.

2. Download the required libraries and dependencies to run the code by navigating to the cloned folder and opening a terminal in your IDE such as VSCode.
    - Run the following command once you have the terminal open and you have navigated to the project folder with the terminal command lind command 
    
    ```
    cd your_file_path/LIAC_agent
    ```
    should look similar to the above but not exact.
    
    - After you navigate to the folder run the following command to download the dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Create a .env file in the root project folder LIAC_AGENT with the following information:

```
OA_OPENAI_KEY = "your api key here" # https://platform.openai.com/docs/overview
OA_FMP_KEY = "ask for the api key for this part since its not safe to store on github" # https://site.financialmodelingprep.com/developer/docs
```

4. Change the config file located in *LIAC_AGENT/configs/_asset_lists_/whatever_exchange_you_want_cryptos.txt* by putting the crypto symbol pair such as BTC-USDT.

5. Download the datasets for prices and news. You can see examples of how to do so in the *testing_news_download* and *testing_prices_download* files.

6. Update the dataset paths in the config file found in *LIAC_AGENT/configs/experiment_cfgs/trading_w_mi_low/BTC-USDT.py*. Look for the dataset config that looks like the following:

```
dataset = dict(
    type="Dataset",
    root=root,
    price_path="datasets/exp_cryptos/price", # Update this to the path of the price data you downloaded.
    news_path="datasets/exp_cryptos/news", # Update this to the path of the news data you downloaded.
    interval="1d",
    assets_path="configs/_asset_lists_/exp_cryptos.txt",
    workdir=workdir,
    tag=tag
)
```

## Running the code

1. To run the agent in training mode use the following command in the terminal:

```
python training/train-w-mi-w-low.py
```

2. To run the agent in validation mode aka backtesting (still has some changes that need to be made), use the following:

```
python training/train-w-mi-w-low.py --no_train --if_valid
```

## Summary of the overall architecture 

### Config Files 
These are the main way to set up the program to run and train on specific dates.

#### Running experiments

**Make sure to keep the workdir as "workdir/"** 
Changing this value would cause issues with how the current structure is set up.

You can also run seperate experiments through using the existing *BTC-USDT.py* config file or creating a new config file in the *configs/experiment_cfgs/trading_w_mi_low* folder by copy pasting the *BTC-USDT.py* file and updating the information that you wish to change such as dates to train on, trader preferences, etc. Keep in mind that you should change the *tag* in the config file to seperate your experiments into different folders.

### Workdir aka experiments folder

Within the *workdir* folder you should see your experiments that you have run. Within each you can view the trading records to view how to the model has performed. Though, within the train_records.json file you can see the full summary of the trading history of the agent throughout the trading period you defined.

### Data Analysis of Trading Records

If you wish to see what the sharpe ratio, annualized return rate, etc. for the training or trading records, you can use the *metric* functions found in the *LIAC_AGENT/src/metrics/metrics.py* file. Read the json file using the json library, perhaps convert it to a pandas dataframe and then play around with the metrics functions to see key information about the trading records.

