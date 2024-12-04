import os
from datetime import datetime
from typing import Any
import numpy as np
import gym

class TradingEnvironment(gym.Env):
    # if the dataset is set to none and the mode = production, have it fetch real-time data for state
    # if dates are none, we could also have it use real time data
    
    def __init__(self,
                 mode: str = "train",
                 dataset: Any = None,
                 selected_asset: str = "BTCUSD",
                 asset_type: str = "cryptocurrency",
                 start_date: str = None,
                 end_date: str = None,
                 look_back_days: int = 14,
                 look_forward_days: int = 0,
                 initial_amount: float = 1e4,
                 transaction_cost_pct: float = 1e-3,
                 discount: float = 1.0,
                 ):
        '''
        Initializes the EnvironmentTrading instance.

        Parameters:
            mode (str, optional):
                Specifies the operational mode of the environment.
                - "train": For training the agent.
                - "test": For evaluating the agent's performance.
                - "production": For running with real-time data in production.
                Defaults to "train".
            
            dataset (Any, optional):
                The dataset containing all necessary data streams for the environment.
                This should include price data, and news articles from start to end date.
                Defaults to None.
            
            selected_asset (str, optional):
                The financial asset symbol that the environment will focus on.
                For example, "BTCUSD" for Bitcoin, etc.
                Defaults to "BTCUSD".
            
            asset_type (str, optional):
                Specifies the type/category of the selected asset.
                Supported types include:
                - "cryptocurrency": Represents digital assets like Bitcoin, Ethereum, etc.
                Defaults to "cryptocurrency".
            
            start_date (str, optional):
                The starting date for the trading simulation in the format "YYYY-MM-DD".
                Defines the beginning of the dataset slice used for the environment.
                If None, the earliest available date in the dataset is used.
                Defaults to None.
            
            end_date (str, optional):
                The ending date for the trading simulation in the format "YYYY-MM-DD".
                Defines the end of the dataset slice used for the environment.
                If None, the latest available date in the dataset is used.
                Defaults to None.
            
            look_back_days (int, optional):
                The number of past days' data to include in the current state representation.
                This historical window allows the agent to consider recent trends and patterns.
                Defaults to 14.
            
            look_forward_days (int, optional):
                The number of future days' data to include in the current state representation.
                Defaults to 0.
            
            initial_amount (float, optional):
                The initial capital (in monetary units) allocated to the agent for trading.
                This amount is used to calculate positions, cash, and overall portfolio value.
                Defaults to 10,000.0.
            
            transaction_cost_pct (float, optional):
                The percentage cost incurred for each transaction (buy or sell).
                This simulates real-world trading costs such as broker fees or slippage.
                For example, a value of 0.001 represents a 0.1% transaction cost.
                Defaults to 0.001.
            
            discount (float, optional):
                The discount factor applied to future rewards to calculate the total return.
                A value between 0 and 1, where values closer to 1 make future rewards more significant.
                Defaults to 1.0.

        Return:
            None
        '''

        # Trading Info Parameters
        self.mode = mode
        self.dataset = dataset
        self.selected_asset = selected_asset
        self.asset_type = asset_type
        self.symbol = selected_asset
        
        # Data Parameters
        self.prices = self.dataset.prices
        self.news = self.dataset.news
        self.prices.set_index("timestamp", inplace=True)
        self.news.set_index("timestamp", inplace=True)
        
        # Calendar Date Parameters
        self.start_date = start_date
        self.end_date = end_date
        self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        self.init_day = self.prices[self.prices["timestamp"] >= start_date].index.values[0]
        self.end_day = self.prices[self.prices["timestamp"] <= end_date].index.values[-1]
        
        # Forward and Backward Data Windows
        self.look_back_days = look_back_days
        self.look_forward_days = look_forward_days
        
        # Portfolio Parameters
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.discount = discount
        
        # Initial Trading Parameters
        self.dat = self.init_day
        self.value = self.initial_amount
        self.cash = self.initial_amount
        self.position = 0
        self.profit = 0
        self.date = self.get_current_date()
        self.price = self.get_current_price()
        self.total_return = 0
        self.total_profit = 0
        
        # Action mapping between string action and integer action
        self.action_map = {
            "SELL": -1,
            "HOLD": 0,
            "BUY": 1,
        }
        
    def get_current_date(self):
        '''
        Retrieves the current date based on the agent's position in the dataset.

        Parameters:
            None

        Returns:
            datetime:
                The timestamp corresponding to the current day in the trading simulation.
        '''
        return self.prices.index[self.day]
    
    def get_current_price(self):
        '''
        Obtains the adjusted closing price of the selected asset for the current day.

        Parameters:
            None

        Returns:
            float:
                The adjusted closing price of the asset on the current day.
        '''
        return self.prices.iloc[self.day]["close"]
    
    def get_current_value(self, price):
        '''
        Calculates the current total portfolio value based on cash and asset holdings.

        Parameters:
            price (float):
                The current price of the selected asset.

        Returns:
            float:
                The total value of the portfolio, combining available cash and the value of held positions.
        '''
        return self.cash + self.position * price
    
    def get_state(self):
        '''
        Constructs the current state representation for the agent.

        The state includes historical and, optionally, future data within specified look-back and look-forward windows.

        Parameters:
            None

        Returns:
            dict:
                A dictionary containing the following keys:
                    - "price" (pd.DataFrame):
                        Historical and future price data within the defined window.
                    - "news" (pd.DataFrame):
                        Historical and future news data within the defined window.
        '''
        state = {}

        days_ago = self.prices_df.index[self.day - self.look_back_days]
        days_future = self.prices_df.index[min(self.day + self.look_forward_days, len(self.prices_df) - 1)]

        price = self.prices_df[self.prices_df.index <= days_future]
        price = price[price.index >= days_ago]

        news = self.news_df[self.news_df.index <= days_future]
        news = news[news.index >= days_ago]

        state["price"] = price
        state["news"] = news
        
        return state
    
    def reset(self, **kwargs):
        self.day = self.init_day
        self.value = self.initial_amount
        self.cash = self.initial_amount
        self.position = 0
        self.ret = 0
        self.date = self.get_current_date()
        self.price = self.get_current_price()
        self.discount = 1.0
        self.total_return = 0
        self.total_profit = 0
        self.action = "HOLD"

        state = self.get_state()

        info = {
            "symbol": str(self.symbol),
            "asset_type": str(self.asset_type),
            "day": int(self.day),
            "value": float(self.value),
            "cash": float(self.cash),
            "position": int(self.position),
            "ret": float(self.ret),
            "date": self.date.strftime('%Y-%m-%d'),
            "price": float(self.price),
            "discount": float(self.discount),
            "total_profit": float(self.total_profit),
            "total_return": float(self.total_return),
            "action": self.action
        }

        return state, info
    
    def eval_buy_position(self, price):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(self.cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self):
        # evaluate sell position
        return int(self.position)
    
    def buy(self, price, amount=1):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount / self.action_radius)) * eval_buy_postion))
        if buy_position == 0:
            self.action = "HOLD"
        else:
            self.action = "BUY"

        self.cash -= buy_position * price * (1 + self.transaction_cost_pct)
        self.position += buy_position
        self.value = self.current_value(price)

    def sell(self, price, amount=-1):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position()

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount / self.action_radius)) * eval_sell_postion))
        if sell_position == 0:
            self.action = "HOLD"
        else:
            self.action = "SELL"

        self.cash += sell_position * price * (1 - self.transaction_cost_pct)
        self.position -= sell_position
        self.value = self.current_value(price)

    def hold_on(self, price, amount=0):
        self.action = "HOLD"
        self.value = self.current_value(price)
    
    def step(self, action: int = 0):

        pre_value = self.value

        if action > 0:
            self.buy(self.price, amount=action)
        elif action < 0:
            self.sell(self.price, amount=action)
        else:
            self.hold_on(self.price, amount=action)

        post_value = self.value

        reward = (post_value - pre_value) / pre_value

        self.day = self.day + 1

        if self.day < self.end_day:
            done = False
            truncted = False
        else:
            done = True
            truncted = True

        next_state = self.get_state()
        self.state = next_state

        self.value = post_value
        self.cash = self.cash
        self.position = self.position
        self.ret = reward
        self.date = self.get_current_date()

        self.price = self.get_current_price()
        self.total_return += self.discount * reward
        self.discount *= 0.99
        self.total_profit = 100 * (self.value - self.initial_amount) / self.initial_amount

        info = {
            "symbol": str(self.symbol),
            "asset_type": str(self.asset_type),
            "day": int(self.day),
            "value": float(self.value),
            "cash": float(self.cash),
            "position": int(self.position),
            "ret": float(self.ret),
            "date": self.date.strftime('%Y-%m-%d'),
            "price": float(self.price),
            "discount": float(self.discount),
            "total_profit": float(self.total_profit),
            "total_return": float(self.total_return),
            "action": str(self.action)
        }

        return next_state, reward, done, truncted, info
        