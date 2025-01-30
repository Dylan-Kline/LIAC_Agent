import os
from datetime import datetime
from typing import Any
import numpy as np
import gym

from src.registry import ENVIRONMENT
@ENVIRONMENT.register_module(force=True)
class TradingEnvironment(gym.Env):
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
        self.prices = self.dataset.prices[selected_asset]
        self.news = self.dataset.news[selected_asset]
        self.prices = self.prices.reset_index(drop=True)
        self.news = self.news.reset_index(drop=True)
        
        # Calendar Date Parameters
        self.start_date = start_date
        self.end_date = end_date
        self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        self.init_day = self.prices[self.prices["timestamp"] >= start_date].index.values[0]
        self.end_day = self.prices[self.prices["timestamp"] <= end_date].index.values[-1]
        
        self.prices.set_index("timestamp", inplace=True)
        self.news.set_index("timestamp", inplace=True)
        
        # Forward and Backward Data Windows
        self.look_back_days = look_back_days
        self.look_forward_days = look_forward_days
        
        # Portfolio Parameters
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.discount = discount
        
        # Initial Trading Parameters
        self.day = self.init_day
        self.value = self.initial_amount
        self.cash = self.initial_amount
        self.position = 0
        self.profit = 0
        self.date = self.get_current_date()
        self.price = self.get_current_price()
        self.total_return = 0
        self.total_profit = 0
        
        self.action_dim = 3
        self.action_radius = int(np.floor(self.action_dim/2))
        
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

        days_ago = self.prices.index[max(self.day - self.look_back_days, 0)]
        days_future = self.prices.index[min(self.day + self.look_forward_days, len(self.prices) - 1)]

        price = self.prices[self.prices.index <= days_future]
        price = price[price.index >= days_ago]
        news = self.news[self.news.index <= days_future]
        news = news[news.index >= days_ago]

        state["price"] = price
        state["news"] = news
        
        return state
    
    def reset(self, **kwargs):
        '''
        Resets the trading environment to its initial state for a new episode.

        Parameters:
            **kwargs:
                Arbitrary keyword arguments. Currently unused but can be utilized for future extensions.

        Returns:
            tuple:
                state (dict):
                    The initial state of the environment after reset, containing relevant data such as price and news within the defined window.
                info (dict):
                    A dictionary containing detailed information about the initial state, including:
                        - "symbol" (str): The symbol of the selected asset.
                        - "asset_type" (str): The type/category of the selected asset.
                        - "day" (int): The current day index in the simulation.
                        - "value" (float): The total portfolio value.
                        - "cash" (float): The available cash balance.
                        - "position" (int): The number of asset units currently held.
                        - "ret" (float): The return since the last action (initially 0).
                        - "date" (str): The current date in 'YYYY-MM-DD' format.
                        - "price" (float): The current price of the asset.
                        - "discount" (float): The current discount factor.
                        - "total_profit" (float): The total profit percentage relative to the initial amount.
                        - "total_return" (float): The cumulative discounted return.
                        - "action" (str): The last action taken ("HOLD" upon reset).
        '''
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
        '''
        Evaluates the maximum number of asset units that can be purchased given the current cash balance and transaction costs.

        Parameters:
            price (float):
                The current price of the selected asset.

        Returns:
            int:
                The maximum number of units that can be bought without exceeding the available cash, accounting for transaction costs.
                Calculated as the floor of (cash) divided by (price * (1 + transaction_cost_pct)).
        '''
        return int(np.floor(self.cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self):
        '''
        Evaluates the maximum number of asset units that can be sold based on the current position holdings.

        Parameters:
            None

        Returns:
            int:
                The total number of asset units currently held, representing the maximum sellable quantity.
        '''
        return int(self.position)
    
    def buy(self, price, amount=1):
        '''
        Executes a buy action, purchasing a calculated number of asset units based on the available cash and transaction costs.

        Parameters:
            price (float):
                The current price of the selected asset.
            amount (int, optional):
                A scaling factor determining the aggressiveness of the buy action.
                Defaults to 1.

        Returns:
            None

        Behavior:
            - Determines the evaluable buy position using `eval_buy_position`.
            - Calculates the actual number of units to buy based on the `amount` and `action_radius`.
            - Updates the cash balance by subtracting the total cost of the purchase, including transaction costs.
            - Increases the asset position by the number of units bought.
            - Updates the total portfolio value based on the new position.
            - Sets the last action to "BUY" if any units were purchased; otherwise, sets it to "HOLD".
        '''

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
        self.value = self.get_current_value(price)

    def sell(self, price, amount=-1):
        '''
        Executes a sell action, selling a calculated number of asset units based on the current holdings.

        Parameters:
            price (float):
                The current price of the selected asset.
            amount (int, optional):
                A scaling factor determining the aggressiveness of the sell action.
                Defaults to -1.

        Returns:
            None

        Behavior:
            - Determines the evaluable sell position using `eval_sell_position`.
            - Calculates the actual number of units to sell based on the `amount` and `action_radius`.
            - Updates the cash balance by adding the proceeds from the sale, minus transaction costs.
            - Decreases the asset position by the number of units sold.
            - Updates the total portfolio value based on the new position.
            - Sets the last action to "SELL" if any units were sold; otherwise, sets it to "HOLD".
        '''

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
        self.value = self.get_current_value(price)

    def hold_on(self, price, amount=0):
        '''
        Executes a hold action, maintaining the current asset position without making any transactions.

        Parameters:
            price (float):
                The current price of the selected asset.
            amount (int, optional):
                An unused parameter included for consistency with other action methods.
                Defaults to 0.

        Returns:
            None

        Behavior:
            - Sets the last action to "HOLD".
            - Updates the total portfolio value based on the current position and price.
        '''
        self.action = "HOLD"
        self.value = self.get_current_value(price)
    
    def step(self, action: int = 0):
        '''
        Advances the environment by one time step based on the agent's action, updating the state, calculating rewards, and determining episode termination.

        Parameters:
            action (int, optional):
                The action chosen by the reinforcement learning agent.
                - Positive integers (>0): Represent buy actions, with the magnitude indicating the aggressiveness.
                - Negative integers (<0): Represent sell actions, with the magnitude indicating the aggressiveness.
                - Zero (0): Represents a hold action.
                Defaults to 0.

        Returns:
            tuple:
                next_state (dict):
                    The updated state after executing the action, containing relevant data such as price and news within the defined window.
                reward (float):
                    The immediate reward obtained from taking the action, calculated as the percentage change in portfolio value.
                done (bool):
                    Indicates whether the episode has ended (True) or not (False).
                truncated (bool):
                    Indicates whether the episode was truncated (True) or not (False). Always mirrors the value of `done` in the current implementation.
                info (dict):
                    A dictionary containing detailed information about the current state, including:
                        - "symbol" (str): The symbol of the selected asset.
                        - "asset_type" (str): The type/category of the selected asset.
                        - "day" (int): The current day index in the simulation.
                        - "value" (float): The total portfolio value.
                        - "cash" (float): The available cash balance.
                        - "position" (int): The number of asset units currently held.
                        - "ret" (float): The return since the last action.
                        - "date" (str): The current date in 'YYYY-MM-DD' format.
                        - "price" (float): The current price of the asset.
                        - "discount" (float): The current discount factor.
                        - "total_profit" (float): The total profit percentage relative to the initial amount.
                        - "total_return" (float): The cumulative discounted return.
                        - "action" (str): The last action taken ("BUY", "SELL", or "HOLD").
        '''
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
        