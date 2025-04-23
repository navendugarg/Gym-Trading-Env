import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import datetime
import glob
from pathlib import Path
from collections import Counter, deque # Use deque for History if performance is critical, but dict is fine for now
from .utils.history import History # Assuming History handles storage efficiently
from .utils.portfolio import Portfolio, TargetPortfolio
import tempfile
import os
import warnings

warnings.filterwarnings("error") # Consider making this configurable or removing for library use

# Default reward and feature functions
def basic_reward_function(history: History) -> float:
    """
    Default reward function: Calculates the log return of the portfolio valuation
    between the last two steps.

    Args:
        history: History object containing past portfolio valuations.

    Returns:
        Log return reward. Returns 0 if prior valuation is zero or not available.
    """
    if len(history) < 2 or history["portfolio_valuation", -2] == 0:
        return 0.0
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history: History) -> float:
    """
    Default dynamic feature: Returns the last position taken by the agent.
    Note: This is the *target* position requested, not necessarily the *real* position
    after considering leverage and interest.

    Args:
        history: History object containing past positions.

    Returns:
        The last position value. Returns 0 if history is empty.
    """
    if not history: return 0.0
    return history['position', -1]

def dynamic_feature_real_position(history: History) -> float:
    """
    Default dynamic feature: Returns the real position ratio of the portfolio
    (net asset exposure / total value) from the last step.

    Args:
        history: History object containing past real positions.

    Returns:
        The last real position ratio. Returns 0 if history is empty.
    """
    if not history: return 0.0
    return history['real_position', -1]


class TradingEnv(gym.Env):
    """
    A Gymnasium environment for simulating stock/crypto trading strategies.

    It allows agents to learn trading policies based on historical market data.
    The environment tracks portfolio value, handles trades with fees and optional
    borrowing interest, and provides observations based on market features and
    dynamic portfolio states.

    Action Space: Discrete space corresponding to the index in the `positions` list.
                  Example: If `positions = [0, 0.5, 1]`, action 0 means target 0% asset,
                           action 1 means target 50% asset, action 2 means target 100% asset.

    Observation Space: Box space containing market features and dynamic features.
                       Shape depends on the `windows` parameter:
                       - If `windows` is None: `(num_features,)` - current step's features.
                       - If `windows` is int: `(windows, num_features)` - features of the last `windows` steps.

    Attributes:
        df (pd.DataFrame): The market data used by the environment.
        positions (list): List of allowed target portfolio positions (e.g., [0, 1] for long-only, [-1, 0, 1] for long/short).
        dynamic_feature_functions (list): List of functions that compute dynamic features based on the trading history.
        reward_function (callable): Function to calculate the reward at each step.
        windows (int | None): Number of past steps included in the observation. If None, only the current step is observed.
        trading_fees (float): Trading fee rate applied to the value of each trade.
        borrow_interest_rate (float): Interest rate applied per step to borrowed assets or fiat.
        portfolio_initial_value (float): The starting value of the portfolio in fiat currency.
        initial_position (str | float): The initial position of the portfolio. Can be 'random' or a value from `positions`.
        max_episode_duration (str | int): Maximum number of steps per episode. 'max' uses the full dataset length.
        verbose (int): Logging level (0: None, 1: Episode metrics).
        name (str): Name of the environment instance (e.g., "StockTrading").
        render_mode (str): Rendering mode ('logs' supported).
        historical_info (History): Object storing step-by-step data of the episode.
        portfolio (Portfolio): The portfolio object managing assets and fiat.
    """
    metadata = {'render_modes': ['logs']}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list = [0, 1], # Example: [0 = short, 1 = long] or [0, 1] for long-only
        dynamic_feature_functions: list = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
        reward_function: callable = basic_reward_function,
        windows: int = None,
        trading_fees: float = 0.0, # e.g. 0.01 for 1% - applied to trade value
        borrow_interest_rate: float = 0.0, # e.g. 0.0001 for 0.01% per step
        portfolio_initial_value: float = 1000.0,
        initial_position: str | float = 'random', # 'random' or a specific value from `positions`
        max_episode_duration: str | int = 'max', # 'max' or integer number of steps
        verbose: int = 1, # 0: No logs, 1: Episode summary logs
        name: str = "TradingEnv",
        render_mode: str = "logs" # Only 'logs' is implemented for now
    ):
        """Initializes the Trading Environment."""
        super().__init__() # Initialize Gymnasium Env

        self.name = name
        self.verbose = verbose

        # --- Validation ---
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(df.index, pd.DatetimeIndex), "df must have a DatetimeIndex"
        required_cols = {"open", "high", "low", "close"}
        assert required_cols.issubset(df.columns), f"df must contain columns: {required_cols}"
        assert isinstance(positions, list) and len(positions) > 1, "positions must be a list with at least two elements"
        assert callable(reward_function), "reward_function must be callable"
        assert isinstance(dynamic_feature_functions, list) and all(callable(f) for f in dynamic_feature_functions), \
            "dynamic_feature_functions must be a list of callable functions"
        assert windows is None or (isinstance(windows, int) and windows > 0), "windows must be None or a positive integer"
        assert isinstance(trading_fees, (int, float)) and 0 <= trading_fees < 1, "trading_fees must be between 0 and 1"
        assert isinstance(borrow_interest_rate, (int, float)) and borrow_interest_rate >= 0, "borrow_interest_rate must be non-negative"
        assert isinstance(portfolio_initial_value, (int, float)) and portfolio_initial_value > 0, "portfolio_initial_value must be positive"
        assert initial_position == 'random' or initial_position in positions, "initial_position must be 'random' or one of the values in 'positions'"
        assert max_episode_duration == 'max' or (isinstance(max_episode_duration, int) and max_episode_duration > 0), \
            "max_episode_duration must be 'max' or a positive integer"
        assert render_mode in self.metadata["render_modes"], f"render_mode must be one of {self.metadata['render_modes']}"

        # --- Assign parameters ---
        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = float(trading_fees)
        self.borrow_interest_rate = float(borrow_interest_rate)
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode

        # --- Process DataFrame ---
        self._set_df(df) # Initializes self.df, self._obs_array, etc.

        # --- Define Spaces ---
        # Action: Choose an index from the `positions` list
        self.action_space = spaces.Discrete(len(positions))

        # Observation: Market features + dynamic features
        # Shape depends on whether a window is used
        obs_shape = [self._nb_features] if windows is None else [windows, self._nb_features]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # --- State variables ---
        self._portfolio: Portfolio = None
        self._idx: int = 0 # Current index in the DataFrame
        self._step: int = 0 # Current step number in the episode
        self._position: float = 0.0 # Current target position
        self._limit_orders: dict = {} # Stores active limit orders {position: {limit: price, persistent: bool}}
        self.historical_info: History = None # Stores episode history
        self.results_metrics: dict = {} # Stores metrics calculated at the end of episode
        self.log_metrics: list = [] # List of custom metrics to log {'name': str, 'function': callable}


    def _set_df(self, df: pd.DataFrame) -> None:
        """
        Internal method to process the input DataFrame, identify feature columns,
        and prepare data arrays for efficient access.

        Args:
            df: The market data DataFrame.
        """
        df = df.copy() # Work on a copy to avoid modifying the original DataFrame

        # Identify feature columns (columns containing 'feature')
        self._features_columns = [col for col in df.columns if "feature" in col.lower()]
        if not self._features_columns:
            warnings.warn("No columns containing 'feature' found. Using OHLVC as features.", UserWarning)
            # Default to OHLCV if no 'feature' columns are present
            default_feature_cols = ['open', 'high', 'low', 'close', 'volume']
            self._features_columns = [col for col in default_feature_cols if col in df.columns]

        # Identify info columns (all columns except features, ensuring 'close' is included)
        self._info_columns = list(set(df.columns).union({"close"}) - set(self._features_columns))

        self._nb_static_features = len(self._features_columns) # Number of features from the original DataFrame

        # Add placeholder columns for dynamic features
        self._dynamic_feature_col_indices = {}
        for i, func in enumerate(self.dynamic_feature_functions):
            col_name = f"dynamic_feature__{func.__name__}" # Use function name for clarity
            df[col_name] = 0.0 # Initialize with zeros
            self._features_columns.append(col_name)
            self._dynamic_feature_col_indices[col_name] = self._nb_static_features + i

        self._nb_features = len(self._features_columns) # Total number of features (static + dynamic)

        # Store data as numpy arrays for faster access
        self.df = df # Keep the processed DataFrame
        # Ensure all feature columns are numeric before converting to numpy
        for col in self._features_columns:
             if not pd.api.types.is_numeric_dtype(self.df[col]):
                 raise TypeError(f"Feature column '{col}' is not numeric. Please ensure all feature columns are numeric.")
        self._obs_array = np.array(self.df[self._features_columns], dtype=np.float32)

        # Ensure all info columns are numeric before converting to numpy
        info_arrays = []
        for col in self._info_columns:
            # Skip conversion for known non-numeric columns
            if col in ['symbol', 'unix']: # Add other known non-numeric columns if necessary
                 info_arrays.append(df[col].values.reshape(-1,1)) # Keep as object or original type
                 continue
            try:
                # Attempt conversion, coercing errors to NaN
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                
                # Check if NaNs were introduced by coercion
                if numeric_col.isnull().any() and not df[col].isnull().any():
                     warnings.warn(f"Info column '{col}' contained non-numeric values that were converted to NaN.", UserWarning)
                
                info_arrays.append(numeric_col.values.reshape(-1,1))

            except Exception as e:
                 # This catch block might be less necessary now with explicit skipping,
                 # but kept for safety against unexpected column types.
                 # We raise a more informative error if conversion fails unexpectedly.
                 raise TypeError(f"Failed to process info column '{col}'. It might contain mixed types or unsupported data. Original error: {e}")

        if info_arrays:
            self._info_array = np.concatenate(info_arrays, axis=1)
        else:
            self._info_array = np.zeros((len(df), 0)) # Empty array if no info columns

        # Store close prices separately for quick access
        self._price_array = np.array(self.df["close"], dtype=np.float32)

        # Validate window size
        if self.windows is not None and self.windows > len(self.df):
            raise ValueError(f"Window size ({self.windows}) cannot be larger than the DataFrame length ({len(self.df)})")


    def _get_ticker(self, delta: int = 0) -> pd.Series:
        """
        Retrieves market data for the current step (+ delta) as a pandas Series.

        Args:
            delta: Offset from the current index `_idx`. Defaults to 0.

        Returns:
            A pandas Series containing the market data for the specified step.
        """
        current_idx = self._idx + delta
        if 0 <= current_idx < len(self.df):
            return self.df.iloc[current_idx]
        else:
            # Handle edge case, maybe return last known ticker or raise error
            # Returning last valid ticker might be problematic, raise error is safer
            raise IndexError(f"Attempted to access ticker at index {current_idx}, out of bounds [0, {len(self.df)-1}]")

    def _get_price(self, delta: int = 0) -> float:
        """
        Retrieves the closing price for the current step (+ delta).

        Args:
            delta: Offset from the current index `_idx`. Defaults to 0.

        Returns:
            The closing price at the specified step.
        """
        current_idx = self._idx + delta
        if 0 <= current_idx < len(self._price_array):
            return self._price_array[current_idx]
        else:
             raise IndexError(f"Attempted to access price at index {current_idx}, out of bounds [0, {len(self._price_array)-1}]")


    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation for the current step.
        It calculates dynamic features and returns the observation array,
        potentially including a window of past steps.

        Returns:
            The observation array (np.ndarray).
        """
        # Update dynamic features for the current index _idx
        for i, func in enumerate(self.dynamic_feature_functions):
            # Find the correct column index for this dynamic feature
            # Assuming dynamic features are appended in order
            dynamic_feature_index = self._nb_static_features + i
            # Calculate feature value using history up to the *previous* step for causality
            # Note: If features depend on current price/portfolio, they should be calculated *after* the step logic
            # The current implementation calculates based on history *before* the current step's action/update.
            self._obs_array[self._idx, dynamic_feature_index] = func(self.historical_info)

        # Return observation based on window setting
        if self.windows is None:
            # Return only the current step's observation vector
            return self._obs_array[self._idx]
        else:
            # Return a window of observations ending at the current step
            # Ensure we don't go before the start of the array
            start_idx = max(0, self._idx + 1 - self.windows)
            obs_window = self._obs_array[start_idx : self._idx + 1]
            # Pad if needed at the beginning of the episode
            if len(obs_window) < self.windows:
                padding = np.zeros((self.windows - len(obs_window), self._nb_features), dtype=np.float32)
                obs_window = np.vstack((padding, obs_window))
            return obs_window


    def reset(self, seed=None, options=None, **kwargs):
        """
        Resets the environment to the initial state for a new episode.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the initial observation and initial info dictionary.
        """
        super().reset(seed=seed) # Set seed for RNG

        # Reset episode state variables
        self._step = 0
        self._limit_orders = {} # Clear any pending limit orders

        # Determine starting index
        # If windows=N, the first valid observation is at index N-1
        # The agent starts making decisions from index N-1
        self._idx = 0 if self.windows is None else self.windows - 1

        # If max_episode_duration is set, randomly choose a start point
        # ensuring the episode can run for the maximum duration.
        if isinstance(self.max_episode_duration, int):
            max_start_idx = len(self.df) - self.max_episode_duration
            # Ensure the random start index is not before the minimum required index
            start_idx_min = 0 if self.windows is None else self.windows - 1
            if max_start_idx > start_idx_min:
                 self._idx = self.np_random.integers(start_idx_min, max_start_idx + 1)
            else:
                # If max_duration is too long for the data, start at the earliest possible point
                self._idx = start_idx_min
                warnings.warn(f"max_episode_duration ({self.max_episode_duration}) is too long for the available data length ({len(self.df)}) after considering window size. Starting at index {self._idx}.", UserWarning)


        # Determine initial position
        if self.initial_position == 'random':
            self._position = self.np_random.choice(self.positions)
        else:
            self._position = self.initial_position

        # Initialize portfolio
        initial_price = self._get_price() # Get price at the starting index _idx
        self._portfolio = TargetPortfolio(
            position=self._position,
            value=self.portfolio_initial_value,
            price=initial_price
        )

        # Initialize history logger
        # Max size can be estimated or set to df length + buffer
        self.historical_info = History(max_size=len(self.df))
        # Log the initial state (step 0)
        initial_portfolio_distribution = self._portfolio.get_portfolio_distribution()
        initial_valuation = self._portfolio.valuation(initial_price)
        initial_real_position = self._portfolio.real_position(initial_price)

        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx], # Use pre-computed index values
            position_index=self.positions.index(self._position), # Store index of the position
            position=self._position, # Store the actual position value
            real_position=initial_real_position, # Store initial real position
            # Store initial market data info
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=initial_valuation,
            portfolio_distribution=initial_portfolio_distribution,
            reward=0.0 # No reward at the first step
        )

        # Get initial observation (dynamic features will use the just-logged history)
        initial_obs = self._get_obs()
        # Get initial info dictionary (usually the last entry in history)
        initial_info = self.historical_info[0] # Get the dict for the first step

        return initial_obs, initial_info


    def render(self):
        """
        Renders the environment. Currently only supports 'logs' mode,
        which prints metrics at the end of the episode via the `log` method.
        """
        if self.render_mode == "logs":
            # Logging is handled by the `log` method, called when done/truncated
            pass
        else:
             # In Gymnasium, render() is expected to return something for modes like 'rgb_array'
             # For 'human' mode, it usually updates a visualization.
             # Since only 'logs' is supported, we do nothing here.
             return None


    def _trade(self, position: float, price: float = None) -> None:
        """
        Executes a trade to adjust the portfolio to the target `position`.

        Args:
            position: The target position ratio (e.g., 0, 0.5, 1).
            price: The price at which to execute the trade. If None, uses the
                   current step's closing price. Defaults to None.
        """
        trade_price = self._get_price() if price is None else price
        self._portfolio.trade_to_position(
            position=position,
            price=trade_price,
            trading_fees=self.trading_fees
        )
        # Update the internal state tracking the *target* position
        self._position = position


    def _take_action(self, position_index: int) -> None:
        """
        Processes the agent's chosen action (position index) and executes
        the trade if the target position is different from the current one.

        Args:
            position_index: The index corresponding to the desired position
                            in the `self.positions` list.
        """
        new_position = self.positions[position_index]
        if new_position != self._position: # Only trade if the target position changes
            self._trade(new_position)


    def _take_action_order_limit(self) -> None:
        """
        Checks if any active limit orders should be executed based on the
        current step's high and low prices. Executes the trade if conditions are met.
        """
        if not self._limit_orders: # Skip if no limit orders are active
            return

        # Get current market data (open, high, low, close)
        # Using _get_ticker is less efficient than direct array access if only H/L needed
        # Let's assume _info_array contains high and low if needed frequently
        # Or get the specific ticker data
        ticker = self._get_ticker() # Contains 'high' and 'low'
        current_high = ticker["high"]
        current_low = ticker["low"]

        # Iterate through a copy of items in case orders are deleted during iteration
        for position, params in list(self._limit_orders.items()):
            limit_price = params['limit']
            # Check if the limit price falls within the day's range (high-low)
            # And ensure the target position is different from the current one
            if (position != self._position and
                limit_price <= current_high and
                limit_price >= current_low):

                # Execute the trade at the limit price
                self._trade(position, price=limit_price)

                # Remove the order if it's not persistent
                if not params['persistent']:
                    del self._limit_orders[position]

                # Important: If multiple limit orders could trigger, this executes only the first one found.
                # If an order executes, the self._position changes, which might affect subsequent checks
                # in the same step if not breaking here. Consider if multiple fills per step are allowed/desired.
                # For simplicity, let's assume only one limit order can fill per step.
                break


    def add_limit_order(self, position: float, limit: float, persistent: bool = False) -> None:
        """
        Adds a limit order to the environment.

        Args:
            position: The target position to take if the limit price is reached.
                      Must be a value from the `self.positions` list.
            limit: The price at which the order should trigger.
            persistent: If True, the order remains active even after triggering.
                        If False, it's removed after triggering once. Defaults to False.
        """
        if position not in self.positions:
            warnings.warn(f"Limit order position {position} is not in the allowed positions {self.positions}. Order ignored.", UserWarning)
            return
        self._limit_orders[position] = {'limit': limit, 'persistent': persistent}


    def step(self, position_index: int = None):
        """
        Advances the environment by one time step.

        Processes the agent's action (if provided), updates the portfolio based on
        market movement and interest, calculates the reward, and determines if the
        episode has ended.

        Args:
            position_index (int, optional): The index of the action taken by the agent,
                                            corresponding to an element in `self.positions`.
                                            If None, no new action is taken (e.g., holding).

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The observation for the next state.
                - reward (float): The reward obtained in this step.
                - terminated (bool): True if the episode ended due to a terminal condition
                                     (e.g., portfolio value <= 0), False otherwise.
                - truncated (bool): True if the episode ended due to reaching the time limit
                                    or end of data, False otherwise.
                - info (dict): A dictionary containing auxiliary information about the step.
        """
        # --- 1. Take Action (based on agent's decision) ---
        if position_index is not None:
            # Ensure action is valid
            if not (0 <= position_index < len(self.positions)):
                 raise ValueError(f"Invalid position_index {position_index}. Must be between 0 and {len(self.positions)-1}.")
            # Pass the integer index directly to _take_action
            self._take_action(position_index)
        # If position_index is None, the agent decided to hold the current position.

        # --- 2. Advance Time ---
        self._idx += 1
        self._step += 1

        # Check if end of data is reached before proceeding
        if self._idx >= len(self.df):
            truncated = True
            terminated = False # Not terminated by condition, just end of data
            # Need to return something sensible. Calculate final state before exiting.
            # Use the *last valid* price for final calculations.
            last_valid_price = self._get_price(-1) # Price from the previous index
            final_valuation = self._portfolio.valuation(last_valid_price)
            final_real_position = self._portfolio.real_position(last_valid_price)
            # Update history with the last valid state before truncation
            self.historical_info.add(
                idx=self._idx -1, # Log against the last valid index
                step=self._step,
                date=self.df.index.values[self._idx - 1],
                position_index=position_index if position_index is not None else self.positions.index(self._position),
                position=self._position,
                real_position=final_real_position,
                data=dict(zip(self._info_columns, self._info_array[self._idx - 1])),
                portfolio_valuation=final_valuation,
                portfolio_distribution=self._portfolio.get_portfolio_distribution(),
                reward=0 # No reward for the step that goes out of bounds
            )
            # Calculate final metrics and log
            self.calculate_metrics()
            self.log()
            # Return the *last valid* observation and info
            # Note: _get_obs() uses _idx, so call it before modifying _idx or use _idx-1
            # However, the standard is to return the obs for the *next* state.
            # In truncation, there is no next state, so returning the last obs is common.
            # We need to be careful here. Let's return the observation corresponding to the state *after* the last action.
            # This requires calculating dynamic features based on the state *before* truncation.
            last_obs = self._get_obs() # This will use the incremented _idx, potentially causing issues if called after truncation check.
                                      # Let's recalculate obs based on _idx-1 if truncated here.

            # Recalculate observation for the state *at* _idx-1
            # Temporarily set _idx back to calculate the correct final observation state
            self._idx -= 1
            final_obs = self._get_obs()
            self._idx += 1 # Restore _idx

            return final_obs, 0.0, terminated, truncated, self.historical_info[-1]


        # --- 3. Process Limit Orders (if any) ---
        # Check if any limit orders were hit by the H/L prices of the *new* current step (_idx)
        self._take_action_order_limit()

        # --- 4. Update Portfolio ---
        # Get the price for the current step
        current_price = self._get_price()
        # Apply borrowing interest based on positions held *during* the previous step
        # Interest should apply *before* calculating the new valuation
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)
        # Calculate the new portfolio valuation based on the current price
        portfolio_valuation = self._portfolio.valuation(current_price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()
        real_position = self._portfolio.real_position(current_price)

        # --- 5. Determine Episode End Conditions ---
        # Terminated: Portfolio value reached zero or below (bankrupt)
        terminated = portfolio_valuation <= 0

        # Truncated: Reached end of data (handled above) or max episode duration
        truncated = False # Reset truncated flag
        if isinstance(self.max_episode_duration, int) and self._step >= self.max_episode_duration:
            truncated = True
        # Check again for end of data index (redundant if handled above, but safe)
        if self._idx >= len(self.df) - 1:
             truncated = True


        # --- 6. Log History for Current Step ---
        # Store data *before* calculating reward for this step
        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=position_index if position_index is not None else self.positions.index(self._position), # Log the action taken *leading* to this state
            position=self._position, # Log the target position *after* action/limit orders
            real_position=real_position, # Log the real position *at* the end of the step
            data=dict(zip(self._info_columns, self._info_array[self._idx])), # Market data for this step
            portfolio_valuation=portfolio_valuation,
            portfolio_distribution=portfolio_distribution,
            reward=0.0 # Placeholder, reward is calculated next
        )

        # --- 7. Calculate Reward ---
        # Reward is based on the change caused by the last step's action and market movement
        # Use the reward function, passing the history object
        reward = 0.0 if terminated else self.reward_function(self.historical_info)
        # Update the reward in the history for the current step
        self.historical_info["reward", -1] = reward

        # --- 8. Prepare Return Values ---
        # Get the observation for the *next* state (which is the current _idx after increment)
        observation = self._get_obs()
        # Get info for the current step
        info = self.historical_info[-1]

        # --- 9. Handle Episode End ---
        if terminated or truncated:
            # Perform end-of-episode calculations and logging
            self.calculate_metrics()
            self.log()
            # If terminated, the episode ends here.
            # If truncated, the episode ends here.

        return observation, reward, terminated, truncated, info


    def add_metric(self, name: str, function: callable) -> None:
        """
        Adds a custom metric to be calculated and logged at the end of each episode.

        Args:
            name: The display name of the metric.
            function: A callable function that takes the `History` object as input
                      and returns the metric value (preferably as a string or number).
                      Example: `lambda history: history['portfolio_valuation', -1]`
        """
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string.")
        if not callable(function):
            raise TypeError("Metric function must be callable.")
        self.log_metrics.append({'name': name, 'function': function})


    def calculate_metrics(self) -> None:
        """
        Calculates standard performance metrics at the end of an episode.
        Stores the results in `self.results_metrics`.
        """
        if not self.historical_info or len(self.historical_info) < 2:
            self.results_metrics = {"Error": "Not enough data to calculate metrics."}
            return

        # Safely access history data
        start_close = self.historical_info['data_close', 0]
        end_close = self.historical_info['data_close', -1]
        start_valuation = self.historical_info['portfolio_valuation', 0]
        end_valuation = self.historical_info['portfolio_valuation', -1]

        # Market Return
        market_return_pct = 0.0
        if start_close != 0:
            market_return_pct = 100 * (end_close / start_close - 1)

        # Portfolio Return
        portfolio_return_pct = 0.0
        if start_valuation != 0:
            portfolio_return_pct = 100 * (end_valuation / start_valuation - 1)

        self.results_metrics = {
            "Episode Length": self._step,
            "Market Return": f"{market_return_pct:5.2f}%",
            "Portfolio Return": f"{portfolio_return_pct:5.2f}%",
            "Final Valuation": f"{end_valuation:10.2f}",
            # Add more default metrics here if desired (e.g., Sharpe Ratio, Max Drawdown)
        }

        # Calculate custom metrics
        for metric in self.log_metrics:
            try:
                self.results_metrics[metric['name']] = metric['function'](self.historical_info)
            except Exception as e:
                warnings.warn(f"Could not calculate custom metric '{metric['name']}': {e}", UserWarning)
                self.results_metrics[metric['name']] = "Error"


    def get_metrics(self) -> dict:
        """
        Returns the dictionary of calculated metrics for the last completed episode.

        Returns:
            A dictionary containing metric names and their values.
        """
        return self.results_metrics


    def log(self) -> None:
        """
        Prints the calculated episode metrics to the console if `verbose` level is sufficient.
        """
        if self.verbose > 0:
            # Format metrics for printing
            log_message = " | ".join(f"{key}: {value}" for key, value in self.results_metrics.items())
            print(f"Episode End ({self.name}): {log_message}")


    def save_for_render(self, dir: str = "render_logs") -> None:
        """
        Saves the historical data of the current episode to a pickle file,
        suitable for later analysis or rendering.

        The saved file includes the original DataFrame columns joined with the
        portfolio and action history from the episode.

        Args:
            dir: The directory where the log file will be saved. Defaults to "render_logs".
        """
        if not self.historical_info:
            warnings.warn("No historical info to save. Run an episode first.", UserWarning)
            return

        # Ensure required columns for basic rendering are present in the original df
        required_render_cols = {"open", "high", "low", "close"}
        if not required_render_cols.issubset(self.df.columns):
             warnings.warn(f"DataFrame missing one or more of {required_render_cols}. Rendering might be limited.", UserWarning)

        # Create DataFrame from history
        # Exclude date objects if they cause issues, use the index directly
        history_data = self.historical_info.to_dict() # Get all data as dict of lists
        # Ensure all lists have the same length
        expected_len = len(self.historical_info)
        valid_cols = {k: v for k, v in history_data.items() if len(v) == expected_len}
        history_df = pd.DataFrame(valid_cols)

        # Use the 'date' column from history as the index
        if 'date' in history_df.columns:
            history_df.set_index("date", inplace=True)
        else:
            warnings.warn("History data missing 'date' column. Cannot merge properly with DataFrame.", UserWarning)
            # Fallback: Use integer index if date is missing? Might not align.
            # For now, proceed without date index if missing.

        # Select relevant columns from the original DataFrame (self.df)
        # Use the index range covered by the history
        start_date = history_df.index.min()
        end_date = history_df.index.max()
        render_df_market = self.df.loc[start_date:end_date].copy()

        # Join market data with history data
        # Use inner join to ensure alignment on the DatetimeIndex
        render_df = render_df_market.join(history_df, how="inner")

        # Create directory if it doesn't exist
        os.makedirs(dir, exist_ok=True)
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = os.path.join(dir, f"{self.name}_render_{timestamp}.pkl")

        # Save to pickle
        try:
            render_df.to_pickle(filepath)
            if self.verbose > 0:
                print(f"Render data saved to: {filepath}")
        except Exception as e:
            warnings.warn(f"Failed to save render data to {filepath}: {e}", UserWarning)


class MultiDatasetTradingEnv(TradingEnv):
    """
    An extension of `TradingEnv` that manages multiple datasets (e.g., different stocks,
    time periods) and switches between them periodically during training.

    This is useful for training agents that generalize across various market conditions.

    Inherits all parameters and methods from `TradingEnv`, plus additional ones below.

    ### Additional Parameters:
    - **dataset_dir**: `str` - A glob pattern specifying the path to the dataset files
                               (e.g., "data/stocks/*.pkl"). Files should be pandas
                               DataFrames loadable with `pd.read_pickle`.
    - **preprocess**: `callable` - A function that takes a loaded DataFrame as input
                                   and returns a preprocessed DataFrame. Use this for
                                   feature engineering or cleaning specific to each dataset.
                                   Defaults to an identity function (no preprocessing).
    - **episodes_between_dataset_switch**: `int` - The number of episodes to run on a
                                                   single dataset before potentially switching
                                                   to a new one. Defaults to 1.
    """
    def __init__(
        self,
        dataset_dir: str,
        *args, # Pass other TradingEnv arguments positionally
        preprocess: callable = lambda df: df, # Default: identity function
        episodes_between_dataset_switch: int = 1,
        **kwargs # Pass other TradingEnv arguments by keyword
    ):
        """Initializes the MultiDatasetTradingEnv."""

        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        if not callable(preprocess):
            raise TypeError("preprocess must be a callable function.")
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        if not isinstance(episodes_between_dataset_switch, int) or episodes_between_dataset_switch < 1:
            raise ValueError("episodes_between_dataset_switch must be a positive integer.")

        # Find dataset files matching the pattern
        self.dataset_paths = glob.glob(dataset_dir)
        if not self.dataset_paths:
            raise FileNotFoundError(f"No dataset files found matching pattern: {dataset_dir}")
        self.dataset_paths.sort() # Ensure consistent order

        # Track usage count for each dataset to encourage variety
        self.dataset_nb_uses = np.zeros(len(self.dataset_paths), dtype=int)
        self._episodes_on_this_dataset = 0 # Counter for episodes on current dataset

        # Load the first dataset to initialize the parent TradingEnv
        first_df = self._load_and_preprocess_dataset(0) # Load the first dataset initially
        self.current_dataset_idx = 0
        self.dataset_nb_uses[0] += 1

        # Initialize the parent TradingEnv with the first dataset and other args/kwargs
        super().__init__(df=first_df, *args, **kwargs)
        # Set the initial environment name based on the first dataset file
        self.name = Path(self.dataset_paths[self.current_dataset_idx]).stem


    def _load_and_preprocess_dataset(self, dataset_index: int) -> pd.DataFrame:
        """Loads and preprocesses a dataset specified by its index."""
        if not (0 <= dataset_index < len(self.dataset_paths)):
            raise IndexError(f"Dataset index {dataset_index} out of bounds.")

        dataset_path = self.dataset_paths[dataset_index]
        try:
            df = pd.read_pickle(dataset_path)
        except Exception as e:
            raise IOError(f"Failed to load dataset: {dataset_path}. Error: {e}")

        if not isinstance(df, pd.DataFrame):
             raise TypeError(f"Loaded file is not a DataFrame: {dataset_path}")

        # Apply preprocessing function
        try:
            processed_df = self.preprocess(df)
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing dataset {dataset_path}: {e}")

        if not isinstance(processed_df, pd.DataFrame):
             raise TypeError(f"Preprocessing function did not return a DataFrame for: {dataset_path}")

        return processed_df


    def _select_next_dataset_index(self) -> int:
        """Selects the index of the next dataset to use, prioritizing least used."""
        # Find indices of datasets with the minimum usage count
        min_uses = self.dataset_nb_uses.min()
        least_used_indices = np.where(self.dataset_nb_uses == min_uses)[0]

        # Randomly choose one from the least used datasets
        next_idx = np.random.choice(least_used_indices)
        return next_idx


    def reset(self, seed=None, options=None, **kwargs):
        """
        Resets the environment. Switches to a new dataset if the required number
        of episodes has been completed on the current one.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent reset method.

        Returns:
            tuple: A tuple containing the initial observation and initial info dictionary
                   for the (potentially new) dataset.
        """
        super().reset(seed=seed) # Handle seeding in parent

        self._episodes_on_this_dataset += 1

        # Check if it's time to switch datasets
        if self._episodes_on_this_dataset >= self.episodes_between_dataset_switch:
            self._episodes_on_this_dataset = 0 # Reset counter

            # Select the next dataset index (prioritizing least used)
            self.current_dataset_idx = self._select_next_dataset_index()
            self.dataset_nb_uses[self.current_dataset_idx] += 1 # Increment usage count

            # Load and preprocess the new dataset
            new_df = self._load_and_preprocess_dataset(self.current_dataset_idx)

            # Update the environment's internal DataFrame and related arrays
            self._set_df(new_df)

            # Update the environment name
            self.name = Path(self.dataset_paths[self.current_dataset_idx]).stem

            if self.verbose > 1:
                print(f"Switched to dataset: {self.name} (Uses: {self.dataset_nb_uses[self.current_dataset_idx]})")

        # Reset the state using the potentially new DataFrame via the parent's reset logic
        # Pass the original seed and options down if they were provided
        # Note: The parent reset logic needs to correctly handle the new self.df
        # We call super().reset() again, but it mainly resets step counters and portfolio.
        # The critical part is that self.df and related arrays are already updated.
        # However, calling reset twice might have unintended side effects depending on parent implementation.
        # A cleaner way is to integrate the dataset switching *before* the main reset logic.

        # --- Revised Reset Logic ---
        # 1. Determine if dataset switch is needed
        switch_dataset = self._episodes_on_this_dataset >= self.episodes_between_dataset_switch

        # 2. If switching, load new data and update internal state
        if switch_dataset:
            self._episodes_on_this_dataset = 0
            self.current_dataset_idx = self._select_next_dataset_index()
            self.dataset_nb_uses[self.current_dataset_idx] += 1
            new_df = self._load_and_preprocess_dataset(self.current_dataset_idx)
            self._set_df(new_df) # This updates self.df, _obs_array, etc.
            self.name = Path(self.dataset_paths[self.current_dataset_idx]).stem
            if self.verbose > 1:
                print(f"Switched to dataset: {self.name} (Uses: {self.dataset_nb_uses[self.current_dataset_idx]})")

        # 3. Call the parent's reset method *once* to reset portfolio, step, idx, etc.
        #    using the (potentially new) dataset loaded in _set_df.
        # Pass the original seed/options if provided.
        return super().reset(seed=seed, options=options, **kwargs)