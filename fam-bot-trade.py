'''
Strategies
look at the strategies daily (1-milli second) 
and then watch 1 minute of close resistance breakout order 
place order between the resistance and the breakout point 
ATR for the stop loss 
'''

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

# Load the millisecond dataset
file_path = "BTC Dataset/M1_BTC.USD_with_volume_2024_03.parquet"
df = pd.read_parquet(file_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Step 1: Aggregate data to 3600-millisecond intervals
ohlc_3600ms = df.resample('3600L').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Step 2: Define the Breakout Strategy
class BreakoutStrategy(Strategy):
    atr_period = 14  # ATR calculation period
    resistance_period = 60  # Resistance calculation period in 3600 ms intervals
    
    def init(self):
        # Calculate ATR for stop-loss
        self.atr = self.I(self.calculate_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Calculate resistance levels using high prices
        self.resistance = self.I(self.calculate_resistance, self.data.High, self.resistance_period)

    def calculate_atr(self, high, low, close, period):
        """Calculate ATR based on high, low, and close."""
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        return tr.rolling(period).mean()

    def calculate_resistance(self, high, period):
        """Calculate resistance level as the rolling maximum of highs."""
        return high.rolling(period).max()

    def next(self):
        # Fetch current resistance level and price data
        resistance = self.resistance[-1]
        close = self.data.Close[-1]
        high = self.data.High[-1]

        # Check for breakout
        if close > resistance:
            # Calculate entry price and stop loss
            entry_price = resistance + (close - resistance) * 0.5
            stop_loss = entry_price - self.atr[-1]

            # Place a buy order
            self.buy(sl=stop_loss)

# Step 3: Prepare data for backtesting
ohlc_3600ms.reset_index(inplace=True)
ohlc_3600ms.rename(columns={'timestamp': 'Date'}, inplace=True)
ohlc_3600ms.set_index('Date', inplace=True)

# Step 4: Backtest the strategy
bt = Backtest(ohlc_3600ms, BreakoutStrategy, cash=10000, commission=0.002)
stats = bt.run()
bt.plot()

