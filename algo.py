import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

# Load price data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    df.dropna(inplace=True)
    return df

# Custom RSI indicator that uses the same calculation as in Pine Script
class CustomRSI(bt.Indicator):
    lines = ('rsi',)
    params = (('period', 14),)

    def __init__(self):
        self.addminperiod(self.params.period)
        self.rsi_source = self.data.close

    def next(self):
        change = self.rsi_source[0] - self.rsi_source[-1]
        up = max(change, 0)
        down = -min(change, 0)
        
        if len(self) < self.params.period:
            self.lines.rsi[0] = 50  # Default value until we have enough data
        else:
            # Calculate the average gains and losses
            up_avg = np.mean([max(self.rsi_source[i] - self.rsi_source[i - 1], 0) for i in range(-self.params.period + 1, 1)])
            down_avg = np.mean([-min(self.rsi_source[i] - self.rsi_source[i - 1], 0) for i in range(-self.params.period + 1, 1)])
            
            rs = up_avg / down_avg if down_avg != 0 else 0
            self.lines.rsi[0] = 100 - (100 / (1 + rs)) if down_avg != 0 else 100

# Strategy that uses Custom RSI and SMA for long and short positions
class RSISMAStrategy(bt.Strategy):
    params = (('sma_period', 10), ('rsi_period', 14), ('overbought', 60), ('oversold', 40), ('risk_per_trade', 0.1))

    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
        self.rsi = CustomRSI(self.data, period=self.params.rsi_period)
        self.closed_positions = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_positions.append(trade.pnl)

    def next(self):
        equity = self.broker.get_value()
        close_price = self.data.close[0]
        
        if isinstance(equity, (int, float)) and isinstance(close_price, (int, float)):
            position_size = (equity * self.params.risk_per_trade) // close_price
        else:
            position_size = 0  # default to 0 if types are incorrect

        if self.rsi[0] > self.params.overbought and close_price > self.sma[0]:
            if self.position.size <= 0:  # If not in position or short position
                self.close()
                self.buy(size=position_size)
        elif self.rsi[0] < self.params.oversold and close_price < self.sma[0]:
            if self.position.size >= 0:  # If not in position or long position
                self.close()
                self.sell(size=position_size)

# Backtest function
def backtest(df, strategy, params=None):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, **params if params else {})
    cerebro.broker.set_cash(10000)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    results = cerebro.run()
    strategy_instance = results[0]
    return cerebro, results, strategy_instance

# Calculate Omega ratio
def omega_ratio(returns, threshold=0):
    returns = np.array(returns)  # Ensure returns is a NumPy array
    positive_returns = returns[returns > threshold]
    negative_returns = returns[returns <= threshold]
    if negative_returns.sum() == 0:
        return np.inf
    omega = positive_returns.sum() / abs(negative_returns.sum())
    return omega

# Calculate Sortino ratio
def sortino_ratio(returns, risk_free_rate=0):
    returns = np.array(returns)  # Ensure returns is a NumPy array
    downside_returns = returns[returns < 0]
    expected_return = np.mean(returns) - risk_free_rate
    downside_deviation = np.std(downside_returns)
    return expected_return / downside_deviation

# Load data
data = load_data('ETH 1D priceData - INDEX_ETHUSD, 1D.csv')

# Initial backtest with RSISMAStrategy
print("Running RSISMAStrategy...")
cerebro, results, strategy_instance = backtest(data, RSISMAStrategy)
final_value = cerebro.broker.getvalue()
initial_value = cerebro.broker.startingcash
total_return = (final_value - initial_value) / initial_value * 100
returns = np.array(results[0].analyzers.returns.get_analysis()['rnorm100'])  # Ensure returns is a NumPy array
sharpe_ratio = results[0].analyzers.sharpe.get_analysis()['sharperatio']
trade_analyzer = results[0].analyzers.tradeanalyzer.get_analysis()

# Calculate cumulative closed positions
closed_positions = np.cumsum(strategy_instance.closed_positions)
trade_numbers = list(range(1, len(closed_positions) + 1))
daily_returns = np.diff(closed_positions) / closed_positions[:-1] if len(closed_positions) > 1 else []

# Calculate Omega ratio
omega = omega_ratio(daily_returns)
sortino = sortino_ratio(daily_returns)
total_trades = trade_analyzer.total.closed

print(f"Initial Portfolio Value: {initial_value}")
print(f"Final Portfolio Value: {final_value}")
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Omega Ratio: {omega}")
print(f"Sortino Ratio: {sortino}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annual Return: {returns:.2f}%")
print(f"Total Number of Trades: {total_trades}")

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(trade_numbers, closed_positions, label='Closed Positions')
plt.title('Closed Positions Over Number of Trades')
plt.xlabel('Number of Trades')
plt.ylabel('Closed Positions')
plt.legend()
plt.tight_layout()
plt.show()