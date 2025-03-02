import backtrader as bt
import datetime
import numpy as np


class TradeCounter(bt.Analyzer):
    def __init__(self):
        self.trade_count = 0
        self.equity_curve = []

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.trade_count += 1
            self.equity_curve.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return {'trades': list(range(1, self.trade_count + 1)), 'equity': self.equity_curve}


class RSIEMAStrategy(bt.Strategy):
    params = (
        ('rsi_period', 8),
        ('sma_period', 50),
        ('ema_period', 20)
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.sma = bt.indicators.SMA(period=self.params.sma_period)
        self.ema = bt.indicators.EMA(period=self.params.ema_period)

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        close_price = self.datas[0].close[0]

        if self.data.close[0] > self.ema[0] and self.rsi[0] > 60 and self.position.size <= 0:
            self.close()
            self.buy(size=10)
            print(f"{dt} - BUY EXECUTED at {close_price}")

        elif self.data.close[0] < self.ema[0] and self.rsi[0] < 40 and self.position.size >= 0:
            self.close()
            self.sell(size=10)
            print(f"{dt} - SELL EXECUTED at {close_price}")

        position_status = "Long" if self.position.size > 0 else "Short" if self.position.size < 0 else "No Trades"
        print(f"{dt} - Close Price: {close_price} - {position_status} - RSI: {self.rsi[0]} - SMA: {self.sma[0]}")

    def notify_order(self, order):
        if order.status in [bt.Order.Completed]:
            exec_price = order.executed.price
            dt = self.datas[0].datetime.datetime(0)
            print(f"{dt} - ORDER EXECUTED at {exec_price}, Size: {order.executed.size}")
        elif order.status in [bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected]:
            print("Order Canceled/Rejected")


cerebro = bt.Cerebro()
cerebro.broker.set_coc(True)
cerebro.broker.setcash(100000.0)

# Load data
data = bt.feeds.YahooFinanceCSVData(
    dataname='INDEX_ETHUSD, 1D.csv',
    reverse=False,
    adjclose=False,
    todate=datetime.datetime(2025, 2, 9)
)

cerebro.adddata(data)
cerebro.addstrategy(RSIEMAStrategy)
cerebro.addanalyzer(TradeCounter, _name='trade_counter')

starting_value = cerebro.broker.getvalue()
print('Starting Portfolio Value: %.2f' % starting_value)

results = cerebro.run()
trade_analysis = results[0].analyzers.trade_counter.get_analysis()

ending_value = cerebro.broker.getvalue()
print('Final Portfolio Value: %.2f' % ending_value)

total_return = (ending_value - starting_value) / starting_value * 100
print('Total Return: %.2f%%' % total_return)

net_profit = ending_value - starting_value
print('Net Profit: %.2f' % net_profit)
print("Remaining Cash:", cerebro.broker.get_cash())

# Compute Sharpe Ratio
risk_free_rate_annual = 0.02  # 2% annual risk-free rate
risk_free_rate_daily = risk_free_rate_annual / 252  # Convert to daily

equity_curve = np.array(trade_analysis['equity'])

if len(equity_curve) > 1:
    log_returns = np.diff(np.log(equity_curve))  # Calculate log returns
    mean_excess_return = np.mean(log_returns) - risk_free_rate_daily
    std_dev = np.std(log_returns)

    if std_dev > 0:
        sharpe_ratio = mean_excess_return / std_dev * np.sqrt(252)  # Convert to annualized Sharpe Ratio
    else:
        sharpe_ratio = 0

    print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
else:
    print("Not enough trades to compute Sharpe Ratio.")

# Plot equity curve
#plt.figure(figsize=(10, 5))
#plt.plot(trade_analysis['trades'], trade_analysis['equity'], marker='o', linestyle='-')
#plt.xlabel('Number of Trades')
#plt.ylabel('Equity ($)')
#plt.title('Equity Curve vs. Number of Trades')
#plt.grid()
#plt.show()

cerebro.plot(style='candlestick')
