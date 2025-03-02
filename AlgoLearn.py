import backtrader as bt
import datetime
import itertools
import multiprocessing
import pandas as pd

class RSIEMAStrategy(bt.Strategy):
    params = (("rsi_period", 14), ("ema_period", 20))

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.ema = bt.indicators.EMA(period=self.params.ema_period)

    def next(self):
        if self.data.close[0] > self.ema[0] and self.rsi[0] > 60:
            self.buy()
        elif self.data.close[0] < self.ema[0] and self.rsi[0] < 40:
            self.sell()

def run_strategy(rsi_period, ema_period):
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    cerebro.addstrategy(RSIEMAStrategy, rsi_period=rsi_period, ema_period=ema_period)
    
    data = bt.feeds.GenericCSVData(
        dataname="INDEX_ETHUSD, 1D.csv",
        dtformat=('%Y-%m-%d'),
        timeframe=bt.TimeFrame.Days,
        compression=1,
        openinterest=-1,
        todate=datetime.datetime(2025, 2, 9))

    cerebro.adddata(data)
    cerebro.broker.set_cash(100000)
    
    start_value = cerebro.broker.getvalue()
    cerebro.run()
    end_value = cerebro.broker.getvalue()
    
    return rsi_period, ema_period, end_value - start_value

if __name__ == "__main__":
    combinations = list(itertools.product(range(8, 21), range(20, 201, 5)))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(run_strategy, combinations)
    
    df = pd.DataFrame(results, columns=["RSI_Period", "EMA_Period", "Net_Profit"])
    df.to_csv("optimization_results.csv", index=False)
    
    print("Optimization complete! Results saved to optimization_results.csv")

