import pandas as pd
import numpy as np
from datetime import datetime
import dukascopy_python

class BacktestEngine:
    def __init__(self, price_df, trades, currency_pair, fees):
        self.df = price_df
        self.trades = trades
        self.pair = currency_pair
        self.fees = fees
        self.equity_curve = pd.Series(index=price_df.index, data=0.0)
        self.drawdown_series = pd.Series(index=price_df.index, data=0.0)
        self.metrics = {}

    def run_backtest(self, initial_capital=1000, exit_rules=None):
        equity = pd.Series(index=self.df.index, data=initial_capital, dtype=float)
        current_capital = float(initial_capital)
        global_max_equity = initial_capital
        risk_manager = RiskManager(capital=current_capital, currency_pair=self.pair)
        wins = 0
        total_fees_accum = 0.0

        for trade in self.trades:
            risk_manager.capital = current_capital
            trade = risk_manager.assign_trade_params(trade)
            if trade.cancelled or trade.lot_size == 0:
                trade.pnl = 0
                trade.win = False
                trade.drawdown = 0
                trade.entry_equity = current_capital
                trade.exit_equity = current_capital
                trade.intraday_equity = [current_capital]
                trade.exit_time = trade.entry_time
                trade.exit_price = trade.entry_price
                continue

            trade.entry_equity = current_capital
            trade.intraday_equity = []

            df_slice = self.df.loc[trade.entry_time:trade.exit_time]['close']
            exit_price = None
            exit_time = df_slice.index[-1]

            for ts, price in df_slice.items():
                trade.intraday_equity.append(current_capital)
                stop_hit = False
                for rule in exit_rules or []:
                    hit, sl_price = rule.update(trade, price, ts)
                    if hit:
                        exit_price = sl_price
                        exit_time = ts
                        stop_hit = True
                        break
                if stop_hit:
                    break

            if exit_price is None:
                exit_price = df_slice.iloc[-1]

            trade.exit_time = exit_time
            trade.exit_price = exit_price

            pip_diff = ((exit_price - trade.entry_price) / self.pair.pip_size
                        if trade.side == 1 else (trade.entry_price - exit_price) / self.pair.pip_size)
            trade.pnl = pip_diff * trade.lot_size * self.pair.pip_value_per_lot(trade.entry_price)
            num_bars = len(df_slice)
            trade_fees = self.fees.total_fees(trade, num_bars)
            trade.pnl -= trade_fees
            total_fees_accum += trade_fees
            current_capital += trade.pnl
            trade.exit_equity = current_capital
            trade.win = trade.pnl > 0
            wins += int(trade.win)
            global_max_equity = max(global_max_equity, current_capital)
            trade.drawdown = global_max_equity - current_capital
            equity.loc[trade.entry_time:exit_time] = current_capital
            self.drawdown_series.loc[trade.entry_time:exit_time] = global_max_equity - equity.loc[trade.entry_time:exit_time]

        self.equity_curve = equity
        self.metrics['cumulative_pnl'] = current_capital - initial_capital
        self.metrics['max_drawdown'] = max(self.drawdown_series)
        self.metrics['num_trades'] = len(self.trades)
        self.metrics['win_rate'] = (wins / len(self.trades) if self.trades else 0)
        self.metrics['total_fees'] = total_fees_accum
        returns = self.equity_curve.pct_change().dropna()
        self.metrics['sharpe'] = returns.mean() / returns.std() * np.sqrt(252 * (24*60/5)) if len(returns) > 1 else None
        total_days = (self.df.index[-1] - self.df.index[0]).days
        self.metrics['cagr'] = (current_capital / initial_capital) ** (365/total_days) - 1 if total_days > 0 else None

def filter_strategies(df, min_pnl_pct=5, max_dd_pct=10, initial_capital=1000):
    min_pnl = initial_capital * (min_pnl_pct / 100)
    max_dd = initial_capital * (max_dd_pct / 100)
    return df[
        (df['cumulative_pnl'] >= min_pnl) &
        (df['max_drawdown'] <= max_dd)
    ]




def fetch_fx(symbol, start: datetime, end: datetime, interval=dukascopy_python.INTERVAL_MIN_5):
    df = dukascopy_python.fetch(symbol, interval, dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
    df.index = df.index.tz_convert('Asia/Bangkok')
    return df

class ExitRule:
    def update(self, trade, price, timestamp):
        raise NotImplementedError

class FixedStop(ExitRule):
    def __init__(self, sl_pips, pip_size):
        self.sl_pips = sl_pips
        self.pip_size = pip_size
    def update(self, trade, price, timestamp):
        if trade.side == 1 and price <= trade.entry_price - self.sl_pips * self.pip_size:
            return True, price
        if trade.side == -1 and price >= trade.entry_price + self.sl_pips * self.pip_size:
            return True, price
        return False, None

class TrailingStop(ExitRule):
    def __init__(self, trail_pips, pip_size):
        self.trail_pips = trail_pips
        self.pip_size = pip_size
    def update(self, trade, price, timestamp):
        if trade.side == 1:
            trade.max_favorable_price = max(getattr(trade, "max_favorable_price", trade.entry_price), price)
            trail_price = trade.max_favorable_price - self.trail_pips * self.pip_size
            if price <= trail_price:
                return True, price
        else:
            trade.max_favorable_price = min(getattr(trade, "max_favorable_price", trade.entry_price), price)
            trail_price = trade.max_favorable_price + self.trail_pips * self.pip_size
            if price >= trail_price:
                return True, price
        return False, None

class BreakEvenStepStop(ExitRule):
    def __init__(self, trigger_pips, step_pips, pip_size, commission=0.0, overnight_fee=0.0):
        self.trigger_pips = trigger_pips
        self.step_pips = step_pips
        self.pip_size = pip_size
        self.commission = commission
        self.overnight_fee = overnight_fee
    def update(self, trade, price, timestamp):
        moved_sl = getattr(trade, "sl_price", None)
        fees_in_pips = (self.commission + self.overnight_fee) / (trade.lot_size * (trade.entry_price * self.pip_size))
        if trade.side == 1:
            profit_pips = (price - trade.entry_price) / self.pip_size
            if profit_pips >= self.trigger_pips:
                break_even_price = trade.entry_price + fees_in_pips * self.pip_size
                if moved_sl is None or moved_sl < break_even_price:
                    trade.sl_price = break_even_price
                else:
                    steps = int((profit_pips - self.trigger_pips) / self.step_pips)
                    trade.sl_price = max(trade.sl_price, break_even_price + steps * self.step_pips * self.pip_size)
            if trade.sl_price and price <= trade.sl_price:
                return True, price
        else:
            profit_pips = (trade.entry_price - price) / self.pip_size
            if profit_pips >= self.trigger_pips:
                break_even_price = trade.entry_price - fees_in_pips * self.pip_size
                if moved_sl is None or moved_sl > break_even_price:
                    trade.sl_price = break_even_price
                else:
                    steps = int((profit_pips - self.trigger_pips) / self.step_pips)
                    trade.sl_price = min(trade.sl_price, break_even_price - steps * self.step_pips * self.pip_size)
            if trade.sl_price and price >= trade.sl_price:
                return True, price
        return False, None

class CurrencyPair:
    def __init__(self, symbol, pip_size=0.01, contract_size=100_000, quote_currency="USD"):
        self.symbol = symbol
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.quote_currency = quote_currency

    def pip_value_per_lot(self, price):
        return (self.contract_size * self.pip_size) / price

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df, x_col, y_col, value_col,annot=False, fixed_params={}):
    filtered_df = df.copy()
    for k, v in fixed_params.items():
        filtered_df = filtered_df[filtered_df[k] == v]
    pivot_df = filtered_df.pivot(index=y_col, columns=x_col, values=value_col)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=annot, fmt=".1f", cmap="coolwarm")
    plt.title(f"{value_col} Heatmap ({', '.join([f'{k}={v}' for k,v in fixed_params.items()])})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

from .trades import Trade
from .instruments import CurrencyPair

class RiskManager:
    def __init__(self, capital, currency_pair: CurrencyPair,
                 default_risk_pct=1.0, default_sl_pips=50,
                 min_lot=0.01, max_lot=50, lot_step=0.01):
        self.capital = capital
        self.pair = currency_pair
        self.default_risk_pct = default_risk_pct
        self.default_sl_pips = default_sl_pips
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.lot_step = lot_step

    def assign_trade_params(self, trade: Trade):
        trade.risk_pct = trade.risk_pct or self.default_risk_pct
        sl_pips = trade.planned_sl_pips or self.default_sl_pips

        risk_amount = self.capital * trade.risk_pct / 100
        pip_value_per_lot = self.pair.pip_value_per_lot(trade.entry_price)
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        if lot_size < self.min_lot:
            trade.lot_size = 0
            trade.cancelled = True
            return trade

        lot_size = min(self.max_lot, round(lot_size / self.lot_step) * self.lot_step)
        trade.lot_size = lot_size
        trade.cancelled = False

        trade.sl_price = (trade.entry_price - sl_pips * self.pair.pip_size if trade.side == 1
                          else trade.entry_price + sl_pips * self.pair.pip_size)
        return trade

class Trade:
    def __init__(self, entry_time, exit_time, side, entry_price, planned_sl_pips=None, strategy=None):
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.side = side
        self.entry_price = entry_price
        self.planned_sl_pips = planned_sl_pips
        self.strategy = strategy

        self.lot_size = None
        self.sl_price = None
        self.exit_price = None
        self.pnl = None
        self.drawdown = None
        self.win = None
        self.risk_pct = None
        self.cancelled = False
        self.entry_equity = None
        self.exit_equity = None
        self.intraday_equity = []
        self.total_fees = 0.0

class Fees:
    def __init__(self, commission=0.0, overnight_fee=0.0, swap_fee=0.0):
        self.commission = commission
        self.overnight_fee = overnight_fee
        self.swap_fee = swap_fee

    def total_fees(self, trade, num_bars=1):
        return self.commission + num_bars * (self.overnight_fee + self.swap_fee)

def generate_trades(signals, df, planned_sl_pips=None, strategy_name="SMA"):
    trades = []
    open_trade = None
    for t, (buy, sell, price) in enumerate(zip(signals['buy'], signals['sell'], df['close'])):
        timestamp = df.index[t]
        if buy and open_trade is None:
            open_trade = Trade(timestamp, None, 1, price, planned_sl_pips, strategy_name)
        elif sell and open_trade is None:
            open_trade = Trade(timestamp, None, -1, price, planned_sl_pips, strategy_name)
        elif (buy or sell) and open_trade is not None:
            open_trade.exit_time = timestamp
            trades.append(open_trade)
            open_trade = None
    if open_trade is not None:
        open_trade.exit_time = df.index[-1]
        trades.append(open_trade)
    return trades
