#%%
from datetime import datetime
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("D:/coding/mt5fin/dev"))

from quant_tools import (
    fetch_fx, CurrencyPair, SignalProcessor,
    generate_trades, Fees,
    FixedStop, TrailingStop, BreakEvenStepStop,
    BacktestEngine, plot_heatmap, filter_strategies 
)

# -----------------------------
# Strategy function (local)
# -----------------------------
def sma_crossover(price, fast_length, slow_length):
    fast_sma = price.rolling(fast_length).mean()
    slow_sma = price.rolling(slow_length).mean()
    buy = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
    sell = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
    return {"buy": buy, "sell": sell}

#%%
# -----------------------------
# 1. Fetch FX data
# -----------------------------
usd_jpy = CurrencyPair("USD/JPY", pip_size=0.01, contract_size=100_000)
df = fetch_fx(usd_jpy.symbol, start=datetime(2025, 1, 1), end=datetime(2025, 1, 31))

# -----------------------------
# 2. Setup fees 
# -----------------------------
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)

# -----------------------------
# 3. Sweep analysis
# -----------------------------
fast_sma_range = range(5, 31, 1)
slow_sma_range = range(6, 55, 1)
risk_pct_list = [1, 2, 5, 10]
sl_pips_list = [20, 30, 50, 70]

results = []

for fast in fast_sma_range:
    for slow in slow_sma_range:
        if slow <= fast:
            continue

        sp = SignalProcessor(df['close'])
        sp.add_signal(f"SMA_{fast}_{slow}", lambda price, f=fast, s=slow: sma_crossover(price, f, s))
        signals = sp.get_signals(f"SMA_{fast}_{slow}")

        for sl_pips in sl_pips_list:
            for risk in risk_pct_list:
                trades = generate_trades(signals, df, planned_sl_pips=sl_pips)
                for t in trades:
                    t.risk_pct = risk

                exit_rules = [
                    FixedStop(sl_pips=sl_pips, pip_size=usd_jpy.pip_size),
                    TrailingStop(trail_pips=30, pip_size=usd_jpy.pip_size),
                    BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=usd_jpy.pip_size,
                                      commission=fees.commission, overnight_fee=fees.overnight_fee)
                ]

                engine = BacktestEngine(df, trades, usd_jpy, fees)
                engine.run_backtest(initial_capital=1000, exit_rules=exit_rules)

                results.append({
                    "fast_sma": fast,
                    "slow_sma": slow,
                    "sl_pips": sl_pips,
                    "risk_pct": risk,
                    "num_trades": engine.metrics['num_trades'],
                    "cumulative_pnl": engine.metrics['cumulative_pnl'],
                    "max_drawdown": engine.metrics['max_drawdown'],
                    "win_rate": engine.metrics['win_rate'],
                    "sharpe": engine.metrics['sharpe'],
                    "cagr": engine.metrics['cagr'],
                    "total_fees": engine.metrics['total_fees']
                })

df_results = pd.DataFrame(results)

#%%
# -----------------------------
# 4. Plot heatmaps
# -----------------------------
plot_heatmap(df_results,
             x_col="fast_sma",
             y_col="slow_sma",
             value_col="cumulative_pnl",
             fixed_params={"risk_pct": 5, "sl_pips": 50})

plot_heatmap(df_results,
             x_col="sl_pips",
             y_col="risk_pct",
             value_col="sharpe",
             fixed_params={"fast_sma": 5, "slow_sma": 45})

#%%
# -----------------------------
# 5. Benchmark good strategies
# -----------------------------
good_strats = filter_strategies(df_results, min_pnl_pct=5, max_dd_pct=15, initial_capital=1000)
print(good_strats.sort_values(by="cumulative_pnl", ascending=False))

#%%
# -----------------------------
# 6. Focused Lookback (example: SMA 10/20, SL=50, Risk=10%)
# -----------------------------
fast, slow, sl_pips, risk = 10, 20, 50, 10

sp = SignalProcessor(df['close'])
sp.add_signal("SMA_10_20", lambda price: sma_crossover(price, fast, slow))
signals = sp.get_signals("SMA_10_20")

trades = generate_trades(signals, df, planned_sl_pips=sl_pips)
for t in trades:
    t.risk_pct = risk

exit_rules = [
    FixedStop(sl_pips=sl_pips, pip_size=usd_jpy.pip_size),
    TrailingStop(trail_pips=30, pip_size=usd_jpy.pip_size),
    BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=usd_jpy.pip_size,
                      commission=fees.commission, overnight_fee=fees.overnight_fee)
]

engine = BacktestEngine(df, trades, usd_jpy, fees)
engine.run_backtest(initial_capital=1000, exit_rules=exit_rules)

# A. Trade log
trade_log = pd.DataFrame([{
    "Entry Time": t.entry_time,
    "Exit Time": t.exit_time,
    "Side": "Long" if t.side == 1 else "Short",
    "Entry Price": t.entry_price,
    "Exit Price": t.exit_price,
    "Lot Size": float(f"{t.lot_size:.3f}") if t.lot_size is not None else None,
    "PnL": round(t.pnl, 2) if t.pnl is not None else None,
    "Win": t.win,
    "Drawdown": round(t.drawdown, 2) if t.drawdown is not None else None,
    "Entry Equity": round(t.entry_equity, 2) if t.entry_equity is not None else None,
    "Exit Equity": round(t.exit_equity, 2) if t.exit_equity is not None else None,
    "Intraday Equity Min": round(min(t.intraday_equity), 2) if t.intraday_equity else None,
    "Intraday Equity Max": round(max(t.intraday_equity), 2) if t.intraday_equity else None,
    "Fees": round(t.entry_equity + t.pnl - t.exit_equity, 2),
    "Strategy": t.strategy,
    "Cancelled": t.cancelled
} for t in trades])

print(trade_log)

# B. Equity curve with drawdowns
plt.figure(figsize=(14, 6))
plt.plot(engine.equity_curve, label="Equity Curve", color="blue")
plt.fill_between(engine.drawdown_series.index,
                 engine.equity_curve - engine.drawdown_series,
                 engine.equity_curve,
                 color='red', alpha=0.3, label="Drawdown")
plt.title(f"Equity Curve with Drawdown (SMA {fast}/{slow}, SL={sl_pips}, Risk={risk}%)")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.show()
#%%