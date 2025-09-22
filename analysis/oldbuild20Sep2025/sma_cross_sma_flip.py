#%%
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath("D:/coding/mt5fin/dev"))

from quant_tools import (
    fetch_fx, CurrencyPair,
    Fees,
    FixedStop, TrailingStop, BreakEvenStepStop,
    BacktestEngine, plot_heatmap, filter_strategies
)

# -----------------------------
# Strategy-local entry rule (lives HERE, not in the package)
# -----------------------------
class SMAFlipEntry:
    """
    Flip long/short on fast/slow SMA cross.
    Lives in this strategy script so you can tweak freely.
    """
    def __init__(self, df: pd.DataFrame, fast: int, slow: int,
                 strategy="SMA", risk_pct=1.0, sl_pips=50):
        self.strategy = strategy
        self.risk_pct = float(risk_pct)
        self.sl_pips = sl_pips
        self.fast = fast
        self.slow = slow

        price = df['close']
        # Precompute once; still realistic (uses only data up to bar i when checking)
        self.fast_sma = price.rolling(fast).mean()
        self.slow_sma = price.rolling(slow).mean()
        self.index = df.index

    def check(self, ts, price, df):
        # Called each bar by the engine
        i = self.index.get_loc(ts)
        if i == 0:
            return False, {}
        f_now, s_now = self.fast_sma.iloc[i], self.slow_sma.iloc[i]
        f_prev, s_prev = self.fast_sma.iloc[i-1], self.slow_sma.iloc[i-1]
        if pd.isna(f_prev) or pd.isna(s_prev) or pd.isna(f_now) or pd.isna(s_now):
            return False, {}

        cross_up = (f_now > s_now) and (f_prev <= s_prev)
        cross_dn = (f_now < s_now) and (f_prev >= s_prev)

        if cross_up:
            return True, {"side": +1, "strategy": self.strategy,
                          "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
        if cross_dn:
            return True, {"side": -1, "strategy": self.strategy,
                          "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
        return False, {}
#%%
# -----------------------------
# 1) Fetch FX data
# -----------------------------
usd_jpy = CurrencyPair("USD/JPY", pip_size=0.01, contract_size=100_000)
df = fetch_fx(usd_jpy.symbol, start=datetime(2025, 1, 1), end=datetime(2025, 1, 31))

# -----------------------------
# 2) Fees
# -----------------------------
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)

# -----------------------------
# 3) Parameter sweep (realistic engine)
# -----------------------------
fast_sma_range = range(5, 31, 5)
slow_sma_range = range(6, 55, 5)
risk_pct_list = [1, 2, 5, 10]
sl_pips_list = [20, 30, 50, 70]

results = []

for fast in fast_sma_range:
    for slow in slow_sma_range:
        if slow <= fast:
            continue
        for sl_pips in sl_pips_list:
            for risk in risk_pct_list:
                entry_rules = [SMAFlipEntry(df, fast, slow,
                                            strategy=f"SMA_{fast}_{slow}",
                                            risk_pct=risk, sl_pips=sl_pips)]
                exit_rules = [
                    FixedStop(sl_pips=sl_pips, pip_size=usd_jpy.pip_size),
                    TrailingStop(trail_pips=30, pip_size=usd_jpy.pip_size),
                    BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=usd_jpy.pip_size,
                                      commission=fees.commission, overnight_fee=fees.overnight_fee)
                ]

                engine = BacktestEngine(df, usd_jpy, fees)
                engine.run(initial_capital=1000, entry_rules=entry_rules, exit_rules=exit_rules)

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
# 4) Heatmaps
# -----------------------------
plot_heatmap(df_results,
             x_col="fast_sma",
             y_col="slow_sma",
             value_col="cumulative_pnl",
             fixed_params={"risk_pct": 5, "sl_pips": 50})
#%%
plot_heatmap(df_results,
             x_col="sl_pips",
             y_col="risk_pct",
             value_col="sharpe",
             fixed_params={"fast_sma": 5, "slow_sma": 45})
#%%
good_strats = filter_strategies(df_results, min_pnl_pct=5, max_dd_pct=10, initial_capital=1000)
print(good_strats.sort_values(by="cumulative_pnl", ascending=False))

#%%
# -----------------------------
# 5) Focused look
# -----------------------------
fast, slow, sl_pips, risk = 5, 11, 1, 30
entry_rules = [SMAFlipEntry(df, fast, slow,
                            strategy=f"SMA_{fast}_{slow}",
                            risk_pct=risk, sl_pips=sl_pips)]
exit_rules = [
    FixedStop(sl_pips=sl_pips, pip_size=usd_jpy.pip_size),
    TrailingStop(trail_pips=30, pip_size=usd_jpy.pip_size),
    BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=usd_jpy.pip_size,
                      commission=fees.commission, overnight_fee=fees.overnight_fee)
]

engine = BacktestEngine(df, usd_jpy, fees)
engine.run(initial_capital=1000, entry_rules=entry_rules, exit_rules=exit_rules)

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
    "Fees": round(t.total_fees, 2),
    "Strategy": t.strategy,
    "Cancelled": t.cancelled
} for t in engine.trades])
print(trade_log)

plt.figure(figsize=(14, 6))
plt.plot(engine.equity_curve, label="Equity Curve")
plt.fill_between(engine.drawdown_series.index,
                 engine.equity_curve - engine.drawdown_series,
                 engine.equity_curve,
                 alpha=0.3, label="Drawdown")
plt.title(f"Equity Curve with Drawdown (SMA {fast}/{slow}, SL={sl_pips}, Risk={risk}%)")
plt.xlabel("Time"); plt.ylabel("Equity"); plt.legend(); plt.grid(True); plt.show()
#%%
