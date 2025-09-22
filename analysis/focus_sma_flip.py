# sma_slope_interactive.py
from datetime import datetime
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("D:/coding/mt5fin/dev"))
from quant_tools import fetch_fx, CurrencyPair, Fees
from quant_tools.backtest import BacktestEngine
from quant_tools.exits import FixedStop, TrailingStop, BreakEvenStepStop

class SMAEntrySlope:
    def __init__(self, df: pd.DataFrame, fast: int, slow: int,
                 strategy="SMA_Slope", risk_pct=1.0, sl_pips=50):
        if slow < fast + 4:
            raise ValueError("Require slow ≥ fast+4")
        self.strategy = strategy
        self.risk_pct = float(risk_pct)
        self.sl_pips  = int(sl_pips)
        self.fast_sma = df["close"].rolling(fast).mean()
        self.slow_sma = df["close"].rolling(slow).mean()
        self.trend_sma = df["close"].rolling(200).mean()
        self.index = df.index

    def check(self, ts, price, df):
        i = self.index.get_loc(ts)
        if i == 0: return False, {}
        f_now, s_now = self.fast_sma.iloc[i], self.slow_sma.iloc[i]
        f_prev, s_prev = self.fast_sma.iloc[i-1], self.slow_sma.iloc[i-1]
        t_now, t_prev = self.trend_sma.iloc[i], self.trend_sma.iloc[i-1]
        if any(pd.isna(x) for x in [f_now, s_now, f_prev, s_prev, t_now, t_prev]):
            return False, {}
        slope_up, slope_down = (t_now > t_prev), (t_now < t_prev)
        cross_up = (f_now > s_now) and (f_prev <= s_prev) and slope_up
        cross_dn = (f_now < s_now) and (f_prev >= s_prev) and slope_down
        if cross_up:
            return True, {"side": +1, "strategy": self.strategy,
                          "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
        if cross_dn:
            return True, {"side": -1, "strategy": self.strategy,
                          "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
        return False, {}

# --- Config ---
SYMBOL = "USD/JPY"
START, END = datetime(2025, 1, 1), datetime(2025, 1, 31)
INITIAL_CAPITAL = 1000.0
FAST, SLOW, SL_PIPS, RISK_PCT = 8, 12, 30, 2.0

pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)
df = fetch_fx(pair.symbol, start=START, end=END)

entry = SMAEntrySlope(df, FAST, SLOW, risk_pct=RISK_PCT, sl_pips=SL_PIPS)
exit_rules = [
    FixedStop(sl_pips=SL_PIPS, pip_size=pair.pip_size),
    TrailingStop(trail_pips=30, pip_size=pair.pip_size),
    BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                      commission=fees.commission, overnight_fee=fees.overnight_fee),
]

engine = BacktestEngine(df, pair, fees,
                        max_active_trades=1,
                        spread_pips=0.5,
                        slippage_pips=0.1)
engine.run(initial_capital=INITIAL_CAPITAL, entry_rules=[entry], exit_rules=exit_rules)

# Trade log
trade_log = pd.DataFrame([{
    "Entry Time": t.entry_time, "Exit Time": t.exit_time,
    "Side": ("Long" if t.side == 1 else "Short"),
    "Entry Price": t.entry_price, "Exit Price": t.exit_price,
    "Lot Size": None if t.lot_size is None else float(f"{t.lot_size:.3f}"),
    "PnL": None if t.pnl is None else round(t.pnl, 2),
    "Win": t.win,
    "Drawdown": None if t.drawdown is None else round(t.drawdown, 2),
    "Entry Equity": None if t.entry_equity is None else round(t.entry_equity, 2),
    "Exit Equity":  None if t.exit_equity  is None else round(t.exit_equity, 2),
    "Fees": round(t.total_fees, 2),
    "Strategy": t.strategy, "Cancelled": t.cancelled
} for t in engine.trades])
print(trade_log)

# Equity curve
plt.figure(figsize=(14, 6))
plt.plot(engine.equity_curve, label="Equity Curve")
plt.fill_between(engine.drawdown_series.index,
                 engine.equity_curve - engine.drawdown_series,
                 engine.equity_curve, alpha=0.3, label="Drawdown")
plt.legend(); plt.grid(True); plt.show()
