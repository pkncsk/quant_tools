#%%
#%% focus_sma_confirm.py
from datetime import datetime
import os, sys
import pandas as pd
import pytz
import matplotlib.pyplot as plt

# --- Dev path ---
sys.path.append(os.path.abspath("D:/coding/quant/dev"))

# Package imports
from quant_tools import fetch_fx, CurrencyPair, Fees
from quant_tools.backtest import BacktestEngine
from quant_tools.exits import FixedStop, TrailingStop, BreakEvenStepStop

# --- Config ---
SYMBOL = "USD/JPY"
BANGKOK = pytz.timezone("Asia/Bangkok")
START = BANGKOK.localize(datetime(2025, 1, 1))
END   = BANGKOK.localize(datetime(2025, 3, 31))
INITIAL_CAPITAL = 1000.0

FAST, SLOW, MID, SL_PIPS, RISK_PCT = 14, 30, 100, 70, 1.0
WARMUP_BARS = 250

# --- Warmup offset ---
BAR_INTERVAL_MIN = 5
bars_per_day = int((24 * 60) / BAR_INTERVAL_MIN)
offset_days = (WARMUP_BARS // bars_per_day) + 1
DATA_START = START - pd.Timedelta(days=offset_days)
import dukascopy_python
pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)
df = fetch_fx(pair.symbol, start=DATA_START, end=END, interval=dukascopy_python.INTERVAL_MIN_10)


# --- Entry rule class ---
class SMAEntryConfirm:
    def __init__(self, df, fast=14, slow=30, mid=100,
                 risk_pct=1.0, sl_pips=50, strategy="SMA_Confirm",
                 max_wait=10):
        self.df = df
        self.fast_sma = df["close"].rolling(fast).mean()
        self.slow_sma = df["close"].rolling(slow).mean()
        self.mid_sma  = df["close"].rolling(mid).mean()
        self.index = df.index
        self.pending = None
        self.strategy = strategy
        self.risk_pct = risk_pct
        self.sl_pips = sl_pips
        self.max_wait = max_wait   # expire pending if too old

    def check(self, ts, price, df, state=None):
        """Return (hit, params) if entry confirmed."""
        i = self.index.get_loc(ts)
        f, s, m = self.fast_sma.iloc[i], self.slow_sma.iloc[i], self.mid_sma.iloc[i]
        f_prev, s_prev = self.fast_sma.iloc[i-1], self.slow_sma.iloc[i-1]
        m_prev = self.mid_sma.iloc[i-1]

        # Skip if warmup not finished
        if i < WARMUP_BARS or ts < START:
            return False, {}

        # --- Step 1: Detect new fast/slow cross (overwrite pending) ---
        if f > s and f_prev <= s_prev:
            self.pending = {"side": +1, "bar": i}
        elif f < s and f_prev >= s_prev:
            self.pending = {"side": -1, "bar": i}

        # --- Step 2a: Expire stale pending ---
        if self.pending and self.max_wait is not None:
            if i - self.pending["bar"] > self.max_wait:
                self.pending = None

        # --- Step 2b: Confirm only if flat ---
        if self.pending and state and state.get("active_trades", 0) == 0:
            side = self.pending["side"]

            if side == +1:
                # Confirm long if fast or slow crosses SMA100 upward
                if f > m and f_prev <= m_prev:
                    self.pending = None
                    return True, {"side": +1, "strategy": self.strategy,
                                  "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
                if s > m and s_prev <= m_prev:
                    self.pending = None
                    return True, {"side": +1, "strategy": self.strategy,
                                  "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}

            elif side == -1:
                # Confirm short if fast or slow crosses SMA100 downward
                if f < m and f_prev >= m_prev:
                    self.pending = None
                    return True, {"side": -1, "strategy": self.strategy,
                                  "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
                if s < m and s_prev >= m_prev:
                    self.pending = None
                    return True, {"side": -1, "strategy": self.strategy,
                                  "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}

        return False, {}




# --- Setup exits ---
exit_rules = [
    FixedStop(sl_pips=70, pip_size=pair.pip_size),
    BreakEvenStepStop(trigger_pips=70, step_pips=
                      30, pip_size=pair.pip_size,
                      commission=fees.commission, overnight_fee=fees.overnight_fee),
    TrailingStop(trail_pips=30, pip_size=pair.pip_size),
    
]

# --- Backtest ---
engine = BacktestEngine(df, pair, fees,
                        max_active_trades=1,
                        spread_pips=0.5,
                        slippage_pips=0.1)
entry_rule = SMAEntryConfirm(df, FAST, SLOW, MID, risk_pct=RISK_PCT, sl_pips=SL_PIPS)
engine.run(initial_capital=INITIAL_CAPITAL, entry_rules=[entry_rule], exit_rules=exit_rules)

# --- Results ---
trade_log = pd.DataFrame([{
    "Entry Time": t.entry_time, "Exit Time": t.exit_time,
    "Side": ("Long" if t.side == 1 else "Short"),
    "Entry Price": t.entry_price, "Exit Price": t.exit_price,
    "PnL": t.pnl, "Win": t.win,
    "Exit Reason": t.exit_reason 
} for t in engine.trades])
print(trade_log)

print("Metrics:", engine.metrics)

#%%
import matplotlib.ticker as mticker

# Convert index to day number relative to start
eq_curve = engine.equity_curve.copy()
dd_curve = engine.drawdown_series.copy()

# Drop timezone and compute day offset
days = (eq_curve.index.tz_convert(None) - START.replace(tzinfo=None)).days

plt.figure(figsize=(14, 6))
plt.plot(days, eq_curve, label="Equity Curve")
plt.fill_between(days, eq_curve - dd_curve, eq_curve,
                 alpha=0.3, label="Drawdown")
plt.xlabel("Day")
plt.ylabel("Equity")

# Add ticks every 5 days (weekly spacing)
ax = plt.gca()
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

plt.legend()
plt.grid(True)
plt.show()

#%%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    row_heights=[0.6, 0.4],
    subplot_titles=("Price + SMAs", "Equity + Drawdown")
)

# --- Top: Candlestick (black up, white down, 1pt border) ---
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles",
        increasing=dict(line=dict(color="black", width=1), fillcolor="black"),
        decreasing=dict(line=dict(color="black", width=1), fillcolor="white")
    ),
    row=1, col=1
)

# --- SMAs with custom colors ---
fig.add_trace(go.Scatter(
    x=df.index, y=df['close'].rolling(FAST).mean(),
    mode="lines", name=f"SMA {FAST}",
    line=dict(color="cyan")
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df['close'].rolling(SLOW).mean(),
    mode="lines", name=f"SMA {SLOW}",
    line=dict(color="magenta")
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df.index, y=df['close'].rolling(100).mean(),
    mode="lines", name="SMA 100",
    line=dict(color="gold", dash="dot")
), row=1, col=1)

# --- Bottom: Equity + Drawdown ---
eq_curve = engine.equity_curve.copy()
dd_curve = engine.drawdown_series.copy()

fig.add_trace(go.Scatter(
    x=eq_curve.index, y=eq_curve,
    mode="lines", name="Equity Curve",
    line=dict(color="blue")
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=eq_curve.index, y=eq_curve - dd_curve,
    mode="lines", line=dict(width=0), showlegend=False
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=eq_curve.index, y=eq_curve,
    fill="tonexty", mode="lines", line=dict(width=0),
    name="Drawdown", opacity=0.3,
), row=2, col=1)
# --- Highlight trade holding periods ---
for t in engine.trades:
    if t.exit_time is None:   # skip still-open trades
        continue
    color = "rgba(0, 200, 0, 0.5)" if t.side == 1 else "rgba(200, 0, 0, 0.5)"
    fig.add_vrect(
        x0=t.entry_time, x1=t.exit_time,
        fillcolor=color, opacity=0.5, line_width=0,
        row=1, col=1
    )
# --- Layout ---
fig.update_layout(
    xaxis_rangeslider_visible=False,
    autosize = True,
    height=700,
    template="plotly_white"
)

fig.show()


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Indicators ---
close = df["close"]
sma_fast = close.rolling(FAST).mean()
sma_slow = close.rolling(SLOW).mean()
sma_trend = close.rolling(200).mean()
sma_100 = close.rolling(100).mean()

# RSI (14)
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# ATR (14)
high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

# --- Subplots: 3 rows ---
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Price + SMAs", "RSI (14)", "ATR (14)")
)

# --- Row 1: Candles + SMAs ---
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'],
    low=df['low'], close=df['close'],
    name="Candles",
    increasing=dict(line=dict(color="black", width=1), fillcolor="black"),
    decreasing=dict(line=dict(color="black", width=1), fillcolor="white")
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=sma_fast, mode="lines",
                         name=f"SMA {FAST}", line=dict(color="cyan")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_slow, mode="lines",
                         name=f"SMA {SLOW}", line=dict(color="magenta")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_trend, mode="lines",
                         name="SMA 200", line=dict(color="gold", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_100, mode="lines",
                         name="SMA 100", line=dict(color="orange", dash="dash")), row=1, col=1)

# Highlight trades (same logic as before)
for t in engine.trades:
    if t.exit_time is None:
        continue
    color = "rgba(0,200,0,0.5)" if t.side == 1 else "rgba(200,0,0,0.5)"
    fig.add_vrect(
        x0=t.entry_time, x1=t.exit_time,
        fillcolor=color, opacity=0.5, line_width=0,
        row=1, col=1
    )

# --- Row 2: RSI ---
fig.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines",
                         name="RSI 14", line=dict(color="blue")), row=2, col=1)
# Add overbought/oversold bands
fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=2, col=1)

# --- Row 3: ATR ---
fig.add_trace(go.Scatter(x=df.index, y=atr, mode="lines",
                         name="ATR 14", line=dict(color="purple")), row=3, col=1)

# --- Layout ---
fig.update_layout(
    xaxis_rangeslider_visible=False,
    autosize = True,
    height=1000,
    template="plotly_white"
)

fig.show()

# %%
# Count trades by exit reason
trade_log["Exit Reason"].value_counts()

# Percentage breakdown
trade_log["Exit Reason"].value_counts(normalize=True) * 100

# Average PnL by exit reason
trade_log.groupby("Exit Reason")["PnL"].mean()

# Median PnL by exit reason
trade_log.groupby("Exit Reason")["PnL"].median()

# Distribution (count + mean + win rate) by exit reason
summary = trade_log.groupby("Exit Reason").agg(
    Trades=("PnL", "count"),
    AvgPnL=("PnL", "mean"),
    MedianPnL=("PnL", "median"),
    WinRate=("Win", "mean")
)
print(summary)

# %%
