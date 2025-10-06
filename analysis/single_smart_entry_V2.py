#%% focus_sma_plot_debug.py
from datetime import datetime
import os, sys
import pandas as pd
import pytz
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Dev path ---
sys.path.append(os.path.abspath("D:/coding/quant/dev"))

# Package imports
from quant_tools import fetch_fx, CurrencyPair, Fees
from quant_tools.backtest import BacktestEngine
from quant_tools.exits import FixedStop, BreakEvenStepStop, TrailingStop, ProgressiveStop
#%%
# --- Config ---
SYMBOL = "USD/JPY"
BANGKOK = pytz.timezone("Asia/Bangkok")
START = BANGKOK.localize(datetime(2025, 1, 1))
END   = BANGKOK.localize(datetime(2025, 1, 5))
INITIAL_CAPITAL = 1000.0

FAST, SLOW, MID = 7, 14, 100
SL_PIPS, TP_PIPS, RISK_PCT = 10, 30, 1.0
WARMUP_BARS = 250

BAR_INTERVAL_MIN = 5
bars_per_day = int((24 * 60) / BAR_INTERVAL_MIN)
offset_days = (WARMUP_BARS // bars_per_day) + 1
DATA_START = START - pd.Timedelta(days=offset_days)

# --- Data fetch ---
import dukascopy_python
pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)
df = fetch_fx(pair.symbol, start=DATA_START, end=END, interval=dukascopy_python.INTERVAL_MIN_5)
# --- Indicators ---
close = df["close"]
sma_fast = close.rolling(FAST).mean()
sma_slow = close.rolling(SLOW).mean()
sma_mid  = close.rolling(MID).mean()
sma_200  = close.rolling(200).mean()

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

# Volume (tick volume from Dukascopy)
volume = df["volume"] if "volume" in df.columns else None

# --- Entry rule (same as focus_sma) ---

# smart_entry.py
import pandas as pd
import numpy as np

class SmartEntryV2:
    """
    Multi-phase adaptive entry logic.
    Modes:
      - 'simultaneous': all conditions align in same bar
      - 'wait': allow RSI confirmation within wait_bars after breakout
      - 'hybrid': both behaviors coexist
    Confidence tiers:
      Low     = price breakout only
      Medium  = + RSI or SMA alignment
      High    = + both RSI & SMA alignment
      Perfect = + ATR breakout confirmation
    """

    def __init__(self, df,
                 fast=14, slow=30, mid=100,
                 rsi_period=14, atr_period=14,
                 rsi_upper=55, rsi_lower=45,
                 atr_mult=1.2, wait_bars=3,
                 sl_pips=50,
                 risk_low=0.5, risk_med=1.0, risk_high=1.5, risk_perf=2.0,
                 mode="hybrid",
                 strategy="SmartEntryV2"):
        self.df = df
        self.index = df.index
        self.fast_sma = df["close"].rolling(fast).mean()
        self.slow_sma = df["close"].rolling(slow).mean()
        self.mid_sma  = df["close"].rolling(mid).mean()
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.atr_mult = atr_mult
        self.wait_bars = wait_bars
        self.sl_pips = sl_pips
        self.risk_map = {
            "Low": risk_low,
            "Medium": risk_med,
            "High": risk_high,
            "Perfect": risk_perf,
        }
        self.mode = mode
        self.strategy = strategy
        self.pending = None  # used in wait mode

        # --- RSI ---
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        self.rsi = 100 - (100 / (1 + rs))

        # --- ATR ---
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.atr = tr.rolling(atr_period).mean()
        self.atr_mean = self.atr.mean()

    # ----------------------------------------------------------------
    def check(self, ts, price, df, state=None):
        i = self.index.get_loc(ts)
        if i < 2:
            return False, {}

        close_now = df["close"].iloc[i]
        close_prev = df["close"].iloc[i - 1]
        f_now, s_now = self.fast_sma.iloc[i], self.slow_sma.iloc[i]
        f_prev, s_prev = self.fast_sma.iloc[i - 1], self.slow_sma.iloc[i - 1]
        rsi_now = self.rsi.iloc[i]
        atr_now = self.atr.iloc[i]
        atr_pass = not np.isnan(atr_now) and atr_now >= self.atr_mean * self.atr_mult
        atr_low = atr_now < self.atr_mean * 0.8

        # --- ATR gating (skip dead sessions) ---
        if atr_low:
            return False, {"filters": {"atr": "too_low"}}

        # --- Directional triggers ---
        long_cross  = (close_prev <= f_prev <= s_prev) and (close_now > f_now and close_now > s_now)
        short_cross = (close_prev >= f_prev >= s_prev) and (close_now < f_now and close_now < s_now)
        sma_align_long  = f_now > s_now
        sma_align_short = f_now < s_now
        trend_up  = self.mid_sma.iloc[i] > self.mid_sma.iloc[i - 1]
        trend_down = self.mid_sma.iloc[i] < self.mid_sma.iloc[i - 1]

        # --- Wait mode pending logic ---
        if self.mode in ["wait", "hybrid"] and self.pending:
            side = self.pending["side"]
            if i - self.pending["bar"] > self.wait_bars:
                self.pending = None  # expired
            else:
                if (side == +1 and rsi_now >= self.rsi_upper) or (side == -1 and rsi_now <= self.rsi_lower):
                    conf = "Medium"  # breakout confirmed later
                    if atr_pass:
                        conf = "High"
                    return True, {
                        "side": side,
                        "strategy": self.strategy,
                        "confidence": conf,
                        "rsi_now": float(rsi_now),
                        "atr_now": float(atr_now),
                        "risk_pct": self.risk_map[conf],
                        "sl_pips": self.sl_pips,
                    }

        # --- Primary signal evaluation ---
        def build(side, conf):
            risk = self.risk_map.get(conf, 1.0)
            return True, {
                "side": side,
                "strategy": self.strategy,
                "confidence": conf,
                "rsi_now": float(rsi_now),
                "atr_now": float(atr_now),
                "risk_pct": risk,
                "sl_pips": self.sl_pips,
            }

        # --- Simultaneous or hybrid mode immediate evaluation ---
        if long_cross and sma_align_long and rsi_now >= self.rsi_upper:
            conf = "High" if not atr_pass else "Perfect"
            return build(+1, conf)
        if short_cross and sma_align_short and rsi_now <= self.rsi_lower:
            conf = "High" if not atr_pass else "Perfect"
            return build(-1, conf)

        # --- Price breakout only (wait for RSI or SMA) ---
        if long_cross and self.mode in ["wait", "hybrid"]:
            self.pending = {"side": +1, "bar": i}
            return False, {"pending": "Long_Breakout"}
        if short_cross and self.mode in ["wait", "hybrid"]:
            self.pending = {"side": -1, "bar": i}
            return False, {"pending": "Short_Breakout"}

        # --- SMA cross + RSI confirm (trend mode) ---
        sma_long = (f_now > s_now and f_prev <= s_prev and rsi_now >= self.rsi_upper)
        sma_short = (f_now < s_now and f_prev >= s_prev and rsi_now <= self.rsi_lower)
        if sma_long and trend_up:
            conf = "Medium" if not atr_pass else "High"
            return build(+1, conf)
        if sma_short and trend_down:
            conf = "Medium" if not atr_pass else "High"
            return build(-1, conf)

        return False, {}



# --- Engine setup ---
engine = BacktestEngine(df, pair, fees,
                        max_active_trades=1,
                        spread_pips=0.5,
                        slippage_pips=0.1)

entry_rule = SmartEntryV2(df, FAST, SLOW, MID,
                             sl_pips=SL_PIPS,
                             )

exit_rules = [
    #FixedStop(sl_pips=30, pip_size=pair.pip_size),
    #BreakEvenStepStop(step_pips=30, pip_size=pair.pip_size,),
    #TrailingStop(trail_pips=30, pip_size=pair.pip_size),
    ProgressiveStop(sl_pips=20, trail_pips=30, pip_size=pair.pip_size)
]

# --- Run with debug_mode ---
df_debug, trades = engine.run(initial_capital=INITIAL_CAPITAL,
                              entry_rules=[entry_rule],
                              exit_rules=exit_rules,
                              debug_mode=True)

print("Closed Trades:", trades)
#%%
# --- Visualization ---
# --- Subplots: 4 rows ---
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.1, 0.1, 0.1],
    subplot_titles=("Price + SMAs", "RSI (14)", "ATR (14)", "Volume")
)

# Candlesticks
fig.add_trace(go.Candlestick(
    x=df_debug["time"], open=df_debug["open"], high=df_debug["high"],
    low=df_debug["low"], close=df_debug["close"], name="Candles",increasing=dict(line=dict(color="black", width=1), fillcolor="black"),
        decreasing=dict(line=dict(color="black", width=1), fillcolor="white")
), row=1, col=1)

# Stop-loss line
if "sl_price" in df_debug.columns:
    fig.add_trace(go.Scatter(
        x=df_debug["time"], y=df_debug["sl_price"],
        mode="lines", line=dict(color="orange", dash="dot"),
        name="SL (active)"
    ), row=1, col=1)
color_map = {
    "FixedStop": "gray",
    "BreakEvenStepStop": "blue",
    "TrailingStop": "orange",
    "ProgressiveStop": "purple",
}

style_map = {
    "FixedStop": "solid",
    "BreakEvenStepStop": "dash",
    "TrailingStop": "dot",
    "ProgressiveStop": "dashdot",
}
skip_cols = {"time", "open", "high", "low", "close", "equity", "drawdown",
             "open_trades", "pending", "exit_reason", "sl_price"}

for col in df_debug.columns:
    if col not in skip_cols and col in color_map:
        fig.add_trace(go.Scatter(
            x=df_debug["time"], y=df_debug[col],
            mode="lines",
            line=dict(color=color_map[col], dash=style_map[col]),
            name=col
        ), row=1, col=1)
# Entry/Exit markers

fig.add_trace(go.Scatter(x=df.index, y=sma_fast, mode="lines", name=f"SMA {FAST}", line=dict(color="cyan")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_slow, mode="lines", name=f"SMA {SLOW}", line=dict(color="magenta")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_mid,  mode="lines", name=f"SMA {MID}", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_200,  mode="lines", name="SMA 200", line=dict(color="gold")), row=1, col=1)
# Entry/Exit markers
for t in trades:
    if t.side == 1:  # long
        fig.add_trace(go.Scatter(
            x=[t.entry_time], y=[t.entry_price],
            mode="markers+text", text=["Long Entry"],
            textposition="bottom center",
            marker=dict(color="green", size=12, symbol="triangle-up"),
            name="Long Entry"
        ), row=1, col=1)
    elif t.side == -1:  # short
        fig.add_trace(go.Scatter(
            x=[t.entry_time], y=[t.entry_price],
            mode="markers+text", text=["Short Entry"],
            textposition="top center",
            marker=dict(color="orange", size=12, symbol="triangle-down"),
            name="Short Entry"
        ), row=1, col=1)

    # Exit marker (red)
    fig.add_trace(go.Scatter(
        x=[t.exit_time], y=[t.exit_price],
        mode="markers+text", text=[f"Exit ({getattr(t,'exit_reason','?')})"],
        textposition="top center",
        marker=dict(color="red", size=12, symbol="x"),
        name="Exit"
    ), row=1, col=1)
# --- Row 2: RSI ---
fig.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", name="RSI 14", line=dict(color="blue")), row=2, col=1)
fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=2, col=1)
fig.update_yaxes(range=[0, 100], row=2, col=1)

# --- Row 3: ATR ---
fig.add_trace(go.Scatter(x=df.index, y=atr, mode="lines", name="ATR 14", line=dict(color="purple")), row=3, col=1)
fig.update_yaxes(rangemode="tozero", row=3, col=1)

# --- Row 4: Volume ---
if volume is not None:
    fig.add_trace(go.Bar(x=df.index, y=volume, name="Volume", marker_color="gray"), row=4, col=1)
    fig.update_yaxes(rangemode="tozero", row=4, col=1)

fig.update_layout(
    title="SMA Debug Backtest",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    height=800
)
fig.show()
# %%
# %%
df_trades = pd.DataFrame([
    {
        "Entry": t.entry_time,
        "Exit": t.exit_time,
        "Side": "Long" if t.side == 1 else "Short",
        "PnL": t.pnl,
        "Win": t.win,
        "RiskPct": t.risk_pct,
        "Confidence": t.meta.get("confidence"),
        "RSI": t.meta.get("rsi_now"),
        "ATR": t.meta.get("atr_now"),
        "TrendUp": t.meta.get("trend_up"),
        "Cancel": t.cancelled,
    }
    for t in trades
])

# %%
df_trades
# %%
