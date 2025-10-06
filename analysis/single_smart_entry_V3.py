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
SL_PIPS, TP_PIPS, RISK_PCT = 30, 30, 1.0
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

rsi_period = 14
atr_period = 14

# RSI Wilder
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
alpha = 1 / rsi_period
avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
df["RSI"] = rsi
# ATR Wilder
high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
df["ATR"] = atr


# Volume (tick volume from Dukascopy)
volume = df["volume"] if "volume" in df.columns else None

# --- Entry rule (same as focus_sma) ---

# smart_entry.py
import pandas as pd
import numpy as np

class SmartEntryV3:
    """
    4-Tier entry system with ATR filter, RSI crossover/level differentiation,
    wait-upgrade logic, and full debug logging.

    Tiers:
      None     = Price breakout only (wait)
      Low      = Breakout + RSI crossover from OB/OS
      Medium   = Breakout + RSI level confirmation
      High     = Breakout + RSI + strong ATR
      Perfect  = Breakout + RSI + high ATR
    """

    def __init__(self, df,
                 fast=14, slow=30,
                 rsi_period=14, atr_period=14,
                 rsi_upper=55, rsi_lower=45,
                 atr_mult_strong=1.0, atr_mult_high=1.5,
                 atr_filter=0.8,
                 wait_bars=5,
                 sl_pips=50,
                 risk_low=0.5, risk_med=1.0, risk_high=1.5, risk_perf=2.0,
                 strategy="SmartEntryV3"):
        self.df = df
        self.index = df.index
        self.fast_sma = df["close"].rolling(fast).mean()
        self.slow_sma = df["close"].rolling(slow).mean()
        self.strategy = strategy
        self.sl_pips = sl_pips
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.atr_mult_strong = atr_mult_strong
        self.atr_mult_high = atr_mult_high
        self.atr_filter = atr_filter
        self.wait_bars = wait_bars

        self.pending = None  # holds pending signal info
        self.debug_mode = True
        self.debug_records = []

        self.risk_map = {
            "Low": risk_low,
            "Medium": risk_med,
            "High": risk_high,
            "Perfect": risk_perf,
        }

        # --- RSI ---
        # --- RSI (Wilder's smoothing, consistent with chart) ---
        if "RSI" in df.columns:
            self.rsi = df["RSI"]
        else:
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            alpha = 1 / rsi_period  # Wilder smoothing factor
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            rs = avg_gain / avg_loss
            self.rsi = 100 - (100 / (1 + rs))


        # --- ATR ---
        # --- ATR (Wilder's smoothing, consistent with chart) ---
        if "ATR" in df.columns:
            self.atr = df["ATR"]
        else:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
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
        rsi_prev = self.rsi.iloc[i - 1]
        atr_now = self.atr.iloc[i]

        atr_ratio = atr_now / self.atr_mean if not np.isnan(atr_now) else 0
        atr_strong = atr_ratio >= self.atr_mult_strong
        atr_high = atr_ratio >= self.atr_mult_high
        atr_gate = atr_now >= self.atr_mean * self.atr_filter

        # --- helper for consistent debug logging ---
        def log_debug(event_tier=None, event_side=None, comment=None):
            if not self.debug_mode:
                return
            self.debug_records.append({
                "time": ts,
                "close": float(close_now),
                "fast_sma": float(f_now),
                "slow_sma": float(s_now),
                "rsi_now": float(rsi_now),
                "rsi_prev": float(rsi_prev),
                "atr_now": float(atr_now),
                "atr_ratio": float(atr_ratio),
                "atr_gate": atr_gate,
                "rsi_cross_up": bool(rsi_prev < 30 and rsi_now >= 30),
                "rsi_cross_down": bool(rsi_prev > 70 and rsi_now <= 70),
                "rsi_level_long": bool(rsi_now >= self.rsi_upper),
                "rsi_level_short": bool(rsi_now <= self.rsi_lower),
                "long_breakout": bool((close_prev <= f_prev <= s_prev) and (close_now > f_now and close_now > s_now)),
                "short_breakout": bool((close_prev >= f_prev >= s_prev) and (close_now < f_now and close_now < s_now)),
                "pending_state": None if not self.pending else self.pending["type"],
                "triggered_tier": event_tier,
                "side": event_side,
                "comment": comment,
            })

        # --- ATR gating (skip low volatility) ---
        if np.isnan(atr_now) or not atr_gate:
            log_debug(None, None, "ATR too low, skip")
            return False, {"filters": {"atr": "too_low"}}

        long_breakout = (close_prev <= f_prev <= s_prev) and (close_now > f_now and close_now > s_now)
        short_breakout = (close_prev >= f_prev >= s_prev) and (close_now < f_now and close_now < s_now)

        rsi_cross_up = rsi_prev < 30 and rsi_now >= 30
        rsi_cross_down = rsi_prev > 70 and rsi_now <= 70
        rsi_level_long = rsi_now >= self.rsi_upper
        rsi_level_short = rsi_now <= self.rsi_lower

        def build(side, tier, comment):
            log_debug(tier, side, comment)
            return True, {
                "side": side,
                "strategy": self.strategy,
                "confidence": tier,
                "rsi_now": float(rsi_now),
                "atr_now": float(atr_now),
                "atr_ratio": float(atr_ratio),
                "risk_pct": self.risk_map.get(tier, 1.0),
                "sl_pips": self.sl_pips,
            }

        # --- handle pending wait signals ---
        if self.pending:
            side = self.pending["side"]
            if i - self.pending["bar"] > self.wait_bars:
                log_debug(None, side, "Pending expired (RSI never confirmed)")
                self.pending = None
            else:
                if side == +1 and (rsi_cross_up or rsi_level_long):
                    tier = "Low" if rsi_cross_up else "Medium"
                    if atr_high:
                        tier = "Perfect"
                    elif atr_strong and tier != "Low":
                        tier = "High"
                    self.pending = None
                    return build(+1, tier, "Pending upgraded via RSI cross/level")

                elif side == -1 and (rsi_cross_down or rsi_level_short):
                    tier = "Low" if rsi_cross_down else "Medium"
                    if atr_high:
                        tier = "Perfect"
                    elif atr_strong and tier != "Low":
                        tier = "High"
                    self.pending = None
                    return build(-1, tier, "Pending upgraded via RSI cross/level")

        # --- LONG direction ---
        if long_breakout:
            if rsi_cross_up:
                return build(+1, "Low", "Immediate entry: RSI cross up")
            elif rsi_level_long:
                if atr_high:
                    return build(+1, "Perfect", "Immediate entry: RSI level + high ATR")
                elif atr_strong:
                    return build(+1, "High", "Immediate entry: RSI level + strong ATR")
                else:
                    return build(+1, "Medium", "Immediate entry: RSI level")
            else:
                self.pending = {"side": +1, "bar": i, "type": "wait_rsi"}
                log_debug(None, +1, "Breakout: waiting for RSI confirm")
                return False, {"pending": "Long_Breakout"}

        # --- SHORT direction ---
        if short_breakout:
            if rsi_cross_down:
                return build(-1, "Low", "Immediate entry: RSI cross down")
            elif rsi_level_short:
                if atr_high:
                    return build(-1, "Perfect", "Immediate entry: RSI level + high ATR")
                elif atr_strong:
                    return build(-1, "High", "Immediate entry: RSI level + strong ATR")
                else:
                    return build(-1, "Medium", "Immediate entry: RSI level")
            else:
                self.pending = {"side": -1, "bar": i, "type": "wait_rsi"}
                log_debug(None, -1, "Breakout: waiting for RSI confirm")
                return False, {"pending": "Short_Breakout"}

        # --- no entry triggered ---
        log_debug(None, None, "No entry conditions met")
        return False, {}





# --- Engine setup ---
engine = BacktestEngine(df, pair, fees,
                        max_active_trades=1,
                        spread_pips=0.5,
                        slippage_pips=0.1)

entry_rule = SmartEntryV3(df, FAST, SLOW, MID,
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
debug_recs=pd.DataFrame(entry_rule.debug_records)
# %%
debug_recs
# %%
