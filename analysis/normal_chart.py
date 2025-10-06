#%%
# chart_sma.py (with broker-day dropdown: 5pm NY → 5pm NY)
from datetime import datetime
import os, sys
import pandas as pd
import pytz
import dukascopy_python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Config ---
SYMBOL = "USD/JPY"
BANGKOK = pytz.timezone("Asia/Bangkok")
START = BANGKOK.localize(datetime(2025, 1, 1))
END   = BANGKOK.localize(datetime(2025, 3, 31))

FAST, SLOW, MID = 14, 30, 100

# --- Fetch Data ---
sys.path.append(os.path.abspath("D:/coding/quant/dev"))
from quant_tools import fetch_fx, CurrencyPair

pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)

# --- Warmup offset ---
WARMUP_BARS = 250
BAR_INTERVAL_MIN = 5
bars_per_day = int((24 * 60) / BAR_INTERVAL_MIN)
offset_days = (WARMUP_BARS // bars_per_day) + 1
DATA_START = START - pd.Timedelta(days=offset_days)

# --- Fetch Data ---
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

# --- Subplots: 4 rows ---
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.1, 0.1, 0.1],
    subplot_titles=("Price + SMAs", "RSI (14)", "ATR (14)", "Volume")
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

fig.add_trace(go.Scatter(x=df.index, y=sma_fast, mode="lines", name=f"SMA {FAST}", line=dict(color="cyan")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_slow, mode="lines", name=f"SMA {SLOW}", line=dict(color="magenta")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_mid,  mode="lines", name=f"SMA {MID}", line=dict(color="orange", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sma_200,  mode="lines", name="SMA 200", line=dict(color="gold", dash="dot")), row=1, col=1)

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

# --- Broker day dropdown (5pm NY → 5pm NY) ---
# Define broker/trading day in Bangkok time (5am → 5am)
df_bkk = df.copy()  # already tz_convert('Asia/Bangkok')
shifted = df_bkk.index - pd.Timedelta(hours=5)  
session_days = shifted.normalize()


def broker_day_bounds(day_ts: pd.Timestamp):
    start = day_ts + pd.Timedelta(hours=5)  
    end   = start + pd.Timedelta(days=1)
    day_slice = df_bkk.loc[(df_bkk.index >= start) & (df_bkk.index < end)]
    if len(day_slice) == 0:
        return start, end, None, None
    lo = float(day_slice["low"].min())
    hi = float(day_slice["high"].max())
    pad = max((hi - lo) * 0.02, (hi + lo) * 0.0001)
    return start, end, lo - pad, hi + pad

buttons = []
# Add "All" button
buttons.append(dict(
    label="All",
    method="relayout",
    args=[{
        "xaxis.autorange": True,
        "yaxis.autorange": True
    }]
))

for d in sorted(session_days.unique()):
    start, end, y_min, y_max = broker_day_bounds(d)
    if y_min is None:
        continue
    buttons.append(dict(
        label=str(d.date()),
        method="relayout",
        args=[{
            "xaxis.range": [start, end],
            "yaxis.autorange": False,
            "yaxis.range": [y_min, y_max]
        }]
    ))

fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        x=0.0, y=1.15,
        xanchor="left", yanchor="top",
        pad={"r": 8, "t": 8}
    )],
    xaxis_rangeslider_visible=False,
    autosize=True,
    height=1200,
    template="plotly_white"
)

fig.show()
#%%
