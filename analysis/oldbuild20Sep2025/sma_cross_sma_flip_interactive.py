#%% Imports & path (adjust/remove sys.path if your package is installed)
from datetime import datetime
import os, sys
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt

# point to your dev repo only if needed
sys.path.append(os.path.abspath("D:/coding/mt5fin/dev"))

# package imports
from quant_tools import fetch_fx, CurrencyPair, Fees, plot_heatmap

# Focus (full realism)
from quant_tools.backtest import BacktestEngine as BacktestEngineFocus
from quant_tools.backtest import filter_strategies
from quant_tools.exits import FixedStop as FixedStopFocus, TrailingStop as TrailingStopFocus, BreakEvenStepStop as BreakEvenStepStopFocus

# Sweep (fast realism)
from quant_tools.backtest_fast import BacktestEngine as BacktestEngineSweep
# Sweep (fast realism)
from quant_tools.backtest_fast import BacktestEngine as BacktestEngineSweep
from quant_tools.exits_fast import (
    FixedStop as FixedStopSweep,
    TrailingStop as TrailingStopSweep,
    BreakEvenStepStop as BreakEvenStepStopSweep,
)


#%% Strategy-local entry rule (keep strategy logic here so you can tweak freely)
class SMAFlipEntry:
    """Flip long/short on fast/slow SMA cross (precompute SMA once, no look-ahead)."""
    def __init__(self, df: pd.DataFrame, fast: int, slow: int,
                 strategy="SMA", risk_pct=1.0, sl_pips=50):
        self.strategy = strategy
        self.risk_pct = float(risk_pct)
        self.sl_pips  = int(sl_pips)
        self.fast = int(fast)
        self.slow = int(slow)

        price = df['close']
        self.fast_sma = price.rolling(self.fast).mean()
        self.slow_sma = price.rolling(self.slow).mean()
        self.index = df.index

    def check(self, ts, price, df):
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


#%% Data & IO config (edit directly)
SYMBOL = "USD/JPY"
START  = datetime(2025, 1, 1)
END    = datetime(2025, 3, 31)
INITIAL_CAPITAL = 1000.0

OUTDIR = None          # e.g., "runs/jan_interactive"; set to None to skip saving
SAVE_CSV = False
SAVE_PLOTS = False

pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)
df = fetch_fx(pair.symbol, start=START, end=END)
df.head()


#%% ---------- FOCUS (full realism) ----------
# Edit these four params directly
FAST = 5
SLOW = 14
SL_PIPS = 30
RISK_PCT = 1

# Realism knobs
SPREAD_PIPS   = 0.5
SLIPPAGE_PIPS = 0.1

entry_focus = SMAFlipEntry(df, FAST, SLOW, strategy=f"SMA_{FAST}_{SLOW}",
                           risk_pct=RISK_PCT, sl_pips=SL_PIPS)
exit_rules_focus = [
    FixedStopFocus(),
    TrailingStopFocus(trail_pips=30,  pip_size=pair.pip_size),
    BreakEvenStepStopFocus(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                           commission=fees.commission, overnight_fee=fees.overnight_fee),
]

engine_focus = BacktestEngineFocus(
    df, pair, fees,
    max_active_trades=1,
    exit_first=True,
    entry_on_next_bar=True,   # trade signals next bar
    entry_fill="open",        # fill at next bar's open
    spread_pips=SPREAD_PIPS,
    slippage_pips=SLIPPAGE_PIPS,
)

engine_focus.run(initial_capital=INITIAL_CAPITAL,
                 entry_rules=[entry_focus],
                 exit_rules=exit_rules_focus)

# Trade log (display)
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
} for t in engine_focus.trades])
trade_log
#%%
# Save outputs (optional)
if OUTDIR:
    os.makedirs(OUTDIR, exist_ok=True)
    if SAVE_CSV:
        trade_log.to_csv(os.path.join(OUTDIR, f"trades_focus_{FAST}_{SLOW}_{SL_PIPS}_{RISK_PCT}_full.csv"), index=False)
#%%
# Equity plot
plt.figure(figsize=(14, 6))
plt.plot(engine_focus.equity_curve, label="Equity Curve")
plt.fill_between(engine_focus.drawdown_series.index,
                 engine_focus.equity_curve - engine_focus.drawdown_series,
                 engine_focus.equity_curve,
                 alpha=0.3, label="Drawdown")
plt.title(f"Equity (SMA {FAST}/{SLOW}, SL={SL_PIPS}, Risk={RISK_PCT}%, full)")
plt.xlabel("Time"); plt.ylabel("Equity"); plt.legend(); plt.grid(True)
if OUTDIR and SAVE_PLOTS:
    plt.savefig(os.path.join(OUTDIR, f"equity_focus_{FAST}_{SLOW}_{SL_PIPS}_{RISK_PCT}_full.png"),
                dpi=160, bbox_inches="tight")
plt.show()


#%% ---------- SWEEP (fast realism with ALL exits) ----------
# Close-only exits (fast), next-bar entries, no spread/slippage.
# Mirrors full logic but runs quicker. Narrow grids if runtime is high.

FAST_RANGE = range(5, 31, 1)
SLOW_RANGE = range(6, 55, 1)
SL_LIST    = [20, 30, 50, 70]
RISK_LIST  = [1, 2, 5, 10]

rows = []
for f, s, slp, r in it.product(FAST_RANGE, SLOW_RANGE, SL_LIST, RISK_LIST):
    if s <= f:
        continue

    entry = SMAFlipEntry(df, f, s, strategy=f"SMA_{f}_{s}",
                         risk_pct=r, sl_pips=slp)

    exit_rules_fast = [
        FixedStopSweep(sl_pips=slp, pip_size=pair.pip_size),
        TrailingStopSweep(trail_pips=30,  pip_size=pair.pip_size),
        BreakEvenStepStopSweep(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                               commission=fees.commission, overnight_fee=fees.overnight_fee),
    ]

    engine_sweep = BacktestEngineSweep(
        df, pair, fees,
        max_active_trades=1,
        entry_on_next_bar=True,
        entry_fill="open",
    )

    engine_sweep.run(initial_capital=INITIAL_CAPITAL,
                     entry_rules=[entry],
                     exit_rules=exit_rules_fast)

    rows.append({
        "fast_sma": int(f), "slow_sma": int(s),
        "sl_pips": int(slp), "risk_pct": float(r),
        "num_trades": engine_sweep.metrics['num_trades'],
        "cumulative_pnl": engine_sweep.metrics['cumulative_pnl'],
        "max_drawdown": engine_sweep.metrics['max_drawdown'],
        "win_rate": engine_sweep.metrics['win_rate'],
        "sharpe": engine_sweep.metrics['sharpe'],
        "cagr": engine_sweep.metrics['cagr'],
        "total_fees": engine_sweep.metrics['total_fees']
    })

df_results = pd.DataFrame(rows).sort_values("cumulative_pnl", ascending=False)
df_results.head(12)

#%%
filter_strategies(df_results, min_pnl_pct=5, max_dd_pct=10, initial_capital=1000)
#%% Heatmaps (saved to file if OUTDIR set)
# Slice choices for visualization (tweak as you like)
fixed_risk = 5 #float(sorted(RISK_LIST)[0])           # e.g., smallest risk
fixed_sl   = 20 #int(sorted(SL_LIST)[len(SL_LIST)//2]) # e.g., median SL

slice_fs = df_results[(df_results["risk_pct"] == fixed_risk) & (df_results["sl_pips"] == fixed_sl)]
if not slice_fs.empty:
    plot_heatmap(slice_fs, x_col="fast_sma", y_col="slow_sma", value_col="cumulative_pnl")
    if OUTDIR and SAVE_PLOTS:
        plt.savefig(os.path.join(OUTDIR, f"heatmap_pnl_fast_slow_risk{fixed_risk}_sl{fixed_sl}_fast.png"),
                    dpi=160, bbox_inches="tight")
    plt.show()

# Another slice: Sharpe over (sl, risk) at a fixed (fast, slow)
fixed_fast = int(sorted(FAST_RANGE)[0])
fixed_slow_candidates = [x for x in SLOW_RANGE if x > fixed_fast]
fixed_slow = int(sorted(fixed_slow_candidates)[0]) if fixed_slow_candidates else int(sorted(SLOW_RANGE)[0])
slice_sr = df_results[(df_results["fast_sma"] == fixed_fast) & (df_results["slow_sma"] == fixed_slow)]
if not slice_sr.empty:
    plot_heatmap(slice_sr, x_col="sl_pips", y_col="risk_pct", value_col="sharpe")
    if OUTDIR and SAVE_PLOTS:
        plt.savefig(os.path.join(OUTDIR, f"heatmap_sharpe_sl_risk_fast{fixed_fast}_slow{fixed_slow}_fast.png"),
                    dpi=160, bbox_inches="tight")
    plt.show()

# %%
#%% ---------- SWEEP (full realism) ----------
# Uses full engine + realistic exits (OHLC touch, trailing, BE) + spread & slippage.
# Tip: narrow ranges first to keep it reasonable.
FAST_RANGE = range(5, 31, 5)
SLOW_RANGE = range(6, 55, 5)
SL_LIST    = [20, 30, 50, 70]
RISK_LIST  = [1, 2, 5, 10]
rows_full = []
for f, s, slp, r in it.product(FAST_RANGE, SLOW_RANGE, SL_LIST, RISK_LIST):
    print(f"Running full sweep for Fast={f}, Slow={s}, SL={slp}, Risk={r} ...")
    if s <= f:
        continue
    else:
        entry_full = SMAFlipEntry(df, f, s, strategy=f"SMA_{f}_{s}",
                                risk_pct=r, sl_pips=slp)
        exit_rules_full = [
            FixedStopFocus(),  # reads trade.sl_price set by RiskManager
            TrailingStopFocus(trail_pips=30,  pip_size=pair.pip_size),
            BreakEvenStepStopFocus(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                                commission=fees.commission, overnight_fee=fees.overnight_fee),
        ]
        engine_full = BacktestEngineFocus(
            df, pair, fees,
            max_active_trades=1,
            exit_first=True,
            entry_on_next_bar=True,
            entry_fill="open",
            spread_pips=SPREAD_PIPS,
            slippage_pips=SLIPPAGE_PIPS,
        )
        engine_full.run(initial_capital=INITIAL_CAPITAL,
                        entry_rules=[entry_full],
                        exit_rules=exit_rules_full)

        rows_full.append({
            "fast_sma": int(f), "slow_sma": int(s),
            "sl_pips": int(slp), "risk_pct": float(r),
            "num_trades": engine_full.metrics['num_trades'],
            "cumulative_pnl": engine_full.metrics['cumulative_pnl'],
            "max_drawdown": engine_full.metrics['max_drawdown'],
            "win_rate": engine_full.metrics['win_rate'],
            "sharpe": engine_full.metrics['sharpe'],
            "cagr": engine_full.metrics['cagr'],
            "total_fees": engine_full.metrics['total_fees']
        })

df_results_full = pd.DataFrame(rows_full).sort_values("cumulative_pnl", ascending=False)
df_results_full.head(12)

# Save results (optional)
if OUTDIR and SAVE_CSV:
    os.makedirs(OUTDIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_results_full.to_csv(os.path.join(OUTDIR, f"sweep_results_full_{ts}.csv"), index=False)

# Heatmaps (saved if OUTDIR+SAVE_PLOTS)
fixed_risk_full = float(sorted(RISK_LIST)[0])
fixed_sl_full   = int(sorted(SL_LIST)[len(SL_LIST)//2])

slice_fs_full = df_results_full[
    (df_results_full["risk_pct"] == fixed_risk_full) &
    (df_results_full["sl_pips"] == fixed_sl_full)
]
if not slice_fs_full.empty:
    plot_heatmap(slice_fs_full, x_col="fast_sma", y_col="slow_sma", value_col="cumulative_pnl")
    if OUTDIR and SAVE_PLOTS:
        plt.savefig(os.path.join(
            OUTDIR, f"heatmapFULL_pnl_fast_slow_risk{fixed_risk_full}_sl{fixed_sl_full}.png"
        ), dpi=160, bbox_inches="tight")
    plt.show()

fixed_fast_full = int(sorted(FAST_RANGE)[0])
cand = [x for x in SLOW_RANGE if x > fixed_fast_full]
fixed_slow_full = int(sorted(cand)[0]) if cand else int(sorted(SLOW_RANGE)[0])
slice_sr_full = df_results_full[
    (df_results_full["fast_sma"] == fixed_fast_full) &
    (df_results_full["slow_sma"] == fixed_slow_full)
]
if not slice_sr_full.empty:
    plot_heatmap(slice_sr_full, x_col="sl_pips", y_col="risk_pct", value_col="sharpe")
    if OUTDIR and SAVE_PLOTS:
        plt.savefig(os.path.join(
            OUTDIR, f"heatmapFULL_sharpe_sl_risk_fast{fixed_fast_full}_slow{fixed_slow_full}.png"
        ), dpi=160, bbox_inches="tight")
    plt.show()

# %%
#%% ---------- SWEEP (full realism) — ThreadPoolExecutor ----------
import itertools as it
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import math

FAST_RANGE = range(5, 31, 5)
SLOW_RANGE = range(6, 55, 5)
SL_LIST    = [20, 30, 50, 70]
RISK_LIST  = [1, 2, 5, 10]

def _full_task(params):
    f, s, slp, r = params
    if s <= f:
        return None
    entry_full = SMAFlipEntry(df, f, s, strategy=f"SMA_{f}_{s}",
                              risk_pct=r, sl_pips=slp)
    exit_rules_full = [
        FixedStopFocus(),  # reads trade.sl_price set by RiskManager
        TrailingStopFocus(trail_pips=30,  pip_size=pair.pip_size),
        BreakEvenStepStopFocus(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                               commission=fees.commission, overnight_fee=fees.overnight_fee),
    ]
    engine_full = BacktestEngineFocus(
        df, pair, fees,
        max_active_trades=1,
        exit_first=True,
        entry_on_next_bar=True,
        entry_fill="open",
        spread_pips=SPREAD_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
    )
    engine_full.run(initial_capital=INITIAL_CAPITAL,
                    entry_rules=[entry_full],
                    exit_rules=exit_rules_full)
    #print(f"Completed full sweep for Fast={f}, Slow={s}, SL={slp}, Risk={r}")
    return {
        "fast_sma": int(f), "slow_sma": int(s),
        "sl_pips": int(slp), "risk_pct": float(r),
        "num_trades": engine_full.metrics['num_trades'],
        "cumulative_pnl": engine_full.metrics['cumulative_pnl'],
        "max_drawdown": engine_full.metrics['max_drawdown'],
        "win_rate": engine_full.metrics['win_rate'],
        "sharpe": engine_full.metrics['sharpe'],
        "cagr": engine_full.metrics['cagr'],
        "total_fees": engine_full.metrics['total_fees']
    }

params = list(it.product(FAST_RANGE, SLOW_RANGE, SL_LIST, RISK_LIST))
rows_full = []
max_workers = 12  # tweak if you like

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futs = [ex.submit(_full_task, p) for p in params]
    for i, fut in enumerate(as_completed(futs), 1):
        res = fut.result()
        if res:
            rows_full.append(res)
        if i % 10 == 0:
            print(f"done {i}/{len(futs)}")

df_results_full = pd.DataFrame(rows_full).sort_values("cumulative_pnl", ascending=False)
df_results_full.head(12)
# %%
filter_strategies(df_results_full, min_pnl_pct=5, max_dd_pct=15, initial_capital=INITIAL_CAPITAL)