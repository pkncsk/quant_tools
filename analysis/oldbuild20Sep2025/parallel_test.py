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
from quant_tools.exits import (
    FixedStop as FixedStopFocus,
    TrailingStop as TrailingStopFocus,
    BreakEvenStepStop as BreakEvenStepStopFocus,
)

# Sweep (fast realism)
from quant_tools.backtest_fast import BacktestEngine as BacktestEngineSweep
from quant_tools.exits_fast import (
    FixedStop as FixedStopSweep,
    TrailingStop as TrailingStopSweep,
    BreakEvenStepStop as BreakEvenStepStopSweep,
)

#%% Ray setup with runtime_env
import ray
DEV_PATH = r"D:\coding\mt5fin\dev"  # folder containing the quant_tools package
num_cpus = min(os.cpu_count() or 2, 8)

ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    num_cpus=num_cpus,
    runtime_env={
        "working_dir": DEV_PATH,                 # ship this dir to workers
        "env_vars": {"PYTHONPATH": DEV_PATH},    # ensure imports succeed
    },
)

#%% ---------- Define grid FIRST (so actors can precompute SMAs) ----------
FAST_RANGE = range(5, 31, 1)
SLOW_RANGE = range(6, 32, 1)
SL_LIST    = [20, 30, 50, 70]
RISK_LIST  = [1, 2, 5, 10]
params = [(f, s, slp, r) for f in FAST_RANGE for s in SLOW_RANGE for slp in SL_LIST for r in RISK_LIST]

#%% Ray Actor: Full-realism worker (precomputes SMA map)
@ray.remote
class FullWorker:
    def __init__(self, df_ref, pair_params, fees_params, spread_pips, slippage_pips, initial_capital,
                 fast_vals=None, slow_vals=None):
        from quant_tools import CurrencyPair, Fees
        import ray as _ray
        try:
            from ray import ObjectRef
        except Exception:
            ObjectRef = ()
        # accept either ObjectRef or already-materialized DataFrame
        self.df = _ray.get(df_ref) if isinstance(df_ref, ObjectRef) else df_ref

        self.pair = CurrencyPair(*pair_params)
        self.fees = Fees(*fees_params)
        self.spread = float(spread_pips)
        self.slip = float(slippage_pips)
        self.ic = float(initial_capital)

        # Precompute SMA once per actor (union of windows)
        close = self.df["close"].astype(float)
        winset = sorted({int(x) for x in (fast_vals or [])} | {int(x) for x in (slow_vals or [])})
        self._sma = {w: close.rolling(w).mean() for w in winset}

    def run(self, f, s, slp, r):
        try:
            import pandas as pd
            from quant_tools.backtest import BacktestEngine as Engine
            from quant_tools.exits import FixedStop, TrailingStop, BreakEvenStepStop

            f = int(f); s = int(s); slp = int(slp); r = float(r)
            if s <= f:
                return None

            # Use precomputed SMAs (fallback compute if unseen)
            fast_sma = self._sma.get(f)
            slow_sma = self._sma.get(s)
            if fast_sma is None:
                fast_sma = self.df["close"].rolling(f).mean(); self._sma[f] = fast_sma
            if slow_sma is None:
                slow_sma = self.df["close"].rolling(s).mean(); self._sma[s] = slow_sma
            idx = self.df.index

            class _Entry:
                def __init__(self, risk_pct, sl_pips, name):
                    self.risk_pct = float(risk_pct); self.sl_pips = int(sl_pips); self.strategy = name
                def check(self, ts, price_val, _df):
                    i = idx.get_loc(ts)
                    if i == 0: return (False, {})
                    f_now, s_now = fast_sma.iloc[i], slow_sma.iloc[i]
                    f_prev, s_prev = fast_sma.iloc[i-1], slow_sma.iloc[i-1]
                    if pd.isna(f_prev) or pd.isna(s_prev) or pd.isna(f_now) or pd.isna(s_now):
                        return (False, {})
                    if (f_now > s_now) and (f_prev <= s_prev):
                        return True, {"side": +1, "strategy": self.strategy, "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
                    if (f_now < s_now) and (f_prev >= s_prev):
                        return True, {"side": -1, "strategy": self.strategy, "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
                    return (False, {})

            entry = _Entry(r, slp, f"SMA_{f}_{s}")
            exit_rules = [
                FixedStop(),  # uses trade.sl_price
                TrailingStop(trail_pips=30,  pip_size=self.pair.pip_size),
                BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=self.pair.pip_size,
                                  commission=self.fees.commission, overnight_fee=self.fees.overnight_fee),
            ]
            engine = Engine(
                self.df, self.pair, self.fees,
                max_active_trades=1,
                exit_first=True,
                entry_on_next_bar=True,
                entry_fill="open",
                spread_pips=self.spread,
                slippage_pips=self.slip,
            )
            engine.run(initial_capital=self.ic, entry_rules=[entry], exit_rules=exit_rules)

            return {
                "fast_sma": f, "slow_sma": s, "sl_pips": slp, "risk_pct": r,
                "num_trades": engine.metrics['num_trades'],
                "cumulative_pnl": engine.metrics['cumulative_pnl'],
                "max_drawdown": engine.metrics['max_drawdown'],
                "win_rate": engine.metrics['win_rate'],
                "sharpe": engine.metrics['sharpe'],
                "cagr": engine.metrics['cagr'],
                "total_fees": engine.metrics['total_fees'],
            }
        except Exception as e:
            return {"fast_sma": f, "slow_sma": s, "sl_pips": slp, "risk_pct": r, "error": repr(e)}

#%% Ray Actor: Fast-realism worker (precomputes SMA map + all exits)
@ray.remote
class FastWorker:
    def __init__(self, df_ref, pair_params, fees_params, initial_capital,
                 fast_vals=None, slow_vals=None):
        from quant_tools import CurrencyPair, Fees
        import ray as _ray
        try:
            from ray import ObjectRef
        except Exception:
            ObjectRef = ()
        self.df = _ray.get(df_ref) if isinstance(df_ref, ObjectRef) else df_ref
        self.pair = CurrencyPair(*pair_params)
        self.fees = Fees(*fees_params)
        self.ic = float(initial_capital)

        close = self.df["close"].astype(float)
        winset = sorted({int(x) for x in (fast_vals or [])} | {int(x) for x in (slow_vals or [])})
        self._sma = {w: close.rolling(w).mean() for w in winset}

    def run(self, f, s, slp, r):
        try:
            import pandas as pd
            from quant_tools.backtest_fast import BacktestEngine as Engine
            from quant_tools.exits_fast import FixedStop, TrailingStop, BreakEvenStepStop

            f = int(f); s = int(s); slp = int(slp); r = float(r)
            if s <= f:
                return None

            fast_sma = self._sma.get(f)
            slow_sma = self._sma.get(s)
            if fast_sma is None:
                fast_sma = self.df["close"].rolling(f).mean(); self._sma[f] = fast_sma
            if slow_sma is None:
                slow_sma = self.df["close"].rolling(s).mean(); self._sma[s] = slow_sma
            idx = self.df.index

            class _Entry:
                def __init__(self, risk_pct, sl_pips, name):
                    self.risk_pct = float(risk_pct); self.sl_pips = int(sl_pips); self.strategy = name
                def check(self, ts, price_val, _df):
                    i = idx.get_loc(ts)
                    if i == 0: return (False, {})
                    f_now, s_now = fast_sma.iloc[i], slow_sma.iloc[i]
                    f_prev, s_prev = fast_sma.iloc[i-1], slow_sma.iloc[i-1]
                    if pd.isna(f_prev) or pd.isna(s_prev) or pd.isna(f_now) or pd.isna(s_now):
                        return (False, {})
                    if (f_now > s_now) and (f_prev <= s_prev):
                        return True, {"side": +1, "strategy": self.strategy, "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
                    if (f_now < s_now) and (f_prev >= s_prev):
                        return True, {"side": -1, "strategy": self.strategy, "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
                    return (False, {})

            entry = _Entry(r, slp, f"SMA_{f}_{s}")
            exit_rules = [
                FixedStop(sl_pips=slp, pip_size=self.pair.pip_size),
                TrailingStop(trail_pips=30,  pip_size=self.pair.pip_size),
                BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=self.pair.pip_size,
                                  commission=self.fees.commission, overnight_fee=self.fees.overnight_fee),
            ]
            engine = Engine(
                self.df, self.pair, self.fees,
                max_active_trades=1,
                entry_on_next_bar=True,
                entry_fill="open",
            )
            engine.run(initial_capital=self.ic, entry_rules=[entry], exit_rules=exit_rules)

            return {
                "fast_sma": f, "slow_sma": s, "sl_pips": slp, "risk_pct": r,
                "num_trades": engine.metrics['num_trades'],
                "cumulative_pnl": engine.metrics['cumulative_pnl'],
                "max_drawdown": engine.metrics['max_drawdown'],
                "win_rate": engine.metrics['win_rate'],
                "sharpe": engine.metrics['sharpe'],
                "cagr": engine.metrics['cagr'],
                "total_fees": engine.metrics['total_fees'],
            }
        except Exception as e:
            return {"fast_sma": f, "slow_sma": s, "sl_pips": slp, "risk_pct": r, "error": repr(e)}

#%% Data & refs
SYMBOL = "USD/JPY"
START  = datetime(2025, 1, 1)
END    = datetime(2025, 3, 31)
INITIAL_CAPITAL = 1000.0

pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)
df = fetch_fx(pair.symbol, start=START, end=END)

df_ref = ray.put(df)  # put once (zero-copy to workers)
assert hasattr(ray, "ObjectRef") and isinstance(df_ref, ray.ObjectRef), f"df_ref is not an ObjectRef: {type(df_ref)}"

pair_params = (pair.symbol, pair.pip_size, pair.contract_size)
fees_params = (fees.commission, fees.overnight_fee, fees.swap_fee)

#%% Realism knobs (for FULL)
SPREAD_PIPS   = 0.5
SLIPPAGE_PIPS = 0.1

#%% Choose worker type
Worker = FullWorker  # or FastWorker

#%% Create actors (pass window lists so they precompute SMA once)
if Worker is FullWorker:
    actors = [
        Worker.remote(df_ref, pair_params, fees_params,
                      SPREAD_PIPS, SLIPPAGE_PIPS, INITIAL_CAPITAL,
                      list(FAST_RANGE), list(SLOW_RANGE))
        for _ in range(num_cpus)
    ]
else:
    actors = [
        Worker.remote(df_ref, pair_params, fees_params,
                      INITIAL_CAPITAL,
                      list(FAST_RANGE), list(SLOW_RANGE))
        for _ in range(num_cpus)
    ]

#%% Submit round-robin and gather in batches (limits outstanding futures)
futures = [actors[i % len(actors)].run.remote(f, s, slp, r)
           for i, (f, s, slp, r) in enumerate(params)]

rows = []
BATCH = max(32, len(actors) * 8)  # tune if needed
for k in range(0, len(futures), BATCH):
    part = ray.get(futures[k:k+BATCH])
    rows.extend([res for res in part if (res and "cumulative_pnl" in res)])
    print(f"Completed {min(k+BATCH, len(futures))}/{len(futures)} tasks")

df_results_ray = pd.DataFrame(rows).sort_values("cumulative_pnl", ascending=False)
df_results_ray.head(12)

#%% Optional: inspect any errors
errs = [r for r in rows if isinstance(r, dict) and "error" in r]
if errs:
    print("Errors encountered (showing up to 5):")
    for e in errs[:5]:
        print(e)

#%% Filter promising strategies (quick sanity view)
filter_strategies(df_results_ray, min_pnl_pct=5, max_dd_pct=25, initial_capital=INITIAL_CAPITAL)

#%% (Optional) Heatmap slices
if not df_results_ray.empty:
    fixed_risk = float(sorted(RISK_LIST)[0])
    fixed_sl   = int(sorted(SL_LIST)[len(SL_LIST)//2])
    slice_fs = df_results_ray[(df_results_ray["risk_pct"] == fixed_risk) & (df_results_ray["sl_pips"] == fixed_sl)]
    if not slice_fs.empty:
        plot_heatmap(slice_fs, x_col="fast_sma", y_col="slow_sma", value_col="cumulative_pnl")
        plt.show()
#%%
dd_cap   = df_results_ray["max_drawdown"].quantile(0.30)  # lowest 30% DD
fee_cap  = df_results_ray["total_fees"].quantile(0.50)    # <= median fees
cands = df_results_ray.query("max_drawdown <= @dd_cap and total_fees <= @fee_cap").copy()

# composite: (Sharpe normalized) + (PnL/DD)
cands["score"] = cands["sharpe"].fillna(0) + (cands["cumulative_pnl"] / (1 + cands["max_drawdown"]))
cands.sort_values(["score","win_rate"], ascending=[False, False]).head(15)

# %%
import matplotlib.pyplot as plt

def plot_frontier_with_labels(df, pnl_col="cumulative_pnl", dd_col="max_drawdown",
                              label_cols=("fast_sma","slow_sma","risk_pct")):
    # Compute frontier again
    points = df[[pnl_col, dd_col]].values
    is_front = []
    for i, (p, d) in enumerate(points):
        dominated = ((points[:,0] >= p) & (points[:,1] <= d) &
                     ((points[:,0] > p) | (points[:,1] < d))).any()
        is_front.append(not dominated)
    front = df[is_front]

    # Plot all vs frontier
    plt.figure(figsize=(8,6))
    plt.scatter(df[dd_col], df[pnl_col], alpha=0.3, label="All strategies")
    plt.scatter(front[dd_col], front[pnl_col], color="red", label="Efficient frontier")

    # Annotate frontier points
    for _, row in front.iterrows():
        label = ",".join(str(row[c]) for c in label_cols)
        plt.annotate(label, (row[dd_col], row[pnl_col]),
                     textcoords="offset points", xytext=(5,5), fontsize=8)

    plt.xlabel("Max Drawdown")
    plt.ylabel("Cumulative PnL")
    plt.title("Efficient Frontier with Parameter Labels")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example call
plot_frontier_with_labels(df_results_ray)


# %%
df_results_ray["pnl_dd_ratio"] = df_results_ray["cumulative_pnl"] / df_results_ray["max_drawdown"].replace(0,1)
top = df_results_ray.sort_values("pnl_dd_ratio", ascending=False).head(20)

# %%
