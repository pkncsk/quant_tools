#%% Imports & path
from datetime import datetime
import os, sys
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("D:/coding/mt5fin/dev"))

from quant_tools import fetch_fx, CurrencyPair, Fees, plot_heatmap
from quant_tools.backtest import BacktestEngine as BacktestEngineFocus, filter_strategies
from quant_tools.exits import FixedStop as FixedStopFocus, TrailingStop as TrailingStopFocus, BreakEvenStepStop as BreakEvenStepStopFocus
from quant_tools.backtest_fast import BacktestEngine as BacktestEngineSweep
from quant_tools.exits_fast import FixedStop as FixedStopSweep, TrailingStop as TrailingStopSweep, BreakEvenStepStop as BreakEvenStepStopSweep

#%% Ray setup
import ray
DEV_PATH = r"D:\coding\mt5fin\dev"
num_cpus = min(os.cpu_count() or 2, 8)
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    num_cpus=num_cpus,
    runtime_env={
        "working_dir": DEV_PATH,
        "env_vars": {"PYTHONPATH": DEV_PATH},
    },
)

#%% Grid
FAST_RANGE = range(5, 31, 1)
SLOW_RANGE = range(6, 32, 1)
SL_LIST    = [20, 30, 50, 70]
RISK_LIST  = [1, 2, 5, 10]
params = [(f, s, slp, r) for f in FAST_RANGE for s in SLOW_RANGE for slp in SL_LIST for r in RISK_LIST]

#%% Entry class with slope filter
def build_entry(idx, fast_sma, slow_sma, trend_sma, risk_pct, sl_pips, name):
    class _EntrySlope:
        def __init__(self):
            self.risk_pct = float(risk_pct)
            self.sl_pips  = int(sl_pips)
            self.strategy = name
        def check(self, ts, price_val, _df):
            i = idx.get_loc(ts)
            if i == 0: return (False, {})
            f_now, s_now = fast_sma.iloc[i], slow_sma.iloc[i]
            f_prev, s_prev = fast_sma.iloc[i-1], slow_sma.iloc[i-1]
            t_now, t_prev = trend_sma.iloc[i], trend_sma.iloc[i-1]
            if any(pd.isna(x) for x in [f_now, s_now, f_prev, s_prev, t_now, t_prev]):
                return (False, {})
            slope_up, slope_down = (t_now > t_prev), (t_now < t_prev)
            cross_up = (f_now > s_now) and (f_prev <= s_prev) and slope_up
            cross_dn = (f_now < s_now) and (f_prev >= s_prev) and slope_down
            if cross_up:
                return True, {"side": +1, "strategy": self.strategy,
                              "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
            if cross_dn:
                return True, {"side": -1, "strategy": self.strategy,
                              "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
            return (False, {})
    return _EntrySlope()

#%% Ray Actor: Full-realism
@ray.remote
class FullWorker:
    def __init__(self, df_ref, pair_params, fees_params, spread_pips, slippage_pips, initial_capital,
                 fast_vals=None, slow_vals=None):
        from quant_tools import CurrencyPair, Fees
        import ray as _ray
        try: from ray import ObjectRef
        except Exception: ObjectRef = ()
        self.df = _ray.get(df_ref) if isinstance(df_ref, ObjectRef) else df_ref
        self.pair = CurrencyPair(*pair_params)
        self.fees = Fees(*fees_params)
        self.spread, self.slip, self.ic = float(spread_pips), float(slippage_pips), float(initial_capital)
        close = self.df["close"].astype(float)
        winset = sorted({int(x) for x in (fast_vals or [])} | {int(x) for x in (slow_vals or [])} | {200})
        self._sma = {w: close.rolling(w).mean() for w in winset}
        self.idx = self.df.index

    def run(self, f, s, slp, r):
        import pandas as pd
        from quant_tools.backtest import BacktestEngine as Engine
        from quant_tools.exits import FixedStop, TrailingStop, BreakEvenStepStop
        f, s, slp, r = int(f), int(s), int(slp), float(r)
        if s < f + 4: return None  # enforce slow ≥ fast+4
        fast_sma = self._sma.get(f) or self.df["close"].rolling(f).mean()
        slow_sma = self._sma.get(s) or self.df["close"].rolling(s).mean()
        trend_sma = self._sma.get(200)
        entry = build_entry(self.idx, fast_sma, slow_sma, trend_sma, r, slp, f"SMA_{f}_{s}")
        exit_rules = [
            FixedStop(sl_pips=slp, pip_size=self.pair.pip_size),
            TrailingStop(trail_pips=30,  pip_size=self.pair.pip_size),
            BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=self.pair.pip_size,
                              commission=self.fees.commission, overnight_fee=self.fees.overnight_fee),
        ]
        engine = Engine(self.df, self.pair, self.fees,
                        max_active_trades=1, exit_first=True,
                        entry_on_next_bar=True, entry_fill="open",
                        spread_pips=self.spread, slippage_pips=self.slip)
        engine.run(initial_capital=self.ic, entry_rules=[entry], exit_rules=exit_rules)
        return {**{"fast_sma": f, "slow_sma": s, "sl_pips": slp, "risk_pct": r}, **engine.metrics}

#%% Ray Actor: Fast-realism
@ray.remote
class FastWorker:
    def __init__(self, df_ref, pair_params, fees_params, initial_capital,
                 fast_vals=None, slow_vals=None):
        from quant_tools import CurrencyPair, Fees
        import ray as _ray
        try: from ray import ObjectRef
        except Exception: ObjectRef = ()
        self.df = _ray.get(df_ref) if isinstance(df_ref, ObjectRef) else df_ref
        self.pair = CurrencyPair(*pair_params)
        self.fees = Fees(*fees_params)
        self.ic = float(initial_capital)
        close = self.df["close"].astype(float)
        winset = sorted({int(x) for x in (fast_vals or [])} | {int(x) for x in (slow_vals or [])} | {200})
        self._sma = {w: close.rolling(w).mean() for w in winset}
        self.idx = self.df.index

    def run(self, f, s, slp, r):
        import pandas as pd
        from quant_tools.backtest_fast import BacktestEngine as Engine
        from quant_tools.exits_fast import FixedStop, TrailingStop, BreakEvenStepStop
        f, s, slp, r = int(f), int(s), int(slp), float(r)
        if s < f + 4: return None
        fast_sma = self._sma.get(f) or self.df["close"].rolling(f).mean()
        slow_sma = self._sma.get(s) or self.df["close"].rolling(s).mean()
        trend_sma = self._sma.get(200)
        entry = build_entry(self.idx, fast_sma, slow_sma, trend_sma, r, slp, f"SMA_{f}_{s}")
        exit_rules = [
            FixedStop(sl_pips=slp, pip_size=self.pair.pip_size),
            TrailingStop(trail_pips=30,  pip_size=self.pair.pip_size),
            BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=self.pair.pip_size,
                              commission=self.fees.commission, overnight_fee=self.fees.overnight_fee),
        ]
        engine = Engine(self.df, self.pair, self.fees,
                        max_active_trades=1, entry_on_next_bar=True, entry_fill="open")
        engine.run(initial_capital=self.ic, entry_rules=[entry], exit_rules=exit_rules)
        return {**{"fast_sma": f, "slow_sma": s, "sl_pips": slp, "risk_pct": r}, **engine.metrics}

#%% Data
SYMBOL = "USD/JPY"
START, END = datetime(2025, 1, 1), datetime(2025, 3, 31)
INITIAL_CAPITAL = 1000.0
pair = CurrencyPair(SYMBOL, pip_size=0.01, contract_size=100_000)
fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)
df = fetch_fx(pair.symbol, start=START, end=END)

df_ref = ray.put(df)
pair_params = (pair.symbol, pair.pip_size, pair.contract_size)
fees_params = (fees.commission, fees.overnight_fee, fees.swap_fee)

SPREAD_PIPS, SLIPPAGE_PIPS = 0.5, 0.1
Worker = FullWorker  # or FastWorker

#%% Actors
if Worker is FullWorker:
    actors = [Worker.remote(df_ref, pair_params, fees_params, SPREAD_PIPS, SLIPPAGE_PIPS,
                            INITIAL_CAPITAL, list(FAST_RANGE), list(SLOW_RANGE)) for _ in range(num_cpus)]
else:
    actors = [Worker.remote(df_ref, pair_params, fees_params, INITIAL_CAPITAL,
                            list(FAST_RANGE), list(SLOW_RANGE)) for _ in range(num_cpus)]

#%% Submit tasks
futures = [actors[i % len(actors)].run.remote(f, s, slp, r) for i, (f, s, slp, r) in enumerate(params)]
rows, BATCH = [], max(32, len(actors) * 8)
for k in range(0, len(futures), BATCH):
    part = ray.get(futures[k:k+BATCH])
    rows.extend([res for res in part if (res and "cumulative_pnl" in res)])
    print(f"Completed {min(k+BATCH, len(futures))}/{len(futures)} tasks")

df_results_ray = pd.DataFrame(rows).sort_values("cumulative_pnl", ascending=False)
df_results_ray.head(12)
