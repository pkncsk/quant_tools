# sweep_workers_full.py
import pandas as pd

_SHARED = {}

def init_worker(symbol, pip_size, contract_size,
                commission, overnight_fee, swap_fee,
                start_iso, end_iso,
                spread_pips, slippage_pips, initial_capital,
                dev_path=None):
    """Initializer: rebuild heavy objects per process."""
    import sys, os
    from datetime import datetime
    if dev_path:
        sys.path.append(os.path.abspath(dev_path))

    try:
        from quant_tools import CurrencyPair, Fees, fetch_fx
        pair = CurrencyPair(symbol, pip_size=float(pip_size), contract_size=int(contract_size))
        fees = Fees(commission=float(commission), overnight_fee=float(overnight_fee), swap_fee=float(swap_fee))
        df = fetch_fx(symbol, start=datetime.fromisoformat(start_iso), end=datetime.fromisoformat(end_iso))
    except Exception as e:
        # Bubble init failures to parent through a single-row DF later
        _SHARED.clear()
        _SHARED.update({"_init_error": f"{type(e).__name__}: {e}"})
        return

    _SHARED.clear()
    _SHARED.update(dict(
        symbol=symbol,
        pair=pair, fees=fees, df=df,
        spread=float(spread_pips), slip=float(slippage_pips),
        icap=float(initial_capital),
    ))

class _SMAFlipEntry:
    """Inline entry rule using provided SMA series (precomputed per chunk)."""
    def __init__(self, idx, fast_sma, slow_sma, risk_pct, sl_pips, name):
        self.idx = idx
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.risk_pct = float(risk_pct)
        self.sl_pips = int(sl_pips)
        self.strategy = name

    def check(self, ts, price, df):
        import pandas as pd
        i = self.idx.get_loc(ts)
        if i == 0:
            return (False, {})
        f_now, s_now = self.fast_sma.iloc[i], self.slow_sma.iloc[i]
        f_prev, s_prev = self.fast_sma.iloc[i-1], self.slow_sma.iloc[i-1]
        if pd.isna(f_prev) or pd.isna(s_prev) or pd.isna(f_now) or pd.isna(s_now):
            return (False, {})
        cross_up = (f_now > s_now) and (f_prev <= s_prev)
        cross_dn = (f_now < s_now) and (f_prev >= s_prev)
        if cross_up:
            return True, {"side": +1, "strategy": self.strategy,
                          "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
        if cross_dn:
            return True, {"side": -1, "strategy": self.strategy,
                          "risk_pct": self.risk_pct, "sl_pips": self.sl_pips}
        return (False, {})

def run_full_chunk(chunk):
    """Run a chunk of (fast, slow, sl_pips, risk_pct) combos; return DataFrame or DF with _error."""
    import traceback
    try:
        if "_init_error" in _SHARED:
            return pd.DataFrame([{"_error": _SHARED["_init_error"], "_traceback": "init_worker failed"}])

        from quant_tools.backtest import BacktestEngine
        from quant_tools.exits import FixedStop, TrailingStop, BreakEvenStepStop

        df   = _SHARED["df"]
        pair = _SHARED["pair"]
        fees = _SHARED["fees"]
        spread = _SHARED["spread"]
        slip   = _SHARED["slip"]
        icap   = _SHARED["icap"]

        price = df["close"]
        f_set = sorted({int(f) for f,_,_,_ in chunk})
        s_set = sorted({int(s) for _,s,_,_ in chunk})
        f_cache = {n: price.rolling(n).mean() for n in f_set}
        s_cache = {n: price.rolling(n).mean() for n in s_set}
        idx = df.index

        rows = []
        for f, s, slp, r in chunk:
            f, s, slp, r = int(f), int(s), int(slp), float(r)
            if s <= f:
                continue

            entry = _SMAFlipEntry(idx, f_cache[f], s_cache[s], r, slp, f"SMA_{f}_{s}")
            exit_rules = [
                FixedStop(),  # uses trade.sl_price from RiskManager
                TrailingStop(trail_pips=30,  pip_size=pair.pip_size),
                BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                                  commission=fees.commission, overnight_fee=fees.overnight_fee),
            ]

            engine = BacktestEngine(
                df, pair, fees,
                max_active_trades=1,
                exit_first=True,
                entry_on_next_bar=True,
                entry_fill="open",
                spread_pips=spread,
                slippage_pips=slip,
            )
            engine.run(initial_capital=icap, entry_rules=[entry], exit_rules=exit_rules)

            rows.append({
                "symbol": _SHARED["symbol"],
                "fast_sma": f, "slow_sma": s,
                "sl_pips": slp, "risk_pct": r,
                "num_trades": engine.metrics['num_trades'],
                "cumulative_pnl": engine.metrics['cumulative_pnl'],
                "max_drawdown": engine.metrics['max_drawdown'],
                "win_rate": engine.metrics['win_rate'],
                "sharpe": engine.metrics['sharpe'],
                "cagr": engine.metrics['cagr'],
                "total_fees": engine.metrics['total_fees']
            })

        return pd.DataFrame(rows)

    except Exception as e:
        tb = traceback.format_exc()
        return pd.DataFrame([{"_error": f"{type(e).__name__}: {e}", "_traceback": tb}])
