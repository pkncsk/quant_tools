# sma_cross_sma_flip.py
# Unified focus/sweep runner with flexible CLI and saved heatmaps.

from datetime import datetime
import itertools as it
import argparse
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

# --- dev path; remove if installed as a package ---
sys.path.append(os.path.abspath("D:/coding/mt5fin/dev"))

from quant_tools import (
    fetch_fx, CurrencyPair, Fees, plot_heatmap
)

# ========= Strategy-local entry rule =========
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

# ========= Helpers =========
def parse_unit4(s: str):
    """Parse 'fast,slow,sl_pips,risk_pct' -> (int,int,int,float)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Must be 'fast,slow,sl_pips,risk_pct' (4 comma-separated numbers).")
    fast, slow, sl_pips = map(int, parts[:3])
    risk = float(parts[3])
    return fast, slow, sl_pips, risk

def parse_dim(spec: str, as_float=False):
    """
    Parse a sweep dimension:
      - 'a:b:c'  -> inclusive range [a, b] step c
      - 'a,b,c'  -> explicit list
      - 'n'      -> single value list [n]
    Returns list[int] or list[float].
    """
    if spec is None or spec == "":
        return []
    spec = spec.strip()
    num = float if as_float else int

    if ":" in spec:  # range
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2,3):
            raise argparse.ArgumentTypeError("Range must be 'start:end[:step]'.")
        start = num(parts[0])
        end   = num(parts[1])
        step  = num(parts[2]) if len(parts) == 3 else (1.0 if as_float else 1)
        # inclusive end
        vals = []
        v = start
        # handle float step carefully
        if as_float:
            # to avoid FP drift, iterate with counter
            nmax = int(round((end - start)/step)) if step != 0 else 0
            for k in range(nmax + 1):
                vk = start + k*step
                if (step > 0 and vk > end + 1e-12) or (step < 0 and vk < end - 1e-12):
                    break
                vals.append(float(vk))
        else:
            rng = range(int(start), int(end) + (1 if step > 0 else -1), int(step))
            vals = list(rng)
        return vals
    elif "," in spec:  # list
        vals = [num(x.strip()) for x in spec.split(",") if x.strip() != ""]
        return vals
    else:  # single
        return [num(spec)]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_current_fig(outpath: str):
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

# ========= Build exits/engines by realism =========
def build_engine(df, pair, fees, realism: str, spread_pips=0.5, slippage_pips=0.1):
    if realism == "full":
        from quant_tools.backtest import BacktestEngine as Engine
        return Engine(
            df, pair, fees,
            max_active_trades=1,
            exit_first=True,
            entry_on_next_bar=True,
            entry_fill="open",
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
        )
    elif realism == "fast":
        from quant_tools.backtest_fast import BacktestEngine as Engine
        return Engine(
            df, pair, fees,
            max_active_trades=1,
            entry_on_next_bar=True,
            entry_fill="open",
        )
    else:
        raise ValueError("realism must be 'full' or 'fast'")

def build_exit_rules(pair: CurrencyPair, fees: Fees, sl_pips: int, realism: str):
    if realism == "full":
        from quant_tools.exits import FixedStop, TrailingStop, BreakEvenStepStop
        return [
            FixedStop(sl_pips=sl_pips, pip_size=pair.pip_size),
            TrailingStop(trail_pips=30,  pip_size=pair.pip_size),
            BreakEvenStepStop(trigger_pips=20, step_pips=10, pip_size=pair.pip_size,
                              commission=fees.commission, overnight_fee=fees.overnight_fee),
        ]
    else:  # fast mode defaults lean for speed
        from quant_tools.exits_fast import FixedStop
        return [FixedStop(sl_pips=sl_pips, pip_size=pair.pip_size)]

# ========= Runners =========
def run_focus(df, pair, fees, fast, slow, sl_pips, risk, realism, initial_capital, outdir=None, save_csv=False):
    entry = SMAFlipEntry(df, fast, slow, strategy=f"SMA_{fast}_{slow}",
                         risk_pct=risk, sl_pips=sl_pips)
    exit_rules = build_exit_rules(pair, fees, sl_pips, realism)
    engine = build_engine(df, pair, fees, realism)

    engine.run(initial_capital=initial_capital, entry_rules=[entry], exit_rules=exit_rules)

    # Trade log (print and optional CSV)
    trade_log = pd.DataFrame([{
        "Entry Time": t.entry_time, "Exit Time":  t.exit_time,
        "Side": ("Long" if t.side == 1 else "Short"),
        "Entry Price": t.entry_price, "Exit Price":  t.exit_price,
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

    if outdir:
        ensure_dir(outdir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_csv:
            trade_log.to_csv(os.path.join(outdir, f"trades_focus_{fast}_{slow}_{sl_pips}_{risk}_{realism}_{ts}.csv"),
                             index=False)

    # Equity plot saved
    if outdir:
        plt.figure(figsize=(14, 6))
        plt.plot(engine.equity_curve, label="Equity Curve")
        plt.fill_between(engine.drawdown_series.index,
                         engine.equity_curve - engine.drawdown_series,
                         engine.equity_curve, alpha=0.3, label="Drawdown")
        plt.title(f"Equity (SMA {fast}/{slow}, SL={sl_pips}, Risk={risk}%, {realism})")
        plt.xlabel("Time"); plt.ylabel("Equity"); plt.legend(); plt.grid(True)
        save_current_fig(os.path.join(outdir, f"equity_focus_{fast}_{slow}_{sl_pips}_{risk}_{realism}.png"))

def run_sweep(df, pair, fees, fast_vals, slow_vals, sl_vals, risk_vals,
              realism, initial_capital, outdir=None, save_csv=True):
    rows = []
    for fast, slow, sl_pips, risk in it.product(fast_vals, slow_vals, sl_vals, risk_vals):
        if slow <= fast:
            continue
        entry = SMAFlipEntry(df, fast, slow, strategy=f"SMA_{fast}_{slow}",
                             risk_pct=risk, sl_pips=sl_pips)
        exit_rules = build_exit_rules(pair, fees, int(sl_pips), realism)
        engine = build_engine(df, pair, fees, realism)
        engine.run(initial_capital=initial_capital, entry_rules=[entry], exit_rules=exit_rules)

        rows.append({
            "fast_sma": int(fast), "slow_sma": int(slow),
            "sl_pips": int(sl_pips), "risk_pct": float(risk),
            "num_trades": engine.metrics['num_trades'],
            "cumulative_pnl": engine.metrics['cumulative_pnl'],
            "max_drawdown": engine.metrics['max_drawdown'],
            "win_rate": engine.metrics['win_rate'],
            "sharpe": engine.metrics['sharpe'],
            "cagr": engine.metrics['cagr'],
            "total_fees": engine.metrics['total_fees']
        })

    df_results = pd.DataFrame(rows).sort_values("cumulative_pnl", ascending=False)
    print(df_results.head(12))

    if outdir:
        ensure_dir(outdir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_csv:
            df_results.to_csv(os.path.join(outdir, f"sweep_results_{realism}_{ts}.csv"), index=False)

        # Heatmap 1: cumulative_pnl over (fast, slow) at a fixed (risk, sl) slice (choose best slice by counts)
        # Pick a common slice: risk = min(risk_vals), sl = median(sl_vals)
        fixed_risk = float(sorted(risk_vals)[0])
        fixed_sl   = int(sorted(sl_vals)[len(sl_vals)//2])

        # Ensure the slice has data
        slice_df = df_results[(df_results["risk_pct"] == fixed_risk) & (df_results["sl_pips"] == fixed_sl)]
        if not slice_df.empty:
            plot_heatmap(slice_df, x_col="fast_sma", y_col="slow_sma", value_col="cumulative_pnl")
            save_current_fig(os.path.join(outdir, f"heatmap_pnl_fast_slow_risk{fixed_risk}_sl{fixed_sl}_{realism}.png"))

        # Heatmap 2: sharpe over (sl_pips, risk_pct) at a fixed (fast, slow) slice (pick best by counts)
        fixed_fast = int(sorted(fast_vals)[0])
        fixed_slow = int(sorted([v for v in slow_vals if v > fixed_fast])[0]) if any(v > fixed_fast for v in slow_vals) else int(sorted(slow_vals)[0])
        slice_df2 = df_results[(df_results["fast_sma"] == fixed_fast) & (df_results["slow_sma"] == fixed_slow)]
        if not slice_df2.empty:
            plot_heatmap(slice_df2, x_col="sl_pips", y_col="risk_pct", value_col="sharpe")
            save_current_fig(os.path.join(outdir, f"heatmap_sharpe_sl_risk_fast{fixed_fast}_slow{fixed_slow}_{realism}.png"))

# ========= CLI =========
def main():
    parser = argparse.ArgumentParser(description="SMA flip: focus (unit-4) or sweep (ranges/lists).")
    parser.add_argument("--mode", choices=["focus", "sweep"], default="focus")
    parser.add_argument("--realism", choices=["full", "fast", "auto"], default="auto")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end",   type=str, default="2025-01-31")
    parser.add_argument("--symbol", type=str, default="USD/JPY")
    parser.add_argument("--initial_capital", type=float, default=1000)

    # focus unit (all-in-one)
    parser.add_argument("--focus", type=parse_unit4, help="Unit params 'fast,slow,sl_pips,risk_pct' (e.g., 5,36,30,2)")

    # sweep dims (range/list/single). Examples: 5:31:1  or  5,8,13  or  12
    parser.add_argument("--fast", type=str, help="fast SMA spec (range/list/single)")
    parser.add_argument("--slow", type=str, help="slow SMA spec (range/list/single)")
    parser.add_argument("--sl",   type=str, help="stop-loss pips spec (range/list/single)")
    parser.add_argument("--risk", type=str, help="risk %% spec (range/list/single), floats allowed")

    parser.add_argument("--outdir", type=str, default=None, help="Directory to save outputs (PNG/CSV).")
    parser.add_argument("--save_csv", action="store_true", help="Save trade log (focus) or results (sweep) as CSV.")
    parser.add_argument("--spread_pips", type=float, default=0.5, help="Only used for full realism engine.")
    parser.add_argument("--slippage_pips", type=float, default=0.1, help="Only used for full realism engine.")

    args = parser.parse_args()

    # realism default: focus->full, sweep->fast
    realism = args.realism
    if realism == "auto":
        realism = "full" if args.mode == "focus" else "fast"

    # data
    pair = CurrencyPair(args.symbol, pip_size=0.01, contract_size=100_000)
    df = fetch_fx(pair.symbol, start=datetime.fromisoformat(args.start), end=datetime.fromisoformat(args.end))
    fees = Fees(commission=0.5, overnight_fee=0.01, swap_fee=0.0)

    # run
    if args.mode == "focus":
        if not args.focus:
            raise SystemExit("Focus mode requires --focus 'fast,slow,sl_pips,risk_pct'")
        f, s, sl_pips, r = args.focus
        run_focus(df, pair, fees, f, s, sl_pips, r, realism, args.initial_capital,
                  outdir=args.outdir, save_csv=args.save_csv)
    else:
        # sweep
        fast_vals = parse_dim(args.fast) if args.fast else list(range(5,31))      # default 5..30
        slow_vals = parse_dim(args.slow) if args.slow else list(range(6,55))      # default 6..54
        sl_vals   = parse_dim(args.sl)   if args.sl   else [20,30,50,70]
        risk_vals = parse_dim(args.risk, as_float=True) if args.risk else [1,2,5,10]

        run_sweep(df, pair, fees, fast_vals, slow_vals, sl_vals, risk_vals,
                  realism, args.initial_capital, outdir=args.outdir, save_csv=args.save_csv)

if __name__ == "__main__":
    main()
