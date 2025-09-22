# backtest_fast.py
import pandas as pd
import numpy as np
from .risk import RiskManager
from .trades import Trade

class BacktestEngine:
    """
    FAST sweep engine:
      - exit-first, next-bar entries (at open if available)
      - no spread/slippage adjustments
      - exit rules are expected to be from exits_fast (close-only)
      - same metrics as the full engine
    """
    def __init__(
        self,
        price_df,
        currency_pair,
        fees,
        max_active_trades=1,
        *,
        entry_on_next_bar=True,
        entry_fill="open",   # "open" or "close" for next-bar
    ):
        self.df = price_df
        self.pair = currency_pair
        self.fees = fees
        self.max_active_trades = max(1, int(max_active_trades))

        self.entry_on_next_bar = bool(entry_on_next_bar)
        self.entry_fill = str(entry_fill).lower()
        if self.entry_fill not in ("open", "close"):
            self.entry_fill = "open"

        self.trades = []
        self.open_trades = []
        self._pending_entries = []

        self.equity_curve = pd.Series(index=price_df.index, data=0.0, dtype=float)
        self.drawdown_series = pd.Series(index=price_df.index, data=0.0, dtype=float)
        self.metrics = {}

    # ---------- helpers ----------

    def _bar_price(self, bar, field):
        if field in bar:
            return float(bar[field])
        return float(bar["close"])

    def _open_trade(self, ts, bar, params, risk_manager, current_capital, immediate=False):
        side = int(params["side"])
        price_field = "close" if immediate or not self.entry_on_next_bar else self.entry_fill
        entry_px = self._bar_price(bar, price_field)

        trade = Trade(
            entry_time=ts,
            exit_time=None,
            side=side,
            entry_price=float(entry_px),
            planned_sl_pips=params.get("sl_pips"),
            strategy=params.get("strategy", "Custom"),
        )
        trade.risk_pct = params.get("risk_pct", 1.0)

        risk_manager.capital = current_capital
        trade = risk_manager.assign_trade_params(trade)
        if not trade.cancelled:
            trade.entry_equity = current_capital
            self.open_trades.append(trade)

    def _close_trade(self, trade, exit_time, raw_fill_price,
                     current_capital, global_max_equity, total_fees_accum, wins):
        # No spread/slippage in fast mode: use raw_fill_price directly
        exit_px = float(raw_fill_price)

        trade.exit_time = exit_time
        trade.exit_price = float(exit_px)

        pip_diff = ((trade.exit_price - trade.entry_price) / self.pair.pip_size
                    if trade.side == 1 else (trade.entry_price - trade.exit_price) / self.pair.pip_size)
        trade.pnl = pip_diff * trade.lot_size * self.pair.pip_value_per_lot(trade.entry_price)

        df_slice = self.df.loc[trade.entry_time:exit_time]['close']
        num_bars = int(len(df_slice))
        trade_fees = self.fees.total_fees(trade, num_bars)
        trade.pnl -= trade_fees
        trade.total_fees = trade_fees
        total_fees_accum += trade_fees

        current_capital += trade.pnl
        trade.exit_equity = current_capital
        trade.win = bool(trade.pnl > 0)
        wins += int(trade.win)

        global_max_equity = max(global_max_equity, current_capital)
        trade.drawdown = global_max_equity - current_capital

        self.trades.append(trade)
        return current_capital, global_max_equity, total_fees_accum, wins

    def _process_exits(self, ts, bar, current_capital, global_max_equity, total_fees_accum, wins, exit_rules):
        if not self.open_trades:
            return current_capital, global_max_equity, total_fees_accum, wins

        still_open = []
        for trade in self.open_trades:
            closed = False
            for rule in (exit_rules or []):
                hit, raw_px = rule.update(trade, bar, ts)  # close-only logic inside exits_fast
                if hit:
                    current_capital, global_max_equity, total_fees_accum, wins = self._close_trade(
                        trade, ts, raw_px, current_capital, global_max_equity, total_fees_accum, wins
                    )
                    closed = True
                    break

            if not closed:
                if ts == self.df.index[-1]:
                    raw_px = float(bar['close'])
                    current_capital, global_max_equity, total_fees_accum, wins = self._close_trade(
                        trade, ts, raw_px, current_capital, global_max_equity, total_fees_accum, wins
                    )
                else:
                    still_open.append(trade)

        self.open_trades = still_open
        return current_capital, global_max_equity, total_fees_accum, wins

    # ---------- main ----------

    def run(self, initial_capital=1000, entry_rules=None, exit_rules=None):
        current_capital = float(initial_capital)
        risk_manager = RiskManager(capital=current_capital, currency_pair=self.pair)
        global_max_equity, wins, total_fees = float(initial_capital), 0, 0.0

        for ts, row in self.df.iterrows():
            bar = {
                'open': float(row['open']) if 'open' in row else float(row['close']),
                'high': float(row['high']) if 'high' in row else float(row['close']),
                'low':  float(row['low'])  if 'low'  in row else float(row['close']),
                'close': float(row['close'])
            }

            # Exits first
            current_capital, global_max_equity, total_fees, wins = self._process_exits(
                ts, bar, current_capital, global_max_equity, total_fees, wins, exit_rules
            )

            # Open pending (from previous bar)
            if self.entry_on_next_bar and self._pending_entries:
                pendings = self._pending_entries
                self._pending_entries = []
                for params in pendings:
                    if len(self.open_trades) < self.max_active_trades:
                        self._open_trade(ts, bar, params, risk_manager, current_capital, immediate=False)

            # Detect new entries this bar
            new_signals = []
            for rule in (entry_rules or []):
                hit, params = rule.check(ts, bar['close'], self.df)
                if hit:
                    new_signals.append(params)

            # Queue or execute
            if self.entry_on_next_bar:
                self._pending_entries.extend(new_signals)
            else:
                for params in new_signals:
                    if len(self.open_trades) < self.max_active_trades:
                        self._open_trade(ts, bar, params, risk_manager, current_capital, immediate=True)

            # Equity track
            self.equity_curve.loc[ts] = current_capital
            self.drawdown_series.loc[ts] = global_max_equity - current_capital

        self._finalize_metrics(initial_capital, current_capital, wins, total_fees)

    def _finalize_metrics(self, initial_capital, current_capital, wins, total_fees):
        self.metrics['cumulative_pnl'] = current_capital - initial_capital
        self.metrics['max_drawdown'] = float(self.drawdown_series.max()) if len(self.drawdown_series) else 0.0
        self.metrics['num_trades'] = len(self.trades)
        self.metrics['win_rate'] = (wins / len(self.trades)) if self.trades else 0.0
        self.metrics['total_fees'] = total_fees

        returns = self.equity_curve.pct_change().dropna()
        self.metrics['sharpe'] = (
            returns.mean() / returns.std() * np.sqrt(252 * (24*60/5))
            if len(returns) > 1 and returns.std() > 0 else None
        )

        total_days = (self.df.index[-1] - self.df.index[0]).days if len(self.df.index) else 0
        if total_days > 0 and current_capital > 0 and initial_capital > 0:
            years = total_days / 365
            self.metrics['cagr'] = (current_capital / initial_capital) ** (1/years) - 1 if years >= 1 else None
        else:
            self.metrics['cagr'] = None


def filter_strategies(df, min_pnl_pct=5, max_dd_pct=10, initial_capital=1000):
    min_pnl = initial_capital * (min_pnl_pct / 100)
    max_dd = initial_capital * (max_dd_pct / 100)
    return df[(df['cumulative_pnl'] >= min_pnl) & (df['max_drawdown'] <= max_dd)]
