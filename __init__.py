from .data_fetch import fetch_fx
from .instruments import CurrencyPair
from .events import Event
from .trades import Trade, Fees
from .risk import RiskManager

# Full realism (default)
from .exits import FixedStop as FixedStop, TrailingStop as TrailingStop, BreakEvenStepStop as BreakEvenStepStop
from .backtest import BacktestEngine as BacktestEngine, filter_strategies

# Fast sweep variants (opt-in)
from .exits_fast import FixedStop as FixedStopFast, TrailingStop as TrailingStopFast, BreakEvenStepStop as BreakEvenStepStopFast
from .backtest_fast import BacktestEngine as BacktestEngineFast
from .plotting import plot_heatmap


