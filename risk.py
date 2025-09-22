from .trades import Trade
from .instruments import CurrencyPair

class RiskManager:
    def __init__(self, capital, currency_pair: CurrencyPair,
                 default_risk_pct=1.0, default_sl_pips=50,
                 min_lot=0.01, max_lot=50, lot_step=0.01):
        self.capital = capital
        self.pair = currency_pair
        self.default_risk_pct = default_risk_pct
        self.default_sl_pips = default_sl_pips
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.lot_step = lot_step

    def assign_trade_params(self, trade: Trade):
        trade.risk_pct = trade.risk_pct or self.default_risk_pct
        sl_pips = trade.planned_sl_pips or self.default_sl_pips

        risk_amount = self.capital * trade.risk_pct / 100
        pip_value_per_lot = self.pair.pip_value_per_lot(trade.entry_price)
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        if lot_size < self.min_lot:
            trade.lot_size = 0
            trade.cancelled = True
            return trade

        lot_size = min(self.max_lot, round(lot_size / self.lot_step) * self.lot_step)
        trade.lot_size = lot_size
        trade.cancelled = False

        trade.sl_price = (trade.entry_price - sl_pips * self.pair.pip_size if trade.side == 1
                          else trade.entry_price + sl_pips * self.pair.pip_size)

        return trade
