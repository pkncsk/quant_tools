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


    def assign_trade_params(self, trade):
        """
        Assign stop and lot size based on capital, risk_pct, and sl_pips.
        Enforces broker constraints (min/max/step).
        Cancels trade if lot below tradable threshold.
        """
        sl_pips = getattr(trade, "planned_sl_pips", None)
        risk_pct = getattr(trade, "risk_pct", None) or self.default_risk_pct

        if sl_pips is None:
            sl_pips = self.default_sl_pips

        pip_value = self.pair.pip_value_per_lot(trade.entry_price)
        risk_value = self.capital * (risk_pct / 100)
        trade.lot_size = risk_value / (sl_pips * pip_value)

        # --- quantize & enforce ---
        trade.lot_size = round(trade.lot_size / self.lot_step) * self.lot_step

        if trade.lot_size < self.min_lot:
            trade.cancelled = True
            trade.cancel_reason = "LotBelowMinimum"
            return trade

        if trade.lot_size > self.max_lot:
            trade.lot_size = self.max_lot

        # --- compute stop loss price ---
        if trade.side == 1:
            trade.sl_price = trade.entry_price - sl_pips * self.pair.pip_size
        else:
            trade.sl_price = trade.entry_price + sl_pips * self.pair.pip_size

        trade.initial_sl_price = trade.sl_price
        trade.be_armed = False
        return trade
    def assign_trade_params(self, trade):
        """
        Assign stop and lot size based on capital, risk_pct, and sl_pips.
        Enforces broker constraints (min/max/step).
        Cancels trade if lot below tradable threshold.
        Logs sizing details into trade.
        """
        sl_pips = getattr(trade, "planned_sl_pips", None)
        risk_pct = getattr(trade, "risk_pct", None) or self.default_risk_pct

        if sl_pips is None:
            sl_pips = self.default_sl_pips

        pip_value = self.pair.pip_value_per_lot(trade.entry_price)
        risk_value = self.capital * (risk_pct / 100)
        trade.lot_size = risk_value / (sl_pips * pip_value)

        # quantize & enforce
        trade.lot_size = round(trade.lot_size / self.lot_step) * self.lot_step

        # cancellation conditions
        if trade.lot_size < self.min_lot:
            trade.cancelled = True
            trade.cancel_reason = f"LotBelowMinimum ({trade.lot_size:.4f})"
            trade.log = {
                "risk_value": risk_value,
                "risk_pct": risk_pct,
                "sl_pips": sl_pips,
                "pip_value": pip_value,
                "lot_raw": trade.lot_size,
            }
            return trade

        if trade.lot_size > self.max_lot:
            trade.lot_size = self.max_lot

        # stop-loss assignment
        if trade.side == 1:
            trade.sl_price = trade.entry_price - sl_pips * self.pair.pip_size
        else:
            trade.sl_price = trade.entry_price + sl_pips * self.pair.pip_size

        trade.initial_sl_price = trade.sl_price
        trade.be_armed = False

        # detailed log
        trade.log = {
            "risk_value": risk_value,
            "risk_pct": risk_pct,
            "sl_pips": sl_pips,
            "pip_value": pip_value,
            "lot_final": trade.lot_size,
            "cancelled": False,
        }
        return trade
