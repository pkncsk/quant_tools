class Trade:
    def __init__(self, entry_time, exit_time, side, entry_price,
                 planned_sl_pips=None, strategy=None):
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.side = side
        self.entry_price = entry_price
        self.planned_sl_pips = planned_sl_pips
        self.strategy = strategy

        # runtime attributes
        self.lot_size = None
        self.sl_price = None
        self.exit_price = None
        self.pnl = None
        self.drawdown = None
        self.win = None
        self.risk_pct = None
        self.cancelled = False
        self.entry_equity = None
        self.exit_equity = None
        self.total_fees = 0.0
        self.exit_reason = None
        self.be_armed = False
        self.initial_sl_price = None

        # --- NEW ---
        # flexible per-trade metadata (strategy-specific info)
        self.meta = {
            "intraday_equity": []
        }
class Fees:
    def __init__(self, commission=0.0, overnight_fee=0.0, swap_fee=0.0):
        self.commission = commission
        self.overnight_fee = overnight_fee
        self.swap_fee = swap_fee

    def total_fees(self, trade, num_bars=1):
        return self.commission + num_bars * (self.overnight_fee + self.swap_fee)
