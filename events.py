class Event:
    """
    A trading event (entry trigger).
    Created inside the backtest loop when an EntryRule fires.
    """
    def __init__(self, time, side, price, strategy, risk_pct=None, sl_pips=None):
        self.time = time          # pandas.Timestamp
        self.side = side          # +1 long, -1 short
        self.price = price        # float
        self.strategy = strategy  # str tag
        self.risk_pct = risk_pct
        self.sl_pips = sl_pips

    def __repr__(self):
        return (f"Event(time={self.time}, side={self.side}, price={self.price}, "
                f"strat={self.strategy}, risk={self.risk_pct}, sl={self.sl_pips})")
