class ExitRule:
    def update(self, trade, bar, timestamp):
        """
        Return (hit: bool, raw_fill_price: float|None).

        bar: dict-like with fields at least ['open','high','low','close'].
        raw_fill_price should be the *level* where exit occurs before spread/slippage.
        The engine will apply spread/slippage and finalize the trade.
        """
        raise NotImplementedError


class FixedStop(ExitRule):
    """
    Static stop at trade.sl_price (set by RiskManager at entry or later by other rules).
    We treat stop as 'touch' on intrabar extremes.
    """
    def update(self, trade, bar, timestamp):
        sl = getattr(trade, "sl_price", None)
        if sl is None:
            return False, None

        # Long: stop triggers if LOW <= SL; Short: if HIGH >= SL
        if trade.side == 1:
            if bar['low'] <= sl:
                return True, float(sl)
        else:
            if bar['high'] >= sl:
                return True, float(sl)
        return False, None


class TrailingStop(ExitRule):
    """
    Classic trailing stop based on max favorable excursion using intrabar extremes.
    """
    def __init__(self, trail_pips, pip_size):
        self.trail_pips = trail_pips
        self.pip_size = pip_size

    def update(self, trade, bar, timestamp):
        # Update MFE using intrabar extremes
        if trade.side == 1:
            trade.max_favorable_price = max(getattr(trade, "max_favorable_price", trade.entry_price), bar['high'])
            trail_price = trade.max_favorable_price - self.trail_pips * self.pip_size
            # Always move stop only in favorable direction
            current_sl = getattr(trade, "sl_price", None)
            trade.sl_price = max(current_sl, trail_price) if current_sl is not None else trail_price
            # Touch?
            if bar['low'] <= trade.sl_price:
                return True, float(trade.sl_price)
        else:
            trade.max_favorable_price = min(getattr(trade, "max_favorable_price", trade.entry_price), bar['low'])
            trail_price = trade.max_favorable_price + self.trail_pips * self.pip_size
            current_sl = getattr(trade, "sl_price", None)
            trade.sl_price = min(current_sl, trail_price) if current_sl is not None else trail_price
            if bar['high'] >= trade.sl_price:
                return True, float(trade.sl_price)
        return False, None


class BreakEvenStepStop(ExitRule):
    """
    Move SL to BE at trigger_pips, then step every step_pips in favorable direction.
    Uses intrabar extremes to both *activate* BE and detect SL touches.
    """
    def __init__(self, trigger_pips, step_pips, pip_size, commission=0.0, overnight_fee=0.0):
        self.trigger_pips = trigger_pips
        self.step_pips = step_pips
        self.pip_size = pip_size
        self.commission = commission
        self.overnight_fee = overnight_fee

    def update(self, trade, bar, timestamp):
        # Convert fees to pips (approx) to set true break-even
        # (commission+overnight) per trade divided by per-lot pip value at entry
        # Avoid DivZero if lot_size is borked (open trades are sized, so this is just in case)
        denom = max(trade.lot_size * (trade.entry_price * self.pip_size), 1e-12)
        fees_in_pips = (self.commission + self.overnight_fee) / denom

        sl = getattr(trade, "sl_price", None)

        if trade.side == 1:
            # Use HIGH to evaluate how far we got this bar
            profit_pips_max = (bar['high'] - trade.entry_price) / self.pip_size
            if profit_pips_max >= self.trigger_pips:
                be_price = trade.entry_price + fees_in_pips * self.pip_size
                if sl is None or sl < be_price:
                    trade.sl_price = be_price
                else:
                    steps = int((profit_pips_max - self.trigger_pips) / self.step_pips)
                    step_price = be_price + steps * self.step_pips * self.pip_size
                    trade.sl_price = max(sl, step_price)
            # Touch check with intrabar LOW
            if getattr(trade, "sl_price", None) is not None and bar['low'] <= trade.sl_price:
                return True, float(trade.sl_price)

        else:
            profit_pips_max = (trade.entry_price - bar['low']) / self.pip_size
            if profit_pips_max >= self.trigger_pips:
                be_price = trade.entry_price - fees_in_pips * self.pip_size
                if sl is None or sl > be_price:
                    trade.sl_price = be_price
                else:
                    steps = int((profit_pips_max - self.trigger_pips) / self.step_pips)
                    step_price = be_price - steps * self.step_pips * self.pip_size
                    trade.sl_price = min(sl, step_price)
            if getattr(trade, "sl_price", None) is not None and bar['high'] >= trade.sl_price:
                return True, float(trade.sl_price)

        return False, None
