# exits_fast.py
class ExitRule:
    def update(self, trade, bar, timestamp):
        """
        Return (hit: bool, raw_fill_price: float|None).
        FAST version: uses close-only checks (no OHLC touch).
        """
        raise NotImplementedError


class FixedStop(ExitRule):
    """ Close-only stop check (no intrabar touch). """
    def __init__(self, sl_pips, pip_size):
        self.sl_pips = sl_pips
        self.pip_size = pip_size

    def update(self, trade, bar, timestamp):
        sl = getattr(trade, "sl_price", None)
        if sl is None:
            return False, None
        px = float(bar["close"])
        if trade.side == 1 and px <= sl:
            return True, sl  # fill at stop level (engine will not add spread in fast mode)
        if trade.side == -1 and px >= sl:
            return True, sl
        return False, None


class TrailingStop(ExitRule):
    """ Trail using close price only (no intrabar extremes). """
    def __init__(self, trail_pips, pip_size):
        self.trail_pips = trail_pips
        self.pip_size = pip_size

    def update(self, trade, bar, timestamp):
        px = float(bar["close"])
        if trade.side == 1:
            trade.max_favorable_price = max(getattr(trade, "max_favorable_price", trade.entry_price), px)
            trail_price = trade.max_favorable_price - self.trail_pips * self.pip_size
            current_sl = getattr(trade, "sl_price", None)
            trade.sl_price = max(current_sl, trail_price) if current_sl is not None else trail_price
            if px <= trade.sl_price:
                return True, trade.sl_price
        else:
            trade.max_favorable_price = min(getattr(trade, "max_favorable_price", trade.entry_price), px)
            trail_price = trade.max_favorable_price + self.trail_pips * self.pip_size
            current_sl = getattr(trade, "sl_price", None)
            trade.sl_price = min(current_sl, trail_price) if current_sl is not None else trail_price
            if px >= trade.sl_price:
                return True, trade.sl_price
        return False, None


class BreakEvenStepStop(ExitRule):
    """ Break-even + steps using close-only progression (faster). """
    def __init__(self, trigger_pips, step_pips, pip_size, commission=0.0, overnight_fee=0.0):
        self.trigger_pips = trigger_pips
        self.step_pips = step_pips
        self.pip_size = pip_size
        self.commission = commission
        self.overnight_fee = overnight_fee

    def update(self, trade, bar, timestamp):
        px = float(bar["close"])
        denom = max(trade.lot_size * (trade.entry_price * self.pip_size), 1e-12)
        fees_in_pips = (self.commission + self.overnight_fee) / denom

        sl = getattr(trade, "sl_price", None)

        if trade.side == 1:
            profit_pips = (px - trade.entry_price) / self.pip_size
            if profit_pips >= self.trigger_pips:
                be_price = trade.entry_price + fees_in_pips * self.pip_size
                if sl is None or sl < be_price:
                    trade.sl_price = be_price
                else:
                    steps = int((profit_pips - self.trigger_pips) / self.step_pips)
                    step_price = be_price + steps * self.step_pips * self.pip_size
                    trade.sl_price = max(sl, step_price)
            if getattr(trade, "sl_price", None) is not None and px <= trade.sl_price:
                return True, trade.sl_price

        else:
            profit_pips = (trade.entry_price - px) / self.pip_size
            if profit_pips >= self.trigger_pips:
                be_price = trade.entry_price - fees_in_pips * self.pip_size
                if sl is None or sl > be_price:
                    trade.sl_price = be_price
                else:
                    steps = int((profit_pips - self.trigger_pips) / self.step_pips)
                    step_price = be_price - steps * self.step_pips * self.pip_size
                    trade.sl_price = min(sl, step_price)
            if getattr(trade, "sl_price", None) is not None and px >= trade.sl_price:
                return True, trade.sl_price

        return False, None
