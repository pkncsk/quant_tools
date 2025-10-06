# exits.py (refactored unified API)

class BaseStop:
    """Unified interface: all stop rules must implement candidate_sl()."""
    def candidate_sl(self, trade, bar, ts):
        raise NotImplementedError

    def update(self, trade, bar, ts):
        """
        Default conservative exit check (touch = hit).
        Does not mutate trade state.
        """
        sl = self.candidate_sl(trade, bar, ts)
        if trade.side == 1 and bar["low"] <= sl:
            return True, sl
        elif trade.side == -1 and bar["high"] >= sl:
            return True, sl
        return False, None


# -------------------------------------------------------------
# 1️⃣ Fixed stop
# -------------------------------------------------------------
class FixedStop(BaseStop):
    """Fixed distance from entry, constant through trade."""
    def __init__(self, sl_pips, pip_size):
        self.sl_pips = float(sl_pips)
        self.pip_size = float(pip_size)

    def candidate_sl(self, trade, bar, ts):
        if trade.side == 1:
            return trade.entry_price - self.sl_pips * self.pip_size
        else:
            return trade.entry_price + self.sl_pips * self.pip_size


# -------------------------------------------------------------
# 2️⃣ Break-even + step stop
# -------------------------------------------------------------
class BreakEvenStepStop(BaseStop):
    """
    Move to breakeven at +1R and step further by step_pips beyond that.
    Stateless: computes level directly each bar.
    """
    def __init__(self, step_pips=0, pip_size=0.0001):
        self.step_pips = float(step_pips)
        self.pip_size = float(pip_size)

    def candidate_sl(self, trade, bar, ts):
        entry = trade.entry_price
        sl_init = trade.initial_sl_price or (
            entry - self.step_pips * self.pip_size if trade.side == 1
            else entry + self.step_pips * self.pip_size
        )

        risk_pips = abs(entry - sl_init) / self.pip_size
        px = bar["close"]

        if trade.side == 1:
            profit_pips = (px - entry) / self.pip_size
            if profit_pips < risk_pips:
                return sl_init
            else:
                steps = int((profit_pips - risk_pips) / self.step_pips) if self.step_pips > 0 else 0
                return entry + steps * self.step_pips * self.pip_size
        else:
            profit_pips = (entry - px) / self.pip_size
            if profit_pips < risk_pips:
                return sl_init
            else:
                steps = int((profit_pips - risk_pips) / self.step_pips) if self.step_pips > 0 else 0
                return entry - steps * self.step_pips * self.pip_size


# -------------------------------------------------------------
# 3️⃣ Trailing stop
# -------------------------------------------------------------
class TrailingStop(BaseStop):
    """
    Ratcheting trailing stop from most favorable price (MFE).
    - For longs: follows the highest high seen since entry.
    - For shorts: follows the lowest low seen since entry.
    - Only tightens (never loosens).
    """
    def __init__(self, trail_pips, pip_size, use_high_low=True):
        self.trail_pips = float(trail_pips)
        self.pip_size = float(pip_size)
        self.use_high_low = bool(use_high_low)

    def candidate_sl(self, trade, bar, ts):
        px = bar["close"]

        # Initialize or update most favorable price
        if not hasattr(trade, "max_favorable_price"):
            trade.max_favorable_price = trade.entry_price

        if trade.side == 1:
            # Long: highest favorable move
            obs = bar["high"] if (self.use_high_low and "high" in bar) else px
            trade.max_favorable_price = max(trade.max_favorable_price, obs)
            # compute stop from MFE
            return trade.max_favorable_price - self.trail_pips * self.pip_size

        else:
            # Short: lowest favorable move
            obs = bar["low"] if (self.use_high_low and "low" in bar) else px
            trade.max_favorable_price = min(trade.max_favorable_price, obs)
            return trade.max_favorable_price + self.trail_pips * self.pip_size



# -------------------------------------------------------------
# 4️⃣ Progressive stop (Fixed → BE → Trailing)
# -------------------------------------------------------------
class ProgressiveStop(BaseStop):
    """
    Unified lifecycle:
      ① Start fixed at initial risk (entry ± sl_pips)
      ② Move to breakeven after +1R profit
      ③ Then trail from most favorable extreme (MFE)
         but never below breakeven (long) or above (short).
    """
    def __init__(self, sl_pips, trail_pips, pip_size, use_high_low=True):
        self.sl_pips = float(sl_pips)
        self.trail_pips = float(trail_pips)
        self.pip_size = float(pip_size)
        self.use_high_low = bool(use_high_low)

    def candidate_sl(self, trade, bar, ts):
        entry = trade.entry_price
        px = bar["close"]
        risk_pips = self.sl_pips

        # --- Initialize persistent state ---
        if not hasattr(trade, "max_favorable_price"):
            trade.max_favorable_price = entry
        if not hasattr(trade, "breakeven_armed"):
            trade.breakeven_armed = False

        # --- Update most favorable move ---
        if trade.side == 1:
            obs = bar["high"] if (self.use_high_low and "high" in bar) else px
            trade.max_favorable_price = max(trade.max_favorable_price, obs)
            profit_pips = (px - entry) / self.pip_size
        else:
            obs = bar["low"] if (self.use_high_low and "low" in bar) else px
            trade.max_favorable_price = min(trade.max_favorable_price, obs)
            profit_pips = (entry - px) / self.pip_size

        # --- Phase ①: fixed stop until +1R ---
        if not trade.breakeven_armed:
            if profit_pips < risk_pips:
                # still within risk zone
                return entry - risk_pips * self.pip_size if trade.side == 1 else entry + risk_pips * self.pip_size
            else:
                # lock breakeven once 1R reached
                trade.breakeven_armed = True

        # --- Phase ②–③: breakeven baseline + trailing ---
        if trade.side == 1:
            trail_level = trade.max_favorable_price - self.trail_pips * self.pip_size
            # ensure never below breakeven
            return max(entry, trail_level)
        else:
            trail_level = trade.max_favorable_price + self.trail_pips * self.pip_size
            # ensure never above breakeven
            return min(entry, trail_level)
