"""Single-asset, next-open execution for completed, contiguous candles."""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from storage import COLUMNS
from timeframes import get_timeframe


@dataclass(frozen=True)
class Execution:
    initial_capital: float = 1000.0
    fee: float = 0.001
    slippage: float = 0.0005
    sizing: str = "fixed"
    fraction: float = 0.10
    risk_fraction: float = 0.01
    max_exposure: float = 1.0
    stop_atr: float | None = 2.0
    target_atr: float | None = 3.0

    def __post_init__(self):
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive and finite")
        if self.sizing not in {"fixed", "atr_risk"}:
            raise ValueError("Sizing must be fixed or atr_risk")
        if not all(0 <= x < 1 for x in (self.fee, self.slippage)):
            raise ValueError("Fee and slippage must be fractions in [0, 1)")
        if not all(0 < x <= 1 for x in (self.fraction, self.risk_fraction, self.max_exposure)):
            raise ValueError("Sizing fractions and exposure cap must be in (0, 1]")
        for multiple in (self.stop_atr, self.target_atr):
            if multiple is not None and (not math.isfinite(multiple) or multiple <= 0):
                raise ValueError("ATR multiples must be positive or None")
        if self.sizing == "atr_risk" and self.stop_atr is None:
            raise ValueError("ATR risk sizing requires a stop")


TRADE_COLUMNS = [
    "signal_candle", "entry_candle", "exit_candle", "exit_phase", "side",
    "quantity", "entry_fill", "exit_fill", "stop", "target", "reason",
    "gross_pnl", "fees", "net_pnl", "equity_after",
]


def simulate(candles, signal, atr=None, *, timeframe="1d", settings=Execution(),
             entry_allowed=None, prior_signal=0):
    """Simulate one long/short target-state strategy, without funding or borrowing.

    Signals are -1 (short), 0 (flat) or +1 (long), known at candle close.
    A change of state can enter or reverse at the next open. Protective exits
    do not re-enter on an unchanged signal. The last candle closes any position
    for reporting. Candle timestamps identify bars, not exact intrabar fills.
    """
    duration = get_timeframe(timeframe).duration
    index = candles.index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing or not len(index)):
        raise ValueError("Supply nonempty candles with unique chronological timestamps")
    utc_index = pd.to_datetime(index, utc=True)
    if (not utc_index.equals(utc_index.floor(duration))
            or not (utc_index[1:] - utc_index[:-1] == duration).all()):
        raise ValueError("Candle gap or off-grid timestamp: cannot assume the next scheduled open")
    if not signal.index.equals(index) or not signal.isin([-1, 0, 1]).all():
        raise ValueError("Signals must align with candles and contain only -1, 0 or 1")
    prices = candles[COLUMNS].apply(pd.to_numeric, errors="coerce")
    o, h, l, c, v = prices.to_numpy().T
    if (not np.isfinite(prices).all().all() or (prices.iloc[:, :4] <= 0).any().any()
            or (v < 0).any() or (h < np.maximum(o, c)).any()
            or (l > np.minimum(o, c)).any() or (h < l).any()):
        raise ValueError("Invalid OHLCV values")
    if atr is not None and not atr.index.equals(index):
        raise ValueError("ATR must align with candles")
    atr_values = atr.to_numpy(dtype=float) if atr is not None else np.full(len(index), np.nan)
    states = signal.to_numpy(dtype=int)
    fee, slip = settings.fee, settings.slippage
    capital, position = settings.initial_capital, None
    trades, marks = [], []

    def value_at(raw_price):
        """Mark net of estimated exit costs using signed price P&L."""
        side, qty, entry = position["side"], position["quantity"], position["entry_fill"]
        fill = raw_price * (1 - side * slip)
        gross = side * qty * (fill - entry)
        fees = fee * qty * (entry + fill)
        return fill, gross, fees

    def exit_trade(i, price, reason, phase):
        nonlocal capital, position
        fill, gross, fees = value_at(price)
        net = gross - fees
        capital += net
        trades.append(dict(position, exit_candle=index[i], exit_phase=phase, exit_fill=fill,
                           reason=reason, gross_pnl=gross, fees=fees, net_pnl=net,
                           equity_after=capital))
        position = None

    if prior_signal not in (-1, 0, 1):
        raise ValueError("Prior signal must be -1, 0 or 1")
    if entry_allowed is not None:
        if not entry_allowed.index.equals(index) or not pd.api.types.is_bool_dtype(entry_allowed) or entry_allowed.isna().any():
            raise ValueError("Entry filter must contain boolean values aligned to candles")
        allowed = entry_allowed.to_numpy(dtype=bool)
    else:
        allowed = np.ones(len(index), dtype=bool)

    synthetic = candles["synthetic"] if "synthetic" in candles else pd.Series(False, index=index)
    if not pd.api.types.is_bool_dtype(synthetic) or synthetic.isna().any():
        raise ValueError("Synthetic candle markers must be boolean")
    synthetic = synthetic.to_numpy(dtype=bool)
    real = np.flatnonzero(~synthetic)
    last_real = real[-1] if real.size else -1
    previous = pending = prior_signal
    signal_at = 0
    for i in range(len(index)):
        # Keep the signal, filter and ATR snapshot while the open is untradable.
        desired = states[i-1] if i else 0
        if i and desired != pending:
            pending, signal_at = desired, i-1
        if synthetic[i]:
            # The carried close marks an existing position. It cannot fill any order.
            equity = capital
            if position is not None:
                _, gross, fees = value_at(c[i])
                equity += gross - fees
            marks.append(equity)
            continue
        if position is not None:
            side, stop, target = position["side"], position["stop"], position["target"]
            if stop is not None and side * (o[i]-stop) <= 0:
                exit_trade(i, o[i], "stop_gap", "open")
            elif target is not None and side * (o[i]-target) >= 0:
                exit_trade(i, o[i], "target_gap", "open")
            elif desired != side:
                exit_trade(i, o[i], "signal_exit", "open")
        if capital <= 0:
            raise ValueError("Collateral exhausted; venue liquidation is not modelled")
        if position is None and desired and desired != previous and allowed[signal_at]:
            entry = o[i] * (1 + desired * slip)
            known_atr = atr_values[signal_at]
            if (settings.stop_atr is not None or settings.target_atr is not None) and (
                    not np.isfinite(known_atr) or known_atr <= 0):
                raise ValueError(f"Entry at {index[i]} requires positive ATR from the signal candle")
            stop = entry - desired * settings.stop_atr * known_atr if settings.stop_atr is not None else None
            target = entry + desired * settings.target_atr * known_atr if settings.target_atr is not None else None
            if any(level is not None and level <= 0 for level in (stop, target)):
                raise ValueError(f"Protective level at {index[i]} must be positive")
            # Reserve entry fees inside the allocation cap. Short proceeds are not reinvested.
            allocation = min(settings.fraction, settings.max_exposure) if settings.sizing == "fixed" else settings.max_exposure
            quantity = capital * allocation / (entry * (1 + fee))
            if settings.sizing == "atr_risk":
                stop_fill = stop * (1 - desired * slip)
                loss_per_unit = desired * (entry - stop_fill) + fee * (entry + stop_fill)
                quantity = min(quantity, capital * settings.risk_fraction / loss_per_unit)
            position = dict(signal_candle=index[signal_at], entry_candle=index[i], side=int(desired),
                            quantity=quantity, entry_fill=entry, stop=stop, target=target)
            # A very tight stop can already be breached at the raw open after entry slippage.
            if stop is not None and desired * (o[i]-stop) <= 0:
                exit_trade(i, o[i], "stop_gap", "open")
        if position is not None:
            side, stop, target = position["side"], position["stop"], position["target"]
            adverse, favourable = (l[i], h[i]) if side == 1 else (h[i], l[i])
            # OHLC cannot tell which intrabar touch came first; take the stop.
            if stop is not None and side * (adverse-stop) <= 0:
                exit_trade(i, stop, "stop", "intrabar")
            elif target is not None and side * (favourable-target) >= 0:
                exit_trade(i, target, "target", "intrabar")
            elif i == last_real:
                exit_trade(i, c[i], "end_of_data", "close")
        equity = capital
        if position is not None:
            _, gross, fees = value_at(c[i])
            equity += gross - fees
        if equity <= 0:
            raise ValueError("Collateral exhausted; venue liquidation is not modelled")
        marks.append(equity)
        if i:
            previous = desired

    return dict(trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
                equity=pd.Series(marks, index=index, name="equity"))
