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
    "gross_pnl", "fees", "funding_paid", "net_pnl", "equity_after",
]


def _open_signal_decision(position, desired, previous, entry_allowed):
    """Determine signal-driven orders at an eligible open."""
    close_position = position is not None and position["side"] != desired
    can_open = position is None or close_position
    open_position = bool(can_open and desired and desired != previous and entry_allowed)
    return close_position, open_position


def simulate(candles, signal, atr=None, *, timeframe="1d", settings=Execution(),
             entry_allowed=None, prior_signal=0, funding=None, tradable=None, preview_next_open=False, record_history=True):
    """Simulate one long/short target-state strategy, without borrowing.

    Signals are -1 (short), 0 (flat) or +1 (long), known at candle close.
    A change of state can enter or reverse at the next open. Protective exits
    do not re-enter on an unchanged signal. The last candle closes any position
    for reporting. Candle timestamps identify bars, not exact intrabar fills.
    Optional observed funding rates use candle close as an oracle proxy.
    Open orders are modeled before the bar's settlements. Intrabar exits
    incur the full bar's adverse payments and receive no rebates.
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
    funding_rates = None
    if funding is not None:
        if not isinstance(funding, pd.DataFrame) or not funding.index.equals(index):
            raise ValueError("Funding must be aligned to candles")
        fields = ["signed_rate", "positive_rate", "negative_rate"]
        if not set(fields) <= set(funding):
            raise ValueError("Funding requires signed, positive and negative rate columns")
        funding_rates = funding[fields].to_numpy(dtype=float)
        signed, positive, negative = funding_rates.T
        if (not np.isfinite(funding_rates).all() or (positive < 0).any() or (negative > 0).any()
                or not np.allclose(signed, positive + negative, rtol=1e-9, atol=1e-12)):
            raise ValueError("Invalid funding rates")
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
        realized = gross - fees
        capital += realized
        net = realized - position["funding_paid"]
        if record_history:
            trades.append(dict(position, exit_candle=index[i], exit_phase=phase, exit_fill=fill,
                               reason=reason, gross_pnl=gross, fees=fees, net_pnl=net,
                               equity_after=capital))
        position = None

    def apply_funding(i, intrabar_exit=False):
        nonlocal capital
        if funding_rates is None:
            return
        signed, positive, negative = funding_rates[i]
        side = position["side"]
        rate = (positive if side == 1 else -negative) if intrabar_exit else side * signed
        paid = rate * position["quantity"] * c[i]
        capital -= paid
        position["funding_paid"] += paid

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
    executable = ~synthetic.to_numpy(dtype=bool)
    if tradable is not None:
        if (not isinstance(tradable, pd.Series) or not tradable.index.equals(index)
                or not pd.api.types.is_bool_dtype(tradable) or tradable.isna().any()):
            raise ValueError("Tradable candle mask must be boolean and aligned")
        executable &= tradable.to_numpy(dtype=bool)
    real = np.flatnonzero(executable)
    last_real = real[-1] if real.size else -1
    def protective_levels(entry, side, at):
        known_atr = atr_values[at]
        stop = entry - side * settings.stop_atr * known_atr if settings.stop_atr is not None else None
        target = entry + side * settings.target_atr * known_atr if settings.target_atr is not None else None
        return stop, target

    closing_position = None
    previous = pending = prior_signal
    signal_at = 0
    for i in range(len(index)):
        # Keep the signal, filter and ATR snapshot while the open is untradable.
        desired = states[i-1] if i else 0
        if i and desired != pending:
            pending, signal_at = desired, i-1
        if not executable[i]:
            # Non-executable candles still mark open positions and accrue funding.
            if position is not None:
                apply_funding(i)
            equity = capital
            if position is not None:
                _, gross, fees = value_at(c[i])
                equity += gross - fees
            if equity <= 0:
                raise ValueError("Collateral exhausted; venue liquidation is not modelled")
            if record_history:
                marks.append(equity)
            continue
        if position is not None:
            side, stop, target = position["side"], position["stop"], position["target"]
            if stop is not None and side * (o[i]-stop) <= 0:
                exit_trade(i, o[i], "stop_gap", "open")
            elif target is not None and side * (o[i]-target) >= 0:
                exit_trade(i, o[i], "target_gap", "open")
        signal_exit, signal_entry = _open_signal_decision(position, desired, previous, allowed[signal_at])
        if signal_exit:
            exit_trade(i, o[i], "signal_exit", "open")
        if capital <= 0:
            raise ValueError("Collateral exhausted; venue liquidation is not modelled")
        if signal_entry:
            entry = o[i] * (1 + desired * slip)
            known_atr = atr_values[signal_at]
            if (settings.stop_atr is not None or settings.target_atr is not None) and (
                    not np.isfinite(known_atr) or known_atr <= 0):
                raise ValueError(f"Entry at {index[i]} requires positive ATR from the signal candle")
            stop, target = protective_levels(entry, desired, signal_at)
            # Consume the signal even when its protective order cannot be placed.
            signal_entry = all(level is None or level > 0 for level in (stop, target))
        if signal_entry:
            # Reserve entry fees inside the allocation cap. Short proceeds are not reinvested.
            allocation = min(settings.fraction, settings.max_exposure) if settings.sizing == "fixed" else settings.max_exposure
            quantity = capital * allocation / (entry * (1 + fee))
            if settings.sizing == "atr_risk":
                stop_fill = stop * (1 - desired * slip)
                loss_per_unit = desired * (entry - stop_fill) + fee * (entry + stop_fill)
                quantity = min(quantity, capital * settings.risk_fraction / loss_per_unit)
            position = dict(signal_candle=index[signal_at], entry_candle=index[i], side=int(desired),
                            quantity=quantity, entry_fill=entry, stop=stop, target=target, funding_paid=0.)
            # A very tight stop can already be breached at the raw open after entry slippage.
            if stop is not None and desired * (o[i]-stop) <= 0:
                exit_trade(i, o[i], "stop_gap", "open")
        if position is not None:
            side, stop, target = position["side"], position["stop"], position["target"]
            adverse, favourable = (l[i], h[i]) if side == 1 else (h[i], l[i])
            # OHLC cannot tell which intrabar touch came first; take the stop.
            if stop is not None and side * (adverse-stop) <= 0:
                apply_funding(i, intrabar_exit=True)
                exit_trade(i, stop, "stop", "intrabar")
            elif target is not None and side * (favourable-target) >= 0:
                apply_funding(i, intrabar_exit=True)
                exit_trade(i, target, "target", "intrabar")
            else:
                apply_funding(i)
                if i == last_real:
                    closing_position = position.copy()
                    exit_trade(i, c[i], "end_of_data", "close")
        equity = capital
        if position is not None:
            _, gross, fees = value_at(c[i])
            equity += gross - fees
        if equity <= 0:
            raise ValueError("Collateral exhausted; venue liquidation is not modelled")
        if record_history:
            marks.append(equity)
        if i:
            previous = desired

    orders = []
    if preview_next_open:
        desired = int(states[-1])
        at = len(index) - 1 if desired != pending else signal_at
        scheduled = index[-1] + duration
        position = closing_position
        signal_exit, signal_entry = _open_signal_decision(position, desired, previous, allowed[at])
        if signal_exit:
            side = position["side"]
            orders.append(dict(event="exit", action="sell" if side == 1 else "buy",
                               position_side="long" if side == 1 else "short",
                               signal_candle_utc=index[at], execution_candle_utc=scheduled,
                               stop_loss=position["stop"], take_profit=position["target"],
                               quantity=position["quantity"], reason="signal_exit"))
        if signal_entry:
            # The next opening price is unknown. The latest close is a reference
            # for estimated protective levels, not an observed or simulated fill.
            reference = c[-1] * (1 + desired * slip)
            stop, target = protective_levels(reference, desired, at)
            if all(level is None or level > 0 for level in (stop, target)):
                orders.append(dict(event="entry", action="buy" if desired == 1 else "sell",
                                   position_side="long" if desired == 1 else "short",
                                   signal_candle_utc=index[at], execution_candle_utc=scheduled,
                                   reference_entry_price=reference, stop_loss=stop, take_profit=target,
                                   reason="signal_entry"))

    if not record_history:
        return {"next_open_orders": orders}
    return dict(trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
                equity=pd.Series(marks, index=index, name="equity"), next_open_orders=orders)
