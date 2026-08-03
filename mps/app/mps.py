"""Core Master Production Schedule (MPS) planning engine (MPS-1).

Given per-period gross requirements (forecast plus customer orders), scheduled
receipts, opening on-hand inventory, safety stock, and an optional lot size,
the engine computes a feasible production plan of planned order receipts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Demand:
    """Gross requirements for a single planning period.

    Attributes:
        period: One-based planning period (week, month, ...).
        gross_requirement: Quantity of demand to satisfy in the period.
        scheduled_receipt: Quantity already committed to arrive in the period.
    """

    period: int
    gross_requirement: float
    scheduled_receipt: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.period, int) or self.period < 1:
            raise ValueError(
                f"period must be a positive integer, got {self.period!r}"
            )
        if self.gross_requirement < 0:
            raise ValueError(
                f"gross_requirement must be >= 0, got {self.gross_requirement}"
            )
        if self.scheduled_receipt < 0:
            raise ValueError(
                f"scheduled_receipt must be >= 0, got {self.scheduled_receipt}"
            )


@dataclass(frozen=True)
class PlannedLine:
    """One period of the computed MPS plan."""

    period: int
    gross_requirement: float
    scheduled_receipt: float
    opening_on_hand: float
    net_requirement: float
    planned_order_receipt: float
    projected_on_hand: float


def _round_qty(value: float) -> float:
    return round(value, 6)


def _lot_receipt(net: float, lot_size: float | None) -> float:
    if net <= 0:
        return 0.0
    if lot_size is None:
        return net
    return math.ceil(net / lot_size) * lot_size


def plan_mps(
    demands: Sequence[Demand],
    *,
    starting_on_hand: float,
    lot_size: float | None = None,
    safety_stock: float = 0.0,
) -> list[PlannedLine]:
    """Compute the Master Production Schedule for a sequence of demands.

    For each period the projected on-hand (POH) carried from the previous
    period is the starting point. Any gross requirement that cannot be covered
    by opening on-hand plus scheduled receipts above safety stock becomes a net
    requirement, planned as an order receipt rounded up to ``lot_size``.

    Args:
        demands: Demands in strictly ascending period order.
        starting_on_hand: On-hand quantity at the start of the first period.
        lot_size: Optional fixed production lot size (receipts are rounded up
            to a multiple of it). When ``None``, receipts match net
            requirements exactly.
        safety_stock: Reserve inventory that demand may not consume.

    Returns:
        One :class:`PlannedLine` per demand, in the same order.

    Raises:
        ValueError: If periods are not strictly ascending or any quantity is
            negative.
    """
    if starting_on_hand < 0:
        raise ValueError(
            f"starting_on_hand must be >= 0, got {starting_on_hand}"
        )
    if safety_stock < 0:
        raise ValueError(f"safety_stock must be >= 0, got {safety_stock}")
    if lot_size is not None and lot_size <= 0:
        raise ValueError(f"lot_size must be > 0, got {lot_size}")

    plan: list[PlannedLine] = []
    on_hand = starting_on_hand
    previous_period = 0
    for demand in demands:
        if demand.period <= previous_period:
            raise ValueError(
                "periods must be strictly ascending; "
                f"{demand.period} follows {previous_period}"
            )
        previous_period = demand.period

        opening = on_hand
        available = opening + demand.scheduled_receipt - safety_stock
        net_requirement = max(0.0, demand.gross_requirement - available)
        receipt = _lot_receipt(net_requirement, lot_size)
        on_hand = opening + demand.scheduled_receipt + receipt - demand.gross_requirement
        plan.append(
            PlannedLine(
                period=demand.period,
                gross_requirement=demand.gross_requirement,
                scheduled_receipt=demand.scheduled_receipt,
                opening_on_hand=_round_qty(opening),
                net_requirement=_round_qty(net_requirement),
                planned_order_receipt=_round_qty(receipt),
                projected_on_hand=_round_qty(on_hand),
            )
        )
    return plan
