"""MPS-3 pricing engine: deterministic fee quotes for OPEN parking tickets.

Given an OPEN ticket's entry time and the active pricing configuration, the
checkout service computes the amount due from entry to the calculation time.
Fees apply the grace period, round every started hour up to a full hour, and
cap the charge per independent 24-hour block. The engine is pure: the clock
and ticket lookup are injected so callers and tests can pin exact times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from .config import ParkingConfiguration
from .ticket import STATUS_OPEN, ParkingTicket


class TicketNotFoundError(Exception):
    """Raised when a quote is requested for an unknown ticket."""


class TicketStateError(Exception):
    """Raised when a quote is requested for a non-OPEN ticket."""


@dataclass(frozen=True)
class FeeQuote:
    """A deterministic fee quote for an OPEN parking ticket.

    Attributes:
        ticket_id: Ticket being quoted.
        entry_time: Ticket entry timestamp (timezone-aware UTC).
        quoted_at: Server clock at quote time (timezone-aware UTC).
        duration: Elapsed time formatted as an ISO-8601 duration string.
        duration_seconds: Elapsed time in whole seconds.
        amount_paise: Amount due in paise (1 INR = 100 paise).
        currency: ISO currency code from the pricing configuration.
        pricing_version: Configuration version that produced this quote.
    """

    ticket_id: str
    entry_time: datetime
    quoted_at: datetime
    duration: str
    duration_seconds: int
    amount_paise: int
    currency: str
    pricing_version: int


def _iso8601_duration(total_seconds: int) -> str:
    days, rem = divmod(total_seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, seconds = divmod(rem, 60)
    return f"P{days}DT{hours}H{minutes}M{seconds}S"


def _billable_paise(
    configuration: ParkingConfiguration,
    entry_time: datetime,
    now: datetime,
) -> int:
    """Sum the charge across independent 24-hour blocks after the grace period."""
    billable_start = entry_time + timedelta(minutes=configuration.grace_period_minutes)
    total = 0
    cursor = billable_start
    while cursor < now:
        block_end = min(cursor + timedelta(hours=24), now)
        block_seconds = (block_end - cursor).total_seconds()
        hours = math.ceil(block_seconds / 3_600)
        block_charge = hours * configuration.hourly_rate_paise
        if configuration.daily_cap_paise > 0:
            block_charge = min(block_charge, configuration.daily_cap_paise)
        total += block_charge
        cursor = block_end
    return total


def calculate_fee(
    configuration: ParkingConfiguration,
    entry_time: datetime,
    now: datetime,
    *,
    ticket_id: str | None = None,
) -> FeeQuote:
    """Compute the fee due from ``entry_time`` to ``now`` (timezone-aware UTC).

    Elapsed time cannot be negative. A stay that ends within the grace period
    costs zero; otherwise every started hour after grace is a full hour, and
    each 24-hour block is capped independently at the daily cap.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    total_seconds = int((now - entry_time).total_seconds())
    if total_seconds < 0:
        raise ValueError(
            "elapsed time cannot be negative "
            f"(now {now.isoformat()} before entry {entry_time.isoformat()})"
        )
    if total_seconds <= configuration.grace_period_minutes * 60:
        amount_paise = 0
    else:
        amount_paise = _billable_paise(configuration, entry_time, now)
    return FeeQuote(
        ticket_id=ticket_id or "",
        entry_time=entry_time,
        quoted_at=now,
        duration=_iso8601_duration(total_seconds),
        duration_seconds=total_seconds,
        amount_paise=amount_paise,
        currency=configuration.currency,
        pricing_version=configuration.version,
    )


@dataclass
class CheckoutService:
    """Checkout service producing fee quotes for OPEN tickets.

    Args:
        configuration: Active pricing configuration.
        get_ticket: Ticket lookup returning the ticket or ``None``.
        now: Injectable clock (defaults to ``datetime.now(timezone.utc)``).
    """

    configuration: ParkingConfiguration
    get_ticket: Callable[[str], ParkingTicket | None]
    now: Callable[[], datetime] | None = None

    def quote(self, ticket_id: str) -> FeeQuote:
        """Return a fee quote for ``ticket_id``, rejecting unknown/CLOSED tickets."""
        ticket = self.get_ticket(ticket_id)
        if ticket is None:
            raise TicketNotFoundError(f"ticket {ticket_id!r} not found")
        if ticket.status != STATUS_OPEN:
            raise TicketStateError(
                f"ticket {ticket_id!r} is {ticket.status}, expected {STATUS_OPEN}"
            )
        current = self.now() if self.now is not None else datetime.now(timezone.utc)
        return calculate_fee(
            self.configuration,
            ticket.entry_time,
            current,
            ticket_id=ticket_id,
        )
