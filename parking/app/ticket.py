"""Core MPS-2 engine: issuing OPEN parking tickets at the entry gate.

One entry transaction creates exactly one unique OPEN ticket, bumps the
occupied-space counter exactly once, and is idempotent under a client-supplied
idempotency key. Entry is rejected with :class:`LotFullError` when occupancy
equals configured capacity. Ticket creation and the occupancy increment are
atomic in :meth:`TicketService.enter` — a failed or full entry leaves both the
store and the occupancy counter unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .config import ParkingConfiguration

STATUS_OPEN = "OPEN"
STATUS_CLOSED = "CLOSED"


class LotFullError(Exception):
    """Raised when no parking capacity remains for an entry request."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_vehicle_number(raw: str) -> str:
    """Trim surrounding whitespace and uppercase a vehicle number."""
    if not isinstance(raw, str):
        raise ValueError(f"vehicle_number must be a string, got {type(raw).__name__}")
    normalized = raw.strip().upper()
    if not normalized:
        raise ValueError("vehicle_number must not be blank")
    return normalized


@dataclass(frozen=True)
class ParkingTicket:
    """An OPEN parking ticket issued at the entry gate.

    Attributes:
        ticket_id: Unique external ticket identifier.
        vehicle_number: Normalized (trimmed, uppercased) vehicle number.
        entry_time: Server-generated entry timestamp (timezone-aware UTC).
        status: Ticket state; ``OPEN`` for newly issued tickets.
    """

    ticket_id: str
    vehicle_number: str
    entry_time: datetime
    status: str = STATUS_OPEN

    def __post_init__(self) -> None:
        if not self.ticket_id:
            raise ValueError("ticket_id must not be empty")


class TicketStore:
    """In-memory parking-ticket store keyed by ticket ID and idempotency key."""

    def __init__(self) -> None:
        self._by_id: dict[str, ParkingTicket] = {}
        self._by_key: dict[str, ParkingTicket] = {}

    def by_id(self, ticket_id: str) -> ParkingTicket | None:
        return self._by_id.get(ticket_id)

    def by_key(self, idempotency_key: str) -> ParkingTicket | None:
        return self._by_key.get(idempotency_key)

    def add(self, ticket: ParkingTicket, *, idempotency_key: str | None) -> None:
        if ticket.ticket_id in self._by_id:
            raise ValueError(f"duplicate ticket_id {ticket.ticket_id!r}")
        if idempotency_key is not None and idempotency_key in self._by_key:
            raise ValueError(f"duplicate idempotency_key {idempotency_key!r}")
        self._by_id[ticket.ticket_id] = ticket
        if idempotency_key is not None:
            self._by_key[idempotency_key] = ticket


class Occupancy:
    """Monotonic occupied-space counter shared by entry and occupancy services."""

    def __init__(self) -> None:
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def increment(self) -> None:
        self._count += 1


class TicketService:
    """Entry-gate service that issues OPEN tickets atomically.

    Args:
        configuration: Active parking configuration providing capacity.
        now: Injectable clock returning the server entry time.
        new_ticket_id: Injectable unique-ID generator (defaults to ``T-1``,
            ``T-2``, ... consumed only for successfully issued tickets).
    """

    def __init__(
        self,
        configuration: ParkingConfiguration,
        *,
        now: Callable[[], datetime] | None = None,
        new_ticket_id: Callable[[], str] | None = None,
    ) -> None:
        self._configuration = configuration
        self._store = TicketStore()
        self._occupancy = Occupancy()
        self._now = now if now is not None else _utcnow
        self._issued = 0
        self._new_ticket_id = (
            new_ticket_id if new_ticket_id is not None else self._next_ticket_id
        )

    def _next_ticket_id(self) -> str:
        self._issued += 1
        return f"T-{self._issued}"

    @property
    def occupancy(self) -> int:
        """Current number of occupied spaces."""
        return self._occupancy.count

    def enter(
        self,
        vehicle_number: str,
        *,
        idempotency_key: str | None = None,
    ) -> ParkingTicket:
        """Process one entry request and issue an OPEN ticket.

        Idempotent: a repeat request carrying the same idempotency key returns
        the existing ticket without touching occupancy. Raises
        :class:`LotFullError` when occupancy already equals capacity.
        """
        vehicle = normalize_vehicle_number(vehicle_number)
        key = self._normalize_key(idempotency_key)
        if key is not None:
            existing = self._store.by_key(key)
            if existing is not None:
                return existing
        if self._occupancy.count >= self._configuration.capacity:
            raise LotFullError(
                f"lot is full (occupancy {self._occupancy.count} of capacity "
                f"{self._configuration.capacity})"
            )
        ticket = ParkingTicket(
            ticket_id=self._new_ticket_id(),
            vehicle_number=vehicle,
            entry_time=self._now(),
        )
        self._store.add(ticket, idempotency_key=key)
        self._occupancy.increment()
        return ticket

    def get(self, ticket_id: str) -> ParkingTicket | None:
        """Return a previously issued ticket by ID, or ``None``."""
        return self._store.by_id(ticket_id)

    @staticmethod
    def _normalize_key(idempotency_key: str | None) -> str | None:
        if idempotency_key is None:
            return None
        stripped = idempotency_key.strip()
        return stripped or None
