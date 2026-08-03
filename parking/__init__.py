"""Mall Parking System entry and checkout services (MPS-2/MPS-3).

The module-level :class:`TicketService` ``default_service`` provides a small
in-memory convenience endpoint (:func:`create_ticket`, :func:`get_ticket`),
and :class:`CheckoutService` ``default_checkout`` provides :func:`quote_ticket`.
Real deployments construct their own services with a configured
:class:`ParkingConfiguration` and a durable ticket store.
"""

from __future__ import annotations

from .app.config import ParkingConfiguration
from .app.pricing import (
    CheckoutService,
    FeeQuote,
    TicketNotFoundError,
    TicketStateError,
    calculate_fee,
)
from .app.ticket import (
    STATUS_CLOSED,
    STATUS_OPEN,
    LotFullError,
    ParkingTicket,
    TicketService,
)

__all__ = [
    "STATUS_CLOSED",
    "STATUS_OPEN",
    "CheckoutService",
    "FeeQuote",
    "LotFullError",
    "ParkingConfiguration",
    "ParkingTicket",
    "TicketNotFoundError",
    "TicketService",
    "TicketStateError",
    "calculate_fee",
    "create_ticket",
    "get_ticket",
    "quote_ticket",
]

_default_configuration = ParkingConfiguration(capacity=1)
default_service = TicketService(_default_configuration)
default_checkout = CheckoutService(_default_configuration, default_service.get)


def create_ticket(
    vehicle_number: str,
    *,
    idempotency_key: str | None = None,
) -> ParkingTicket:
    """Issue an OPEN parking ticket via ``default_service``."""
    return default_service.enter(vehicle_number, idempotency_key=idempotency_key)


def get_ticket(ticket_id: str) -> ParkingTicket | None:
    """Return a ticket by ID from ``default_service``."""
    return default_service.get(ticket_id)


def quote_ticket(ticket_id: str) -> FeeQuote:
    """Return a fee quote for an OPEN ticket via ``default_checkout``."""
    return default_checkout.quote(ticket_id)
