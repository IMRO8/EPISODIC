"""Parking configuration slice shared by the entry service (MPS-1/MPS-2).

MPS-1 defines the full configuration (capacity, grace period, hourly rate,
daily cap, lost-ticket fee, currency, and an auditable version). The entry
service (MPS-2) only depends on ``capacity``; the remaining fields are
placeholders so a full MPS-1 implementation can plug in without API churn.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParkingConfiguration:
    """Validated parking capacity and fee configuration.

    Attributes:
        capacity: Total number of parking bays (must be a positive integer).
        grace_period_minutes: Free minutes before metering starts.
        hourly_rate_paise: Hourly rate in paise (1 INR = 100 paise).
        daily_cap_paise: Maximum daily charge in paise.
        lost_ticket_fee_paise: Flat fee for a lost ticket in paise.
        currency: ISO currency code (money uses INR with two decimals).
        version: Monotonic configuration version; updates create new versions.
    """

    capacity: int
    grace_period_minutes: int = 0
    hourly_rate_paise: int = 0
    daily_cap_paise: int = 0
    lost_ticket_fee_paise: int = 0
    currency: str = "INR"
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or self.capacity < 1:
            raise ValueError(
                f"capacity must be a positive integer, got {self.capacity!r}"
            )
        if self.grace_period_minutes < 0:
            raise ValueError("grace_period_minutes must be >= 0")
        if self.hourly_rate_paise < 0:
            raise ValueError("hourly_rate_paise must be >= 0")
        if self.daily_cap_paise < 0:
            raise ValueError("daily_cap_paise must be >= 0")
        if self.lost_ticket_fee_paise < 0:
            raise ValueError("lost_ticket_fee_paise must be >= 0")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")


class ConfigStore:
    """Holds the single active parking configuration.

    Implements the MPS-1 rule that the active configuration is retrievable by
    entry, pricing, payment, and occupancy services via ``active()``.
    """

    def __init__(self, configuration: ParkingConfiguration) -> None:
        self._configuration = configuration

    def active(self) -> ParkingConfiguration:
        return self._configuration
