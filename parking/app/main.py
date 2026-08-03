"""Command-line interface for the parking entry and checkout services.

Accepts a JSON request from stdin or a file — a single object or a list of
objects processed sequentially through one service instance. Entry requests
``{"vehicle_number": ..., "idempotency_key": ...}`` issue OPEN tickets (MPS-2).
Quote requests ``{"action": "quote", "ticket_id": ...}`` return a fee quote
(MPS-3). Emits the issued ticket or quote as JSON, or an error result when the
lot is full (``LOT_FULL``) or the ticket is unknown/closed
(``TICKET_NOT_FOUND``/``TICKET_CLOSED``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from .config import ConfigStore, ParkingConfiguration
from .pricing import CheckoutService, TicketNotFoundError, TicketStateError
from .ticket import LotFullError, TicketService


def _parse_request(item: Any) -> dict:
    if not isinstance(item, dict):
        raise ValueError("each request must be a JSON object")
    action = item.get("action", "entry")
    if action == "entry":
        if "vehicle_number" not in item:
            raise ValueError("each entry request must include 'vehicle_number'")
        return {
            "action": "entry",
            "vehicle_number": item["vehicle_number"],
            "idempotency_key": item.get("idempotency_key"),
        }
    if action == "quote":
        ticket_id = item.get("ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise ValueError("each quote request must include a non-blank 'ticket_id'")
        return {"action": "quote", "ticket_id": ticket_id.strip()}
    raise ValueError(f"unknown request action {action!r}")


def _load_requests(payload: Any) -> tuple[list[dict], bool]:
    """Normalize a single request object or a list of them.

    Returns the parsed requests and whether the input was a batch (list).
    """
    if isinstance(payload, dict):
        return [_parse_request(payload)], False
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return [_parse_request(item) for item in payload], True
    raise ValueError(
        "payload must be a JSON object or a list of objects with "
        "'vehicle_number' (entry) or 'action': 'quote' (checkout)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m parking.app",
        description="Issue a parking ticket (MPS-2) or quote its fee (MPS-3).",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        required=True,
        help="Parking lot capacity (total number of bays)",
    )
    parser.add_argument(
        "--grace-minutes",
        type=int,
        default=0,
        help="Free minutes before metering starts (default: 0)",
    )
    parser.add_argument(
        "--hourly-rate-paise",
        type=int,
        default=0,
        help="Hourly rate in paise (default: 0)",
    )
    parser.add_argument(
        "--daily-cap-paise",
        type=int,
        default=0,
        help="Maximum charge per 24-hour block in paise (default: 0)",
    )
    parser.add_argument(
        "--currency",
        default="INR",
        help="ISO currency code for quotes (default: INR)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Pricing-rule version reported in quotes (default: 1)",
    )
    parser.add_argument(
        "request",
        nargs="?",
        type=argparse.FileType("r", encoding="utf-8"),
        default=sys.stdin,
        help=(
            "JSON file with entry {'vehicle_number', 'idempotency_key'} or "
            "quote {'action': 'quote', 'ticket_id'} requests (defaults to stdin)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        payload = json.load(args.request)
        requests, is_batch = _load_requests(payload)
        configuration = ParkingConfiguration(
            capacity=args.capacity,
            grace_period_minutes=args.grace_minutes,
            hourly_rate_paise=args.hourly_rate_paise,
            daily_cap_paise=args.daily_cap_paise,
            currency=args.currency,
            version=args.version,
        )
        service = TicketService(ConfigStore(configuration).active())
        checkout = CheckoutService(configuration, service.get)
        results: list[dict] = []
        for request in requests:
            try:
                if request["action"] == "quote":
                    results.append(asdict(checkout.quote(request["ticket_id"])))
                else:
                    ticket = service.enter(
                        request["vehicle_number"],
                        idempotency_key=request["idempotency_key"],
                    )
                    results.append(asdict(ticket))
            except LotFullError as exc:
                results.append({"error": "LOT_FULL", "detail": str(exc)})
            except TicketNotFoundError as exc:
                results.append({"error": "TICKET_NOT_FOUND", "detail": str(exc)})
            except TicketStateError as exc:
                results.append({"error": "TICKET_CLOSED", "detail": str(exc)})
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if is_batch:
        json.dump(results, sys.stdout, default=str, indent=2)
        sys.stdout.write("\n")
        return 0
    result = results[0]
    json.dump(result, sys.stdout, default=str, indent=2)
    sys.stdout.write("\n")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
