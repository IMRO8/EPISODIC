"""Command-line interface for the MPS planning engine (MPS-1)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from .mps import Demand, plan_mps


def _load_demands(payload: Any) -> list[Demand]:
    """Parse a JSON payload into demands.

    Accepts either a list of numbers (one gross requirement per period) or a
    list of objects with ``period`` and ``gross_requirement`` keys and an
    optional ``scheduled_receipt`` key.
    """
    if isinstance(payload, list) and all(
        isinstance(item, (int, float)) for item in payload
    ):
        return [
            Demand(period=index + 1, gross_requirement=float(value))
            for index, value in enumerate(payload)
        ]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        try:
            return [
                Demand(
                    period=int(item["period"]),
                    gross_requirement=float(item["gross_requirement"]),
                    scheduled_receipt=float(item.get("scheduled_receipt", 0.0)),
                )
                for item in payload
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "each demand object needs 'period' and 'gross_requirement'"
            ) from exc
    raise ValueError(
        "demands must be a JSON list of numbers or of objects with 'period' and "
        "'gross_requirement'"
    )


def _print_table(plan: list) -> None:
    header = ("period", "gross", "sched", "opening", "net", "receipt", "POH")
    print(
        f"{header[0]:>6} {header[1]:>8} {header[2]:>8} {header[3]:>8} "
        f"{header[4]:>8} {header[5]:>8} {header[6]:>8}"
    )
    for line in plan:
        print(
            f"{line.period:>6} {line.gross_requirement:>8g} "
            f"{line.scheduled_receipt:>8g} {line.opening_on_hand:>8g} "
            f"{line.net_requirement:>8g} {line.planned_order_receipt:>8g} "
            f"{line.projected_on_hand:>8g}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mps.app",
        description="Compute a Master Production Schedule (MPS-1).",
    )
    parser.add_argument(
        "--on-hand",
        type=float,
        required=True,
        help="On-hand inventory at the start of the first period",
    )
    parser.add_argument(
        "--lot-size",
        type=float,
        default=None,
        help="Fixed production lot size (receipts round up to a multiple)",
    )
    parser.add_argument(
        "--safety-stock",
        type=float,
        default=0.0,
        help="Reserve inventory that demand may not consume",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the plan as JSON instead of a table",
    )
    parser.add_argument(
        "demands",
        nargs="?",
        type=argparse.FileType("r", encoding="utf-8"),
        default=sys.stdin,
        help="JSON file with the demands (defaults to stdin)",
    )
    args = parser.parse_args(argv)

    try:
        payload = json.load(args.demands)
        demands = _load_demands(payload)
        plan = plan_mps(
            demands,
            starting_on_hand=args.on_hand,
            lot_size=args.lot_size,
            safety_stock=args.safety_stock,
        )
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        json.dump([asdict(line) for line in plan], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_table(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
