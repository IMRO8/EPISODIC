from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone

from parking.app.config import ParkingConfiguration
from parking.app.main import _load_requests, main
from parking.app.pricing import (
    CheckoutService,
    FeeQuote,
    TicketNotFoundError,
    TicketStateError,
    calculate_fee,
)
from parking.app.ticket import STATUS_CLOSED, STATUS_OPEN, ParkingTicket

RATE_PAISE = 10_000  # INR 100.00/hour
CAP_PAISE = 200_000  # INR 2,000.00/day
ENTRY = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def make_config(**kwargs: object) -> ParkingConfiguration:
    values = {
        "capacity": 1,
        "grace_period_minutes": 0,
        "hourly_rate_paise": RATE_PAISE,
        "daily_cap_paise": CAP_PAISE,
        "currency": "INR",
        "version": 1,
    }
    values.update(kwargs)
    return ParkingConfiguration(**values)


def at(hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime:
    return ENTRY + timedelta(hours=hours, minutes=minutes, seconds=seconds)


class GraceTests(unittest.TestCase):
    def test_within_grace_period_costs_zero(self) -> None:
        config = make_config(grace_period_minutes=5)
        quote = calculate_fee(config, ENTRY, at(minutes=1), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 0)

    def test_exactly_at_grace_limit_costs_zero(self) -> None:
        config = make_config(grace_period_minutes=5)
        quote = calculate_fee(config, ENTRY, at(minutes=5), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 0)

    def test_zero_elapsed_costs_zero(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, ENTRY, ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 0)
        self.assertEqual(quote.duration_seconds, 0)


class HourlyBillingTests(unittest.TestCase):
    def test_partial_hour_after_grace_rounds_up(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(seconds=1), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, RATE_PAISE)

    def test_one_second_over_grace_charges_full_hour(self) -> None:
        config = make_config(grace_period_minutes=5)
        quote = calculate_fee(config, ENTRY, at(minutes=5, seconds=1), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, RATE_PAISE)

    def test_two_full_hours_charge_two(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=2), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 2 * RATE_PAISE)

    def test_fractional_hours_round_up_to_full(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=1, minutes=30), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 2 * RATE_PAISE)


class DailyCapTests(unittest.TestCase):
    def test_24_hours_capped_at_daily_cap(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=24), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, CAP_PAISE)

    def test_25_hours_caps_first_block_then_bills_hour(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=25), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, CAP_PAISE + RATE_PAISE)

    def test_48_hours_is_two_full_caps(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=48), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 2 * CAP_PAISE)

    def test_partial_second_block_rounds_up(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=24, minutes=1), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, CAP_PAISE + RATE_PAISE)

    def test_no_cap_charges_full_hourly_total(self) -> None:
        config = make_config(daily_cap_paise=0)
        quote = calculate_fee(config, ENTRY, at(hours=24), ticket_id="T-1")
        self.assertEqual(quote.amount_paise, 24 * RATE_PAISE)


class FeeQuoteFieldTests(unittest.TestCase):
    def test_quote_carries_duration_iso8601(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=25, minutes=1, seconds=1))
        self.assertEqual(quote.duration, "P1DT1H1M1S")

    def test_quote_carries_currency_and_pricing_version(self) -> None:
        config = make_config(currency="USD", version=7)
        quote = calculate_fee(config, ENTRY, at(hours=2))
        self.assertEqual(quote.currency, "USD")
        self.assertEqual(quote.pricing_version, 7)
        self.assertIsInstance(quote.amount_paise, int)

    def test_quote_echoes_ticket_and_times(self) -> None:
        quote = calculate_fee(make_config(), ENTRY, at(hours=2), ticket_id="T-9")
        self.assertEqual(quote.ticket_id, "T-9")
        self.assertEqual(quote.entry_time, ENTRY)
        self.assertEqual(quote.quoted_at, at(hours=2))


class CalculationErrorTests(unittest.TestCase):
    def test_negative_elapsed_raises(self) -> None:
        with self.assertRaises(ValueError):
            calculate_fee(make_config(), ENTRY, at(hours=-1))


class CheckoutServiceTests(unittest.TestCase):
    def _service(self, ticket: ParkingTicket | None) -> CheckoutService:
        return CheckoutService(
            make_config(),
            lambda _ticket_id: ticket,
            now=lambda: at(hours=2),
        )

    def test_unknown_ticket_rejected(self) -> None:
        with self.assertRaises(TicketNotFoundError):
            self._service(None).quote("T-999")

    def test_closed_ticket_rejected(self) -> None:
        closed = ParkingTicket(
            ticket_id="T-1",
            vehicle_number="MH12-AB 1234",
            entry_time=ENTRY,
            status=STATUS_CLOSED,
        )
        with self.assertRaises(TicketStateError):
            self._service(closed).quote("T-1")

    def test_open_ticket_quoted(self) -> None:
        ticket = ParkingTicket(ticket_id="T-1", vehicle_number="MH12-AB 1234", entry_time=ENTRY)
        quote = self._service(ticket).quote("T-1")
        self.assertIsInstance(quote, FeeQuote)
        self.assertEqual(quote.ticket_id, "T-1")
        self.assertEqual(quote.amount_paise, 2 * RATE_PAISE)

    def test_quote_uses_injected_server_clock(self) -> None:
        ticket = ParkingTicket(ticket_id="T-1", vehicle_number="MH12-AB 1234", entry_time=ENTRY)
        service = CheckoutService(
            make_config(), lambda _ticket_id: ticket, now=lambda: at(hours=3)
        )
        quote = service.quote("T-1")
        self.assertEqual(quote.quoted_at, at(hours=3))
        self.assertEqual(quote.amount_paise, 3 * RATE_PAISE)


class CliTests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def _request_file(self, payload: object) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        return handle.name

    def test_batch_entry_then_quote_succeeds(self) -> None:
        path = self._request_file(
            [
                {"vehicle_number": "MH12-AB 1234", "idempotency_key": "k1"},
                {"action": "quote", "ticket_id": "T-1"},
            ]
        )
        try:
            code, out, err = self._run_cli(
                [
                    "--capacity", "1",
                    "--hourly-rate-paise", "10000",
                    path,
                ]
            )
            self.assertEqual(code, 0, err)
            results = json.loads(out)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["status"], STATUS_OPEN)
            self.assertEqual(results[1]["ticket_id"], "T-1")
            self.assertEqual(results[1]["amount_paise"], 0)
            self.assertEqual(results[1]["currency"], "INR")
            self.assertEqual(results[1]["pricing_version"], 1)
        finally:
            os.unlink(path)

    def test_single_unknown_quote_exits_one(self) -> None:
        path = self._request_file({"action": "quote", "ticket_id": "T-999"})
        try:
            code, out, err = self._run_cli(["--capacity", "1", path])
            self.assertEqual(code, 1, err)
            result = json.loads(out)
            self.assertEqual(result["error"], "TICKET_NOT_FOUND")
        finally:
            os.unlink(path)

    def test_batch_mixes_entry_and_quote_results_inline(self) -> None:
        path = self._request_file(
            [
                {"vehicle_number": "MH12-AB 1111"},
                {"action": "quote", "ticket_id": "T-1"},
                {"action": "quote", "ticket_id": "T-999"},
            ]
        )
        try:
            code, out, err = self._run_cli(["--capacity", "2", path])
            self.assertEqual(code, 0, err)
            results = json.loads(out)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[1]["ticket_id"], "T-1")
            self.assertEqual(results[2]["error"], "TICKET_NOT_FOUND")
        finally:
            os.unlink(path)

    def test_quote_missing_ticket_id_exits_two(self) -> None:
        path = self._request_file({"action": "quote"})
        try:
            code, out, err = self._run_cli(["--capacity", "1", path])
            self.assertEqual(code, 2)
            self.assertIn("error:", err)
        finally:
            os.unlink(path)

    def test_legacy_entry_only_request_still_works(self) -> None:
        path = self._request_file(
            [
                {"vehicle_number": "MH12-AB 1234"},
                {"action": "quote", "ticket_id": "T-1"},
            ]
        )
        try:
            code, out, err = self._run_cli(["--capacity", "1", path])
            self.assertEqual(code, 0, err)
            results = json.loads(out)
            self.assertEqual(results[0]["status"], STATUS_OPEN)
            self.assertEqual(results[1]["ticket_id"], "T-1")
        finally:
            os.unlink(path)

    def test_load_requests_defaults_to_entry_without_action(self) -> None:
        requests, is_batch = _load_requests({"vehicle_number": "MH12-AB 1111"})
        self.assertFalse(is_batch)
        self.assertEqual(requests[0]["action"], "entry")

    def test_load_requests_rejects_unknown_action(self) -> None:
        with self.assertRaises(ValueError):
            _load_requests({"action": "refund"})


if __name__ == "__main__":
    unittest.main()
