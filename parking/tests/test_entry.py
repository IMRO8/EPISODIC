from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone

from parking.app.config import ConfigStore, ParkingConfiguration
from parking.app.main import _load_requests, main
from parking.app.ticket import (
    STATUS_OPEN,
    LotFullError,
    TicketService,
    normalize_vehicle_number,
)


def make_service(capacity: int = 3, **kwargs: object) -> TicketService:
    return TicketService(
        ConfigStore(ParkingConfiguration(capacity=capacity)).active(), **kwargs
    )


class ConfigurationTests(unittest.TestCase):
    def test_requires_positive_capacity(self) -> None:
        with self.assertRaises(ValueError):
            ParkingConfiguration(capacity=0)
        with self.assertRaises(ValueError):
            ParkingConfiguration(capacity=-1)
        with self.assertRaises(ValueError):
            ParkingConfiguration(capacity="2")

    def test_rejects_negative_grace_and_money(self) -> None:
        with self.assertRaises(ValueError):
            ParkingConfiguration(capacity=2, grace_period_minutes=-1)
        with self.assertRaises(ValueError):
            ParkingConfiguration(capacity=2, hourly_rate_paise=-5)
        with self.assertRaises(ValueError):
            ParkingConfiguration(capacity=2, version=0)

    def test_active_configuration_retrievable(self) -> None:
        config = ParkingConfiguration(capacity=4)
        self.assertEqual(ConfigStore(config).active(), config)


class NormalizationTests(unittest.TestCase):
    def test_trims_and_uppercases(self) -> None:
        self.assertEqual(
            normalize_vehicle_number("  MH12-AB 1234 "), "MH12-AB 1234"
        )

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(ValueError):
            normalize_vehicle_number(1234)

    def test_rejects_blank(self) -> None:
        with self.assertRaises(ValueError):
            normalize_vehicle_number("   ")


class EntryTests(unittest.TestCase):
    def test_each_entry_creates_unique_open_ticket(self) -> None:
        service = make_service(capacity=3)
        tickets = [service.enter(f"mh12-ab 123{i}") for i in range(3)]
        self.assertEqual(len(tickets), 3)
        self.assertEqual(len({t.ticket_id for t in tickets}), 3)
        for ticket in tickets:
            self.assertEqual(ticket.status, STATUS_OPEN)

    def test_occupancy_increments_exactly_once_per_entry(self) -> None:
        service = make_service(capacity=3)
        service.enter("MH12-AB 1111")
        service.enter("MH12-AB 2222")
        self.assertEqual(service.occupancy, 2)

    def test_lot_full_rejects_without_ticket_or_occupancy_change(self) -> None:
        service = make_service(capacity=2)
        first = service.enter("MH12-AB 1111")
        second = service.enter("MH12-AB 2222")
        with self.assertRaises(LotFullError):
            service.enter("MH12-AB 3333")
        self.assertEqual(service.occupancy, 2)
        self.assertEqual(service.get(first.ticket_id), first)
        self.assertEqual(service.get(second.ticket_id), second)

    def test_idempotent_repeat_returns_existing_ticket(self) -> None:
        service = make_service(capacity=2)
        first = service.enter("  MH12-AB 1111 ", idempotency_key="k1")
        second = service.enter("MH12-AB 1111", idempotency_key="k1")
        self.assertIs(first, second)
        self.assertEqual(service.occupancy, 1)

    def test_idempotent_repeat_wins_when_lot_full(self) -> None:
        service = make_service(capacity=1)
        first = service.enter("MH12-AB 1111", idempotency_key="k1")
        with self.assertRaises(LotFullError):
            service.enter("MH12-AB 2222")
        second = service.enter("MH12-AB 1111", idempotency_key="k1")
        self.assertIs(first, second)
        self.assertEqual(service.occupancy, 1)

    def test_server_entry_time_ignores_client_input(self) -> None:
        now = datetime(2026, 8, 3, 12, 30, tzinfo=timezone.utc)
        service = make_service(capacity=1, now=lambda: now)
        ticket = service.enter("MH12-AB 1111")
        self.assertEqual(ticket.entry_time, now)
        self.assertEqual(ticket.entry_time.tzinfo, timezone.utc)

    def test_duplicate_ticket_id_aborts_without_occupancy_change(self) -> None:
        service = make_service(capacity=3, new_ticket_id=lambda: "T-X")
        first = service.enter("MH12-AB 1111")
        self.assertEqual(first.ticket_id, "T-X")
        with self.assertRaises(ValueError):
            service.enter("MH12-AB 2222")
        self.assertEqual(service.occupancy, 1)
        self.assertIsNone(service.get("T-2"))

    def test_get_returns_none_for_unknown_ticket(self) -> None:
        service = make_service(capacity=1)
        self.assertIsNone(service.get("T-999"))


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

    def test_issues_ticket_from_single_request(self) -> None:
        path = self._request_file(
            {"vehicle_number": "  mh12-ab 1234 ", "idempotency_key": "k1"}
        )
        try:
            code, out, err = self._run_cli(["--capacity", "2", path])
            self.assertEqual(code, 0, err)
            ticket = json.loads(out)
            self.assertEqual(ticket["vehicle_number"], "MH12-AB 1234")
            self.assertEqual(ticket["status"], STATUS_OPEN)
            self.assertIn("ticket_id", ticket)
            self.assertIn("entry_time", ticket)
        finally:
            os.unlink(path)

    def test_batch_reports_lot_full_inline(self) -> None:
        path = self._request_file(
            [
                {"vehicle_number": "MH12-AB 1111", "idempotency_key": "k1"},
                {"vehicle_number": "MH12-AB 2222", "idempotency_key": "k2"},
                {"vehicle_number": "MH12-AB 3333", "idempotency_key": "k3"},
            ]
        )
        try:
            code, out, err = self._run_cli(["--capacity", "2", path])
            self.assertEqual(code, 0, err)
            results = json.loads(out)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[2]["error"], "LOT_FULL")
            self.assertEqual(results[0]["status"], STATUS_OPEN)
        finally:
            os.unlink(path)

    def test_invalid_payload_reports_error(self) -> None:
        path = self._request_file({"not": "a request"})
        try:
            code, out, err = self._run_cli(["--capacity", "2", path])
            self.assertEqual(code, 2)
            self.assertIn("error:", err)
        finally:
            os.unlink(path)

    def test_load_requests_accepts_single_object(self) -> None:
        requests, is_batch = _load_requests(
            {"vehicle_number": "MH12-AB 1111", "idempotency_key": "k1"}
        )
        self.assertFalse(is_batch)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["vehicle_number"], "MH12-AB 1111")

    def test_load_requests_accepts_list(self) -> None:
        requests, is_batch = _load_requests(
            [{"vehicle_number": "A"}, {"vehicle_number": "B"}]
        )
        self.assertTrue(is_batch)
        self.assertEqual(len(requests), 2)

    def test_load_requests_rejects_unknown_shape(self) -> None:
        with self.assertRaises(ValueError):
            _load_requests([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
