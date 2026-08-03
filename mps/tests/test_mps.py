from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from mps.app.main import _load_demands, main
from mps.app.mps import Demand, plan_mps


class DemandValidationTests(unittest.TestCase):
    def test_requires_positive_period(self) -> None:
        with self.assertRaises(ValueError):
            Demand(period=0, gross_requirement=10)

    def test_rejects_negative_gross_requirement(self) -> None:
        with self.assertRaises(ValueError):
            Demand(period=1, gross_requirement=-1)

    def test_rejects_negative_scheduled_receipt(self) -> None:
        with self.assertRaises(ValueError):
            Demand(period=1, gross_requirement=1, scheduled_receipt=-5)


class PlanMpsTests(unittest.TestCase):
    def test_demand_covered_by_on_hand_produces_no_receipts(self) -> None:
        plan = plan_mps(
            [Demand(1, 30), Demand(2, 40)],
            starting_on_hand=100,
        )
        self.assertEqual(
            [line.planned_order_receipt for line in plan], [0.0, 0.0]
        )
        self.assertEqual(
            [line.projected_on_hand for line in plan], [70.0, 30.0]
        )

    def test_net_requirement_produces_exact_receipt(self) -> None:
        plan = plan_mps([Demand(1, 80)], starting_on_hand=20)
        line = plan[0]
        self.assertEqual(line.net_requirement, 60.0)
        self.assertEqual(line.planned_order_receipt, 60.0)
        self.assertEqual(line.projected_on_hand, 0.0)

    def test_lot_size_rounds_receipt_up(self) -> None:
        plan = plan_mps([Demand(1, 35)], starting_on_hand=5, lot_size=20)
        line = plan[0]
        self.assertEqual(line.net_requirement, 30.0)
        self.assertEqual(line.planned_order_receipt, 40.0)
        self.assertEqual(line.projected_on_hand, 10.0)

    def test_scheduled_receipts_cover_demand(self) -> None:
        plan = plan_mps(
            [Demand(1, 50, scheduled_receipt=40)],
            starting_on_hand=0,
        )
        line = plan[0]
        self.assertEqual(line.net_requirement, 10.0)
        self.assertEqual(line.planned_order_receipt, 10.0)

    def test_safety_stock_keeps_reserve(self) -> None:
        plan = plan_mps(
            [Demand(1, 50)],
            starting_on_hand=60,
            safety_stock=20,
        )
        line = plan[0]
        self.assertEqual(line.net_requirement, 10.0)
        self.assertEqual(line.planned_order_receipt, 10.0)
        self.assertEqual(line.projected_on_hand, 20.0)

    def test_multiple_periods_carry_on_hand_forward(self) -> None:
        plan = plan_mps(
            [Demand(1, 40), Demand(2, 40), Demand(3, 40)],
            starting_on_hand=30,
        )
        self.assertEqual(
            [line.planned_order_receipt for line in plan], [10.0, 40.0, 40.0]
        )
        self.assertEqual(
            [line.projected_on_hand for line in plan], [0.0, 0.0, 0.0]
        )

    def test_periods_must_be_strictly_ascending(self) -> None:
        with self.assertRaises(ValueError):
            plan_mps([Demand(2, 10), Demand(2, 10)], starting_on_hand=0)

    def test_rejects_negative_on_hand(self) -> None:
        with self.assertRaises(ValueError):
            plan_mps([Demand(1, 10)], starting_on_hand=-1)

    def test_rejects_non_positive_lot_size(self) -> None:
        with self.assertRaises(ValueError):
            plan_mps([Demand(1, 10)], starting_on_hand=0, lot_size=0)


class CliTests(unittest.TestCase):
    def test_loads_numeric_list(self) -> None:
        demands = _load_demands([10, 20, 30])
        self.assertEqual([d.period for d in demands], [1, 2, 3])
        self.assertEqual([d.gross_requirement for d in demands], [10.0, 20.0, 30.0])

    def test_loads_object_list(self) -> None:
        demands = _load_demands(
            [{"period": 4, "gross_requirement": 12, "scheduled_receipt": 3}]
        )
        self.assertEqual(demands[0], Demand(period=4, gross_requirement=12.0, scheduled_receipt=3.0))

    def test_rejects_unknown_shape(self) -> None:
        with self.assertRaises(ValueError):
            _load_demands({"period": 1, "gross_requirement": 5})

    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_json_output(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump([35], handle)
            path = handle.name
        try:
            code, out, err = self._run_cli(
                ["--on-hand", "5", "--lot-size", "20", "--json", path]
            )
            self.assertEqual(code, 0, err)
            plan = json.loads(out)
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["planned_order_receipt"], 40.0)
            self.assertEqual(plan[0]["projected_on_hand"], 10.0)
        finally:
            os.unlink(path)

    def test_table_output_lists_every_period(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump([10, 20], handle)
            path = handle.name
        try:
            code, out, err = self._run_cli(["--on-hand", "0", path])
            self.assertEqual(code, 0, err)
            self.assertIn("period", out)
            self.assertIn("receipt", out)
        finally:
            os.unlink(path)

    def test_invalid_demands_report_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"not": "a list"}, handle)
            path = handle.name
        try:
            code, out, err = self._run_cli(["--on-hand", "0", path])
            self.assertEqual(code, 2)
            self.assertIn("error:", err)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
