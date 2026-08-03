from __future__ import annotations

import json
import unittest

from episodic_memory.extract import extract_episode, graph_projection, sanitize_text


class EpisodeExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = {
            "created_at": "2026-08-02T12:00:00+00:00",
            "project": "/work/erp",
            "task": "Implement Jira ticket MPS-2",
            "status": "completed",
            "trace_id": "abc123",
            "plan": {
                "agent": "plan",
                "returncode": 0,
                "duration_seconds": 2,
                "text": "I recommend a repository boundary.",
                "events": [
                    {
                        "type": "tool_use",
                        "timestamp": 1785672000000,
                        "part": {
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": "/work/erp/mps/app/main.py"},
                            },
                        },
                    }
                ],
            },
            "build": {
                "agent": "build",
                "returncode": 0,
                "duration_seconds": 5,
                "text": (
                    "Going with the repository boundary.\n"
                    "Test helper bug fixed.\n12 tests passed."
                ),
                "events": [
                    {
                        "type": "tool_use",
                        "timestamp": 1785672005000,
                        "part": {
                            "tool": "write",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": "/work/erp/mps/app/ticket.py"},
                            },
                        },
                    },
                    {
                        "type": "tool_use",
                        "timestamp": 1785672006000,
                        "part": {
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {
                                    "command": "API_KEY=sk-supersecretvalue pytest -q"
                                },
                            },
                        },
                    },
                ],
            },
            "langfuse_observations": {"data": [{"type": "AGENT"}]},
        }

    def test_extracts_compact_memory_and_provenance(self) -> None:
        episode = extract_episode(self.trace)
        self.assertEqual(episode["episode_id"], "episode:abc123")
        self.assertEqual(episode["primary_work_items"], ["MPS-2"])
        self.assertIn("12 tests passed", episode["summary"])
        self.assertEqual(episode["metrics"]["tool_call_count"], 3)
        artifacts = {a["path"]: a for a in episode["artifacts"]}
        self.assertFalse(artifacts["mps/app/main.py"]["changed"])
        self.assertTrue(artifacts["mps/app/ticket.py"]["changed"])
        self.assertTrue(episode["decisions"])
        self.assertTrue(episode["obstacles"][0]["resolved"])
        self.assertNotIn("tool output", json.dumps(episode))

    def test_projection_has_typed_idempotent_ids(self) -> None:
        episode = extract_episode(self.trace)
        first = graph_projection(episode)
        second = graph_projection(episode)
        self.assertEqual(first, second)
        labels = {node["label"] for node in first["nodes"]}
        edge_types = {edge["type"] for edge in first["edges"]}
        self.assertTrue({"Episode", "Project", "Ticket", "Outcome"} <= labels)
        self.assertTrue({"ABOUT", "OCCURRED_IN", "PRODUCED"} <= edge_types)
        self.assertEqual(
            len({node["id"] for node in first["nodes"]}), len(first["nodes"])
        )

    def test_redacts_common_secret_forms(self) -> None:
        text = sanitize_text(
            "api_key=abc123456789012345 Bearer tokenvalue123456 sk-abcdefghijklmno"
        )
        self.assertNotIn("abc123456789012345", text)
        self.assertNotIn("tokenvalue123456", text)
        self.assertNotIn("sk-abcdefghijklmno", text)


if __name__ == "__main__":
    unittest.main()
