#!/usr/bin/env python3
"""Query the JSONL episodic-memory file used by the OpenCode orchestrator."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_./:-]+")


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if len(token) >= 3
    }


def load_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    episodes: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Ignoring malformed line {number}: {exc}", file=sys.stderr)
            continue
        if isinstance(value, dict):
            episodes.append(value)
    return episodes


def searchable_text(episode: dict[str, Any]) -> str:
    return json.dumps(
        {
            "ticket_key": episode.get("ticket_key"),
            "task": episode.get("task"),
            "summary": episode.get("outcome", {}).get("summary"),
            "decisions": episode.get("decisions", []),
            "artifacts": episode.get("artifacts", []),
            "interfaces": episode.get("interfaces_produced", []),
            "failures": episode.get("failure_and_recovery", []),
            "knowledge": episode.get("reusable_knowledge", []),
        },
        ensure_ascii=False,
    )


def query(
    episodes: list[dict[str, Any]],
    *,
    text: str,
    ticket: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = tokenize(text)
    ticket = ticket.upper() if ticket else None
    ranked: list[tuple[float, dict[str, Any]]] = []

    for episode in episodes:
        if episode.get("outcome", {}).get("status") != "success":
            continue
        episode_ticket = str(episode.get("ticket_key", "")).upper()
        memory_tokens = tokenize(searchable_text(episode))

        score = 0.0
        if ticket and episode_ticket == ticket:
            score += 100.0
        if query_tokens:
            score += 50.0 * len(query_tokens & memory_tokens) / len(query_tokens)
        if score > 0:
            ranked.append((score, episode))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:limit]]


def compact(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id"),
        "ticket_key": episode.get("ticket_key"),
        "task": episode.get("task"),
        "outcome": episode.get("outcome"),
        "decisions": episode.get("decisions", []),
        "artifacts": episode.get("artifacts", []),
        "interfaces_produced": episode.get("interfaces_produced", []),
        "failure_and_recovery": episode.get("failure_and_recovery", []),
        "verification": episode.get("verification", []),
        "reusable_knowledge": episode.get("reusable_knowledge", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Interface, failure, artifact, or implementation question")
    parser.add_argument("--memory", default=".episodic_memory/episodes.jsonl")
    parser.add_argument("--ticket", help="Prefer an exact ticket key")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    try:
        results = query(
            load_episodes(Path(args.memory)),
            text=args.query,
            ticket=args.ticket,
            limit=max(1, args.limit),
        )
    except OSError as exc:
        print(f"Memory read failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps([compact(item) for item in results], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
