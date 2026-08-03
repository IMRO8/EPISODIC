"""Command-line interface for episodic-memory extraction and ingestion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .extract import extract_episode, graph_projection


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(value: Any, path: Path | None) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m episodic_memory",
        description="Extract and persist episodic memory from Langfuse trace JSON.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Create compact episode JSON")
    extract.add_argument("trace", type=Path)
    extract.add_argument("-o", "--output", type=Path)
    extract.add_argument(
        "--with-graph", action="store_true", help="Include AGE nodes and edges"
    )

    inspect = commands.add_parser("inspect", help="Print a compact extraction summary")
    inspect.add_argument("trace", type=Path)

    ingest = commands.add_parser("ingest", help="Load JSONB and Apache AGE")
    ingest.add_argument("trace", type=Path)
    ingest.add_argument("--dsn", default=os.getenv("EPISODIC_POSTGRES_DSN"))
    ingest.add_argument("--graph", default="episodic_memory")
    ingest.add_argument("-o", "--episode-output", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        trace = _read_json(args.trace)
        episode = extract_episode(trace)
        if args.command == "extract":
            value: dict[str, Any] = episode
            if args.with_graph:
                value = {"episode": episode, "graph": graph_projection(episode)}
            _write_json(value, args.output)
            return 0
        if args.command == "inspect":
            projection = graph_projection(episode)
            print(episode["summary"])
            print(
                f"work_items={episode['referenced_work_items']} "
                f"artifacts={len(episode['artifacts'])} "
                f"nodes={len(projection['nodes'])} edges={len(projection['edges'])}"
            )
            return 0
        if args.command == "ingest":
            if not args.dsn:
                raise ValueError(
                    "Pass --dsn or set EPISODIC_POSTGRES_DSN; credentials are not read "
                    "from the trace"
                )
            if args.episode_output:
                _write_json(episode, args.episode_output)
            from .postgres import ingest

            result = ingest(args.dsn, trace, episode, args.graph)
            print(json.dumps(result, indent=2))
            return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1
