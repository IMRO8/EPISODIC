"""Deterministic Langfuse trace to episodic-memory extraction.

The extractor deliberately does not copy tool outputs into the compact memory.
They can contain whole files, credentials, and a great deal of low-value text.
The original trace is retained as JSONB and compact records point back to it
with JSON Pointer-like evidence paths.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
WORK_ITEM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-\d+\b")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    ),
)
FILE_TOOLS = {"read": "read", "write": "write", "edit": "edit"}


def sanitize_text(value: Any, limit: int | None = None) -> str:
    """Remove common credential forms from text copied into compact memory."""
    text = "" if value is None else str(value)
    text = SECRET_PATTERNS[0].sub("<redacted-api-key>", text)
    text = SECRET_PATTERNS[1].sub(r"\1<redacted-token>", text)
    text = SECRET_PATTERNS[2].sub(r"\1\2<redacted>", text)
    if limit is not None and len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _canonical_bytes(trace: dict[str, Any]) -> bytes:
    return json.dumps(
        trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _iso_from_milliseconds(value: Any) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _relative_path(value: Any, project: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    project_path = Path(project) if project else None
    if project_path and candidate.is_absolute():
        try:
            return candidate.relative_to(project_path).as_posix() or "."
        except ValueError:
            pass
    return candidate.as_posix()


def _command_names(command: Any) -> list[str]:
    """Keep executable names, not full shell commands or their arguments."""
    if not isinstance(command, str):
        return []
    names: list[str] = []
    for segment in re.split(r"(?:&&|\|\||[;|])", command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        while tokens and ("=" in tokens[0] and not tokens[0].startswith(("/", "./"))):
            tokens.pop(0)
        if not tokens:
            continue
        name = Path(tokens[0]).name
        if name in {"sudo", "env", "command"} and len(tokens) > 1:
            name = Path(tokens[1]).name
        if re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
            names.append(name)
    return names


def _phase_memory(
    phase_name: str, phase: dict[str, Any], project: str
) -> tuple[dict[str, Any], dict[str, set[str]], list[dict[str, Any]]]:
    events = phase.get("events") if isinstance(phase.get("events"), list) else []
    event_types: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    command_names: Counter[str] = Counter()
    artifacts: dict[str, set[str]] = defaultdict(set)
    failures: list[dict[str, Any]] = []
    timestamps: list[int | float] = []

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "unknown")
        event_types[event_type] += 1
        if isinstance(event.get("timestamp"), (int, float)):
            timestamps.append(event["timestamp"])
        if event_type != "tool_use":
            continue

        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        tool = str(part.get("tool") or "unknown")
        status = str(state.get("status") or "unknown")
        tool_input = state.get("input") if isinstance(state.get("input"), dict) else {}
        tools[tool] += 1
        statuses[status] += 1

        if tool in FILE_TOOLS:
            relative = _relative_path(tool_input.get("filePath"), project)
            if relative:
                artifacts[relative].add(FILE_TOOLS[tool])
        if tool == "bash":
            command_names.update(_command_names(tool_input.get("command")))
        if status in {"error", "failed"}:
            failures.append(
                {
                    "phase": phase_name,
                    "tool": tool,
                    "message": sanitize_text(
                        state.get("error") or state.get("output") or status, 1000
                    ),
                    "evidence": f"/{phase_name}/events/{index}",
                }
            )

    returncode = phase.get("returncode")
    stderr = sanitize_text(phase.get("stderr"), 1000).strip()
    if returncode not in (None, 0):
        failures.append(
            {
                "phase": phase_name,
                "tool": "phase",
                "message": stderr or f"Phase returned {returncode}",
                "evidence": f"/{phase_name}",
            }
        )

    memory = {
        "name": phase_name,
        "agent": sanitize_text(phase.get("agent"), 100),
        "session_id": sanitize_text(phase.get("session_id"), 200),
        "status": "succeeded" if returncode == 0 and not failures else "failed",
        "returncode": returncode,
        "duration_seconds": phase.get("duration_seconds"),
        "started_at": _iso_from_milliseconds(min(timestamps)) if timestamps else None,
        "ended_at": _iso_from_milliseconds(max(timestamps)) if timestamps else None,
        "event_count": len(events),
        "event_types": dict(sorted(event_types.items())),
        "tool_counts": dict(sorted(tools.items())),
        "tool_status_counts": dict(sorted(statuses.items())),
        "command_kinds": dict(sorted(command_names.items())),
        "narrative": sanitize_text(phase.get("text"), 8000),
        "evidence": f"/{phase_name}",
    }
    return memory, artifacts, failures


def _decisions(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract explicit choice statements without inventing model conclusions."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    markers = re.compile(
        r"(?i)\b(?:going with|recommend(?:ed)?|chose|chosen|decided|per the plan)\b"
    )
    for phase_name in ("plan", "build"):
        phase = trace.get(phase_name)
        if not isinstance(phase, dict):
            continue
        text = sanitize_text(phase.get("text"), 12000)
        for paragraph in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text):
            candidate = " ".join(paragraph.strip().split())
            if not candidate or not markers.search(candidate):
                continue
            candidate = sanitize_text(candidate, 1200)
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "text": candidate,
                    "phase": phase_name,
                    "evidence": f"/{phase_name}/text",
                }
            )
    return found[:10]


def _obstacles(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Capture explicitly reported bugs that were encountered and repaired."""
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    marker = re.compile(r"(?i)\b(?:bug|error|failure|failed|fixing|fixed)\b")
    for phase_name in ("plan", "build"):
        phase = trace.get(phase_name)
        if not isinstance(phase, dict):
            continue
        text = sanitize_text(phase.get("text"), 12000)
        for line in text.splitlines():
            candidate = " ".join(line.strip(" -*#\t").split())
            if not candidate or not marker.search(candidate):
                continue
            if len(candidate) > 800:
                candidate = candidate[:799].rstrip() + "…"
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "text": candidate,
                    "resolved": bool(
                        re.search(r"(?i)\b(?:fixing|fixed|pass(?:es|ed)?)\b", candidate)
                    ),
                    "phase": phase_name,
                    "evidence": f"/{phase_name}/text",
                }
            )
    return found[:10]


def _summary(trace: dict[str, Any], phases: list[dict[str, Any]]) -> str:
    task = sanitize_text(trace.get("task"), 500).strip() or "Agent task"
    status = sanitize_text(trace.get("status"), 50).strip() or "unknown"
    phase_bits = [
        f"{p['name']}: {p['status']} with {sum(p['tool_counts'].values())} tool calls"
        for p in phases
    ]
    verification = ""
    build_text = sanitize_text((trace.get("build") or {}).get("text"), 8000)
    match = re.search(r"(?i)\b(\d+)\s+tests?\s+pass", build_text)
    if match:
        verification = f"; {match.group(1)} tests passed"
    return f"{task} — {status}. " + "; ".join(phase_bits) + verification + "."


def extract_episode(trace: dict[str, Any]) -> dict[str, Any]:
    """Create a compact, provenance-preserving episode from one trace export."""
    if not isinstance(trace, dict):
        raise TypeError("trace must be a JSON object")
    trace_id = str(trace.get("trace_id") or "").strip()
    if not trace_id:
        trace_id = hashlib.sha256(_canonical_bytes(trace)).hexdigest()[:32]

    project = sanitize_text(trace.get("project"), 1000)
    phase_memories: list[dict[str, Any]] = []
    artifact_modes: dict[str, set[str]] = defaultdict(set)
    failures: list[dict[str, Any]] = []
    for phase_name in ("plan", "build"):
        phase = trace.get(phase_name)
        if not isinstance(phase, dict):
            continue
        phase_memory, artifacts, phase_failures = _phase_memory(
            phase_name, phase, project
        )
        phase_memories.append(phase_memory)
        for path, modes in artifacts.items():
            artifact_modes[path].update(modes)
        failures.extend(phase_failures)

    task = sanitize_text(trace.get("task"), 4000)
    work_item_source = "\n".join(
        [
            task,
            sanitize_text((trace.get("plan") or {}).get("text"), 12000),
            sanitize_text((trace.get("build") or {}).get("text"), 12000),
        ]
    )
    work_items = sorted(set(WORK_ITEM_RE.findall(work_item_source)))
    primary_work_items = sorted(set(WORK_ITEM_RE.findall(task)))
    decisions = _decisions(trace)
    obstacles = _obstacles(trace)
    artifacts = [
        {
            "path": path,
            "access": sorted(modes),
            "changed": bool(modes & {"write", "edit"}),
        }
        for path, modes in sorted(artifact_modes.items())
    ]
    observations = (trace.get("langfuse_observations") or {}).get("data") or []

    episode = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": f"episode:{trace_id}",
        "source": {
            "kind": "langfuse_trace",
            "trace_id": trace_id,
            "trace_url": sanitize_text(trace.get("trace_url"), 2000),
            "raw_sha256": hashlib.sha256(_canonical_bytes(trace)).hexdigest(),
        },
        "created_at": sanitize_text(trace.get("created_at"), 100),
        "project": project,
        "project_name": Path(project).name if project else "unknown",
        "task": task,
        "status": sanitize_text(trace.get("status"), 100) or "unknown",
        "model": sanitize_text(trace.get("model"), 500),
        "primary_work_items": primary_work_items,
        "referenced_work_items": work_items,
        "summary": "",
        "phases": phase_memories,
        "artifacts": artifacts,
        "decisions": decisions,
        "obstacles": obstacles,
        "failures": failures,
        "outcome": {
            "status": sanitize_text(trace.get("status"), 100) or "unknown",
            "final_report": sanitize_text((trace.get("build") or {}).get("text"), 8000),
            "evidence": "/build/text" if isinstance(trace.get("build"), dict) else "/status",
        },
        "metrics": {
            "phase_count": len(phase_memories),
            "event_count": sum(p["event_count"] for p in phase_memories),
            "tool_call_count": sum(
                sum(p["tool_counts"].values()) for p in phase_memories
            ),
            "artifact_count": len(artifacts),
            "langfuse_observation_count": len(observations),
        },
    }
    episode["summary"] = _summary(trace, phase_memories)
    return episode


def _stable_id(prefix: str, *parts: str) -> str:
    material = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(material).hexdigest()[:20]}"


def graph_projection(episode: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Project an episode into typed AGE vertices and directed relationships."""
    episode_id = episode["episode_id"]
    project_name = episode.get("project_name") or "unknown"
    project_id = _stable_id("project", episode.get("project") or project_name)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(label: str, node_id: str, **properties: Any) -> None:
        nodes.append({"label": label, "id": node_id, "properties": properties})

    def edge(kind: str, source: str, target: str, **properties: Any) -> None:
        edge_id = _stable_id("edge", kind, source, target)
        edges.append(
            {
                "type": kind,
                "id": edge_id,
                "source": source,
                "target": target,
                "properties": properties,
            }
        )

    node(
        "Episode",
        episode_id,
        kind="episode",
        name=episode.get("task", ""),
        text=episode.get("summary", ""),
        status=episode.get("status", ""),
        created_at=episode.get("created_at", ""),
        trace_id=(episode.get("source") or {}).get("trace_id", ""),
    )
    node(
        "Project",
        project_id,
        kind="project",
        name=project_name,
        path=episode.get("project", ""),
    )
    edge("OCCURRED_IN", episode_id, project_id)

    primary = set(episode.get("primary_work_items") or [])
    for key in episode.get("referenced_work_items") or []:
        ticket_id = f"work-item:{key}"
        node("Ticket", ticket_id, kind="ticket", name=key, ticket_key=key)
        edge("ABOUT", episode_id, ticket_id, primary=key in primary)
        if key in primary and episode.get("status") == "completed":
            edge("IMPLEMENTED_BY", ticket_id, episode_id)

    previous_phase_id: str | None = None
    for ordinal, phase in enumerate(episode.get("phases") or []):
        phase_id = f"{episode_id}:phase:{phase['name']}"
        node(
            "Phase",
            phase_id,
            kind="phase",
            name=phase["name"],
            phase=phase["name"],
            status=phase.get("status", ""),
            text=phase.get("narrative", ""),
            ordinal=ordinal,
            metadata=phase,
        )
        edge("HAS_PHASE", episode_id, phase_id, ordinal=ordinal)
        if previous_phase_id:
            edge("NEXT", previous_phase_id, phase_id)
        previous_phase_id = phase_id

    for artifact in episode.get("artifacts") or []:
        path = artifact["path"]
        artifact_id = _stable_id("artifact", episode.get("project", ""), path)
        node(
            "Artifact",
            artifact_id,
            kind="artifact",
            name=Path(path).name,
            path=path,
            status="changed" if artifact.get("changed") else "read",
            metadata=artifact,
        )
        edge("TOUCHED", episode_id, artifact_id, access=artifact.get("access", []))

    for ordinal, decision in enumerate(episode.get("decisions") or []):
        decision_id = _stable_id("decision", episode_id, decision.get("text", ""))
        node(
            "Decision",
            decision_id,
            kind="decision",
            name=f"Decision {ordinal + 1}",
            text=decision.get("text", ""),
            phase=decision.get("phase", ""),
            ordinal=ordinal,
            metadata=decision,
        )
        edge("MADE_DECISION", episode_id, decision_id, ordinal=ordinal)

    for ordinal, obstacle in enumerate(episode.get("obstacles") or []):
        obstacle_id = _stable_id("obstacle", episode_id, obstacle.get("text", ""))
        node(
            "Obstacle",
            obstacle_id,
            kind="obstacle",
            name=f"Obstacle {ordinal + 1}",
            text=obstacle.get("text", ""),
            status="resolved" if obstacle.get("resolved") else "observed",
            phase=obstacle.get("phase", ""),
            ordinal=ordinal,
            metadata=obstacle,
        )
        edge("ENCOUNTERED", episode_id, obstacle_id, ordinal=ordinal)

    outcome_id = f"{episode_id}:outcome"
    outcome = episode.get("outcome") or {}
    node(
        "Outcome",
        outcome_id,
        kind="outcome",
        name="Final outcome",
        text=outcome.get("final_report", ""),
        status=outcome.get("status", ""),
        metadata=outcome,
    )
    edge("PRODUCED", episode_id, outcome_id)

    # Different paths can discover the same shared node. Deduplicate by id.
    unique_nodes = {item["id"]: item for item in nodes}
    unique_edges = {item["id"]: item for item in edges}
    return {
        "nodes": list(unique_nodes.values()),
        "edges": list(unique_edges.values()),
    }
