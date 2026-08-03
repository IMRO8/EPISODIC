#!/usr/bin/env python3
"""Plan and build with OpenCode, trace both phases with Langfuse, save JSON."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes


def jsonable(value: Any) -> Any:
    """Convert Langfuse/Pydantic responses into ordinary JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def parse_jsonl(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    events, other = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            events.append(item) if isinstance(item, dict) else other.append(line)
        except json.JSONDecodeError:
            other.append(line)
    return events, other


def collect_text(value: Any, assistant: bool = False) -> list[str]:
    """Extract assistant text from raw events or `opencode export` output."""
    found: list[str] = []
    if isinstance(value, dict):
        info = value.get("info")
        assistant = assistant or value.get("role") == "assistant" or (
            isinstance(info, dict) and info.get("role") == "assistant"
        )
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            # Raw JSON events contain completed assistant text parts without a role.
            if assistant or "sessionID" in value or "messageID" in value:
                if value["text"].strip():
                    found.append(value["text"].strip())
        for child in value.values():
            found.extend(collect_text(child, assistant))
    elif isinstance(value, list):
        for child in value:
            found.extend(collect_text(child, assistant))
    return found


def export_session(opencode: str, cwd: Path, session_id: str) -> Any | None:
    result = subprocess.run(
        [opencode, "export", session_id],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        return {"export_error": result.stderr.strip()}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout}


def run_phase(
    *,
    langfuse: Any,
    phase: str,
    agent: str,
    prompt: str,
    cwd: Path,
    opencode: str,
    model: str | None,
    auto: bool,
    timeout: int,
) -> dict[str, Any]:
    command = [
        opencode,
        "run",
        "--format",
        "json",
        "--dir",
        str(cwd),
        "--agent",
        agent,
        "--title",
        f"orchestrator-{phase}",
    ]
    if model:
        command += ["--model", model]
    if auto:
        command.append("--auto")
    command.append(prompt)

    started = time.monotonic()
    with langfuse.start_as_current_observation(
        as_type="agent",
        name=f"opencode-{phase}",
        input={"agent": agent, "prompt": prompt},
    ) as span:
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout, stderr, returncode = (
                process.stdout,
                process.stderr,
                process.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = f"Timed out after {timeout} seconds"
            returncode = 124

        events, unparsed = parse_jsonl(stdout)
        session_id = next(
            (str(e["sessionID"]) for e in events if e.get("sessionID")), None
        )
        session = export_session(opencode, cwd, session_id) if session_id else None

        text_parts = [
            part
            for event in events
            if event.get("type") == "text"
            for part in collect_text(event)
        ]
        if not text_parts and session is not None:
            text_parts = collect_text(session)
        text = "\n\n".join(dict.fromkeys(text_parts)).strip()

        # Mirror visible OpenCode tool calls into Langfuse. This is not hidden CoT.
        for event in events:
            if event.get("type") != "tool_use":
                continue
            part = event.get("part", {})
            state = part.get("state", {}) if isinstance(part, dict) else {}
            langfuse.create_event(
                name=f"opencode.tool.{part.get('tool', 'unknown')}",
                input=state.get("input"),
                output=state.get("output") or state.get("error"),
                metadata={"phase": phase, "session_id": session_id},
                level="ERROR" if state.get("status") == "error" else "DEFAULT",
            )

        result = {
            "phase": phase,
            "agent": agent,
            "command": command[:-1] + ["<prompt stored in trace input>"],
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "session_id": session_id,
            "text": text,
            "events": events,
            "unparsed_stdout": unparsed,
            "stderr": stderr.strip(),
            "session_export": session,
        }
        failed = returncode != 0 or not text
        span.update(
            output={
                "returncode": returncode,
                "session_id": session_id,
                "text": text,
                "stderr": stderr.strip(),
            },
            level="ERROR" if failed else "DEFAULT",
            status_message=(
                f"OpenCode {phase} failed" if returncode else "No assistant text returned"
            )
            if failed
            else None,
        )
        return result


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(data), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="+", help="Coding task to implement")
    parser.add_argument("--project", default=".")
    parser.add_argument("--model", help="OpenCode model as provider/model")
    parser.add_argument("--output", default="langfuse_trace.json")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--opencode", default=os.getenv("OPENCODE_BIN", "opencode"))
    return parser.parse_args()


def main() -> int:
    cfg = args()
    project = Path(cfg.project).expanduser().resolve()
    load_dotenv(project / ".env")

    if not project.is_dir():
        print(f"Project directory not found: {project}", file=sys.stderr)
        return 2
    if shutil.which(cfg.opencode) is None and not Path(cfg.opencode).exists():
        print(f"OpenCode executable not found: {cfg.opencode}", file=sys.stderr)
        return 2

    missing = [
        name
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
        if not os.getenv(name)
    ]
    if missing:
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)
        return 2

    task = " ".join(cfg.task).strip()
    output = Path(cfg.output)
    output = output if output.is_absolute() else project / output
    langfuse = get_client()
    trace_id = None
    record: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": str(project),
        "task": task,
        "model": cfg.model,
        "status": "running",
    }

    plan_prompt = f"""Inspect this project and create a concrete implementation plan for the task below.
Do not edit files. Include files to change, implementation steps, and tests.

TASK:
{task}
"""

    try:
        with langfuse.start_as_current_observation(
            as_type="agent",
            name="opencode-plan-build-orchestrator",
            input={"task": task, "project": str(project)},
        ) as root:
            trace_id = langfuse.get_current_trace_id()
            record["trace_id"] = trace_id

            with propagate_attributes(
                tags=["opencode", "orchestrator", "plan-build"],
                metadata={"project": project.name},
            ):
                print("[1/2] Planning...")
                plan = run_phase(
                    langfuse=langfuse,
                    phase="plan",
                    agent="plan",
                    prompt=plan_prompt,
                    cwd=project,
                    opencode=cfg.opencode,
                    model=cfg.model,
                    auto=False,
                    timeout=cfg.timeout,
                )
                record["plan"] = plan
                if plan["returncode"] != 0 or not plan["text"]:
                    raise RuntimeError("OpenCode planning failed")

                build_prompt = f"""Implement the task now in the current project.
Use the plan as guidance, edit files, run relevant tests, and finish with a concise summary.
Do not merely explain what should be done.

TASK:
{task}

PLAN:
{plan['text']}
"""
                print("[2/2] Building...")
                build = run_phase(
                    langfuse=langfuse,
                    phase="build",
                    agent="build",
                    prompt=build_prompt,
                    cwd=project,
                    opencode=cfg.opencode,
                    model=cfg.model,
                    auto=True,
                    timeout=cfg.timeout,
                )
                record["build"] = build
                if build["returncode"] != 0 or not build["text"]:
                    raise RuntimeError("OpenCode build failed")

                record["status"] = "completed"
                root.update(
                    output={
                        "status": "completed",
                        "plan_session": plan["session_id"],
                        "build_session": build["session_id"],
                        "summary": build["text"],
                    }
                )

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(record["error"], file=sys.stderr)

    finally:
        langfuse.flush()
        if trace_id:
            try:
                record["trace_url"] = langfuse.get_trace_url(trace_id=trace_id)
                for attempt in range(3):
                    response = langfuse.api.observations.get_many(
                        trace_id=trace_id,
                        fields="core,basic,time,io,metadata,model,usage,metrics,trace_context",
                        limit=1000,
                    )
                    record["langfuse_observations"] = jsonable(response)
                    if record["langfuse_observations"].get("data") or attempt == 2:
                        break
                    time.sleep(1)
            except Exception as exc:
                record["langfuse_export_error"] = str(exc)
        save_json(output, record)
        langfuse.shutdown()

    print(f"Trace JSON: {output}")
    if record.get("trace_url"):
        print(f"Langfuse: {record['trace_url']}")
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
