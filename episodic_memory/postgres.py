"""PostgreSQL JSONB and Apache AGE persistence for episodic memories."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .extract import graph_projection

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
NODE_LABELS = {
    "Episode",
    "Project",
    "Ticket",
    "Phase",
    "Artifact",
    "Decision",
    "Obstacle",
    "Outcome",
}
EDGE_TYPES = {
    "OCCURRED_IN",
    "ABOUT",
    "IMPLEMENTED_BY",
    "HAS_PHASE",
    "NEXT",
    "TOUCHED",
    "MADE_DECISION",
    "ENCOUNTERED",
    "PRODUCED",
}


def _driver() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(
            "PostgreSQL loading requires psycopg. Install with: "
            "python -m pip install 'psycopg[binary]>=3.2'"
        ) from exc
    return psycopg, sql


def _validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _node_parameters(item: dict[str, Any]) -> dict[str, Any]:
    properties = item.get("properties") or {}
    return {
        "id": item["id"],
        "kind": str(properties.get("kind") or item["label"].lower()),
        "name": str(properties.get("name") or ""),
        "text": str(properties.get("text") or ""),
        "status": str(properties.get("status") or ""),
        "created_at": str(properties.get("created_at") or ""),
        "trace_id": str(properties.get("trace_id") or ""),
        "path": str(properties.get("path") or ""),
        "phase": str(properties.get("phase") or ""),
        "ticket_key": str(properties.get("ticket_key") or ""),
        "ordinal": int(properties.get("ordinal") or 0),
        "metadata_json": json.dumps(
            properties.get("metadata") or {}, sort_keys=True, ensure_ascii=False
        ),
    }


def _edge_parameters(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "source_id": item["source"],
        "target_id": item["target"],
        "kind": item["type"].lower(),
        "metadata_json": json.dumps(
            item.get("properties") or {}, sort_keys=True, ensure_ascii=False
        ),
    }


def _prepare_graph_statements(cursor: Any, graph_name: str) -> tuple[dict[str, str], dict[str, str]]:
    _, sql = _driver()
    suffix = uuid.uuid4().hex[:10]
    node_statements: dict[str, str] = {}
    edge_statements: dict[str, str] = {}

    for label in sorted(NODE_LABELS):
        statement = f"ep_node_{label.lower()}_{suffix}"
        node_statements[label] = statement
        cypher = f"""
            MERGE (n:{label} {{id: $id}})
            SET n.kind = $kind,
                n.name = $name,
                n.text = $text,
                n.status = $status,
                n.created_at = $created_at,
                n.trace_id = $trace_id,
                n.path = $path,
                n.phase = $phase,
                n.ticket_key = $ticket_key,
                n.ordinal = $ordinal,
                n.metadata_json = $metadata_json
            RETURN n
        """
        cursor.execute(
            sql.SQL(
                "PREPARE {}(agtype) AS "
                "SELECT * FROM ag_catalog.cypher({}, {}, $1) AS (value ag_catalog.agtype)"
            ).format(
                sql.Identifier(statement),
                sql.Literal(graph_name),
                sql.Literal(cypher),
            )
        )

    for edge_type in sorted(EDGE_TYPES):
        statement = f"ep_edge_{edge_type.lower()}_{suffix}"
        edge_statements[edge_type] = statement
        cypher = f"""
            MATCH (source), (target)
            WHERE source.id = $source_id AND target.id = $target_id
            MERGE (source)-[r:{edge_type} {{id: $id}}]->(target)
            SET r.kind = $kind, r.metadata_json = $metadata_json
            RETURN r
        """
        cursor.execute(
            sql.SQL(
                "PREPARE {}(agtype) AS "
                "SELECT * FROM ag_catalog.cypher({}, {}, $1) AS (value ag_catalog.agtype)"
            ).format(
                sql.Identifier(statement),
                sql.Literal(graph_name),
                sql.Literal(cypher),
            )
        )
    return node_statements, edge_statements


def _execute_prepared(cursor: Any, statement: str, parameters: dict[str, Any]) -> None:
    _, sql = _driver()
    payload = json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
    cursor.execute(
        sql.SQL("EXECUTE {} ({}::ag_catalog.agtype)").format(
            sql.Identifier(statement), sql.Literal(payload)
        )
    )
    cursor.fetchall()


def ingest(
    dsn: str,
    trace: dict[str, Any],
    episode: dict[str, Any],
    graph_name: str = "episodic_memory",
) -> dict[str, Any]:
    """Upsert raw/compact records and their graph projection in one transaction."""
    psycopg, _ = _driver()
    graph_name = _validate_identifier(graph_name, "graph name")
    projection = graph_projection(episode)
    for item in projection["nodes"]:
        if item["label"] not in NODE_LABELS:
            raise ValueError(f"Unsupported node label: {item['label']!r}")
    for item in projection["edges"]:
        if item["type"] not in EDGE_TYPES:
            raise ValueError(f"Unsupported edge type: {item['type']!r}")

    schema_sql = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    trace_id = episode["source"]["trace_id"]
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(schema_sql)
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s)",
                (graph_name,),
            )
            if not cursor.fetchone()[0]:
                cursor.execute("SELECT ag_catalog.create_graph(%s)", (graph_name,))

            cursor.execute(
                """
                INSERT INTO episodic.raw_traces
                    (trace_id, source_kind, source_sha256, created_at, payload)
                VALUES (%s, %s, %s, NULLIF(%s, '')::timestamptz, %s::jsonb)
                ON CONFLICT (trace_id) DO UPDATE SET
                    source_kind = EXCLUDED.source_kind,
                    source_sha256 = EXCLUDED.source_sha256,
                    created_at = EXCLUDED.created_at,
                    payload = EXCLUDED.payload,
                    ingested_at = now()
                """,
                (
                    trace_id,
                    episode["source"]["kind"],
                    episode["source"]["raw_sha256"],
                    episode.get("created_at") or "",
                    json.dumps(trace, ensure_ascii=False),
                ),
            )
            cursor.execute(
                """
                INSERT INTO episodic.episodes
                    (episode_id, trace_id, project, task, status, summary, memory)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (episode_id) DO UPDATE SET
                    project = EXCLUDED.project,
                    task = EXCLUDED.task,
                    status = EXCLUDED.status,
                    summary = EXCLUDED.summary,
                    memory = EXCLUDED.memory,
                    updated_at = now()
                """,
                (
                    episode["episode_id"],
                    trace_id,
                    episode.get("project", ""),
                    episode.get("task", ""),
                    episode.get("status", ""),
                    episode.get("summary", ""),
                    json.dumps(episode, ensure_ascii=False),
                ),
            )

            node_statements, edge_statements = _prepare_graph_statements(
                cursor, graph_name
            )
            for item in projection["nodes"]:
                _execute_prepared(
                    cursor, node_statements[item["label"]], _node_parameters(item)
                )
            for item in projection["edges"]:
                _execute_prepared(
                    cursor, edge_statements[item["type"]], _edge_parameters(item)
                )

            cursor.execute(
                """
                INSERT INTO episodic.graph_sync
                    (episode_id, graph_name, node_count, edge_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (episode_id, graph_name) DO UPDATE SET
                    node_count = EXCLUDED.node_count,
                    edge_count = EXCLUDED.edge_count,
                    synced_at = now()
                """,
                (
                    episode["episode_id"],
                    graph_name,
                    len(projection["nodes"]),
                    len(projection["edges"]),
                ),
            )

    return {
        "trace_id": trace_id,
        "episode_id": episode["episode_id"],
        "graph_name": graph_name,
        "nodes": len(projection["nodes"]),
        "edges": len(projection["edges"]),
    }
