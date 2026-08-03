CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

CREATE SCHEMA IF NOT EXISTS episodic;

CREATE TABLE IF NOT EXISTS episodic.raw_traces (
    trace_id text PRIMARY KEY,
    source_kind text NOT NULL,
    source_sha256 text NOT NULL,
    created_at timestamptz,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    payload jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS episodic.episodes (
    episode_id text PRIMARY KEY,
    trace_id text NOT NULL REFERENCES episodic.raw_traces(trace_id),
    project text NOT NULL,
    task text NOT NULL,
    status text NOT NULL,
    summary text NOT NULL,
    memory jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    search_document tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(task, '') || ' ' || coalesce(summary, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS episodes_search_idx
    ON episodic.episodes USING gin (search_document);
CREATE INDEX IF NOT EXISTS episodes_project_idx
    ON episodic.episodes (project, updated_at DESC);
CREATE INDEX IF NOT EXISTS episodes_memory_idx
    ON episodic.episodes USING gin (memory jsonb_path_ops);

CREATE TABLE IF NOT EXISTS episodic.graph_sync (
    episode_id text NOT NULL REFERENCES episodic.episodes(episode_id),
    graph_name text NOT NULL,
    node_count integer NOT NULL,
    edge_count integer NOT NULL,
    synced_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (episode_id, graph_name)
);
