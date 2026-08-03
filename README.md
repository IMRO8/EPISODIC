# OpenCode tracing with Langfuse

This directory is configured to send OpenCode session telemetry to Langfuse
through OpenTelemetry. Traces include prompts, responses, reasoning, and tool
inputs/outputs, so do not use tracing for data that should not be stored in
Langfuse.

## Start a traced session

```sh
./opencode-with-langfuse
```

The launcher loads the credentials in `.env`, changes to this directory, and
starts OpenCode. You can also pass normal OpenCode arguments, for example:

```sh
./opencode-with-langfuse run "Reply with: telemetry is working"
```

Completed turns should appear in the Langfuse project for the configured US
region. Restart OpenCode after changing `opencode.json` or `.env`.

## Configuration

- `opencode.json` enables experimental OpenTelemetry and the Langfuse plugin.
- `.env` contains local Langfuse credentials and is excluded from Git.
- `LANGFUSE_BASEURL` must match the region where the project was created.

Optional labels can be added to `.env`:

```sh
export LANGFUSE_ENVIRONMENT="development"
export LANGFUSE_USER_ID="your-user-id"
```
