# Copilot / AI agent instructions for Swarm_v5

Purpose: Give AI coding agents immediate, actionable context for editing and extending this multi-agent Swarm project.

- Architecture (big picture):
  - Coordinator: `src/swarm_coordinator_v2.py` orchestrates workflows, parallel agent execution, clarification loops, and test sandboxing (see `DockerSandbox`).
  - Agents: individual agent programs live under `src/*/*_agent.py` (examples: `src/architect/architect_agent.py`, `src/coder/coder_agent.py`, `src/clarifier/clarifier_agent.py`). Each agent is a standalone script that receives prompts, calls an LLM via an OpenAI-like HTTP API, and returns strict JSON/YAML results.
  - UI & service: `interactive_v2.py` is the local CLI entrypoint; `service.py` and `swarm_coordinator_ui.py` bridge to the web/UI layer and use SSE for progress updates.
  - Project import and workspaces: `src/project_import.py` and the `workspaces/` snapshots implement import/export workflows and example projects.

- Key integration points and conventions (must-follow):
  - Model configuration is read from `config_v2.json` or `config/config_v2.json` (see `interactive_v2.py` and agent modules). Agents expect an OpenAI-compatible endpoint at `<url>/chat/completions`.
  - Agents expect strict output formats. Example: `architect_agent.py` must produce a YAML plan with top-level `program`, `architecture`, `files`, and `execution_order`. See `extract_yaml()` and `validate_plan_schema()` in `src/architect/architect_agent.py` for exact constraints.
  - Filenames and variants are normalized by `normalize_filename()` / `build_filename_lookup()` in `src/architect/architect_agent.py`. When generating filenames, prefer canonical paths (e.g., `package/module.py`) and include `.py` extensions.
  - Do NOT include `config_v2.json` contents in agent outputs — the architect agent enforces this via `FORBIDDEN_FILENAME`.
  - Clarification flow: the coordinator emits progress updates of type `question`. The UI/service calls `coordinator.provide_input('clarification', answer)` (see `SwarmUIService.provide_answer` in `service.py`) to resume.
  - Test execution: `DockerSandbox` (in `src/swarm_coordinator_v2.py`) is used to run hot tests and to `pip install` project requirements. If Docker is unavailable the coordinator falls back to local execution — check `PLAN_EXECUTOR_AVAILABLE` and sandbox flags.

- Common developer workflows (how to run things):
  - Run CLI: `python3 interactive_v2.py` from the repo root (it adds `src/` to `sys.path`).
  - Run coordinator programmatically: instantiate `SwarmCoordinator(config_file='config_v2.json')` in scripts/tests and call `run_workflow(user_request, workflow_type='standard')`.
  - Tests: the repository uses `pytest` in the `test/` folder and dynamic sandbox tests via `DockerSandbox`; ensure `docker` is available for sandboxed tests.
  - Dependencies: check `requirements.txt` at the repo root before running; agent scripts assume `requests` and `pyyaml` are present.

- Patterns and examples agents should follow:
  - HTTP LLM calls: agents POST to `{model_cfg['url'].rstrip('/')}/chat/completions` with `messages` and expect `choices[0].message.content` (see `call_llm()` in `src/architect/architect_agent.py`).
  - Output validation: agents must validate output, return structured JSON/YAML, and use explicit error codes (see `fail()` and `success()` helpers in `src/architect/architect_agent.py`). Re-run with stricter system/user prompts if output is malformed.
  - Plan schema example (architect):

```yaml
program:
  name: example
  description: Example app
  type: cli
architecture:
  pattern: modular
  entry_point: main.py
files:
  - name: main.py
    purpose: app entry
    requirements: []
execution_order: [main.py]
```

- Editing and PR guidance for AI agents:
  - When modifying agent behavior, prefer updating the agent module under `src/<role>/<role>_agent.py` and add unit tests in `test/src/` or a new workspace snapshot under `workspaces/`.
  - Preserve strict output schemas; add normalization via existing helpers rather than ad-hoc parsing.
  - If you change how model calls are made, update `config_v2.json` examples and `interactive_v2.py` checks that display `model_config` settings.

If anything above is unclear or you'd like more examples (e.g., a failing-agent fix or a sample test run), tell me which section to expand.
