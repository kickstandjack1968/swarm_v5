# Swarm v5 Design Document

## Executive Summary

Swarm v5 is a multi-agent orchestration system that automates software development workflows using specialized AI agents. It takes a user request and produces complete project outputs (code, tests, documentation) by coordinating 11 agent roles through task-based execution.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     USER REQUEST                              │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              SWARM COORDINATOR (3,885 lines)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ State Mgmt  │  │  Task Queue  │  │ Progress Callbacks│   │
│  └─────────────┘  └──────────────┘  └───────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    AGENT EXECUTOR                             │
│   Subprocess calls to external agent scripts                  │
│   JSON stdin/stdout protocol                                  │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  AGENTS (11 roles, separate Python scripts)                  │
│  clarifier | architect | coder | reviewer | tester           │
│  documenter | verifier | security | optimizer | debugger     │
│  data_analyst                                                 │
└──────────────────────────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT: src/*.py | tests/*.py | docs/ | requirements.txt   │
└──────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. SwarmCoordinator (`swarm_coordinator_v2.py`)
The central 3,885-line monolith handling:
- Workflow selection and task DAG creation
- Task execution with dependency resolution
- State persistence across phases
- Docker sandbox integration for testing
- File output generation

### 2. Agent Execution Model
Agents run as **external subprocess scripts** with:
- JSON input via stdin
- JSON output via stdout
- Shared base class (`AgentBase`) for LLM calls
- Per-agent model configuration (multi-model support)

### 3. Task System
```python
@dataclass
class Task:
    task_id: str
    task_type: str
    assigned_role: AgentRole
    status: TaskStatus  # pending/in_progress/completed/failed
    dependencies: List[str]  # task_ids this depends on
    result: Optional[str]
    revision_count: int
    max_revisions: int = 3
```

### 4. State Management
```python
self.state = {
    "phase": "initial|planning|implementation|qa|complete",
    "swarm_state": SwarmState.IDLE|RUNNING|WAITING_FOR_INPUT|COMPLETED|FAILED,
    "context": {
        "user_request": str,
        "plan_yaml": str,      # from architect
        "job_scope": str,      # from clarifier
        "code": str,           # from coder
    }
}
```

---

## Workflow Types

| Workflow | Stages | Use Case |
|----------|--------|----------|
| **STANDARD** | Clarify → Architect → Code → Review×3 → Document → Verify | Basic projects |
| **PLANNED** | YAML-driven file-by-file execution | Complex multi-file |
| **FULL** | All agents including security/optimizer | Production quality |
| **IMPORT** | Analyze existing project → modify | Existing codebases |
| **BUGFIX** | Diagnose → fix → verify | Bug fixing |

---

## Configuration (`config_v2.json`)

**Multi-model support** - different LLMs per agent:
```json
{
  "multi_model": {
    "architect": { "model": "qwen/qwen3-coder-30b", "temperature": 0.5 },
    "coder": { "model": "qwen/qwen3-coder-30b", "temperature": 0.2 },
    "reviewer": { "model": "...", "temperature": 0.8 }
  }
}
```

---

## Notable Design Decisions

### What Works
1. **Explicit task DAG** - Clear dependency declarations, easy to debug
2. **Multi-model support** - Per-agent model/temperature tuning
3. **Docker sandbox** - Isolated test execution
4. **Smoke testing** - pytest + CLI verification before final approval
5. **Revision cycles** - Reviewer can reject code, triggers coder retry (max 3)

### Pain Points

| Issue | Details |
|-------|---------|
| **Monolithic coordinator** | 3,885 lines mixing state, execution, prompts, I/O |
| **Subprocess overhead** | Process startup + JSON serialization per agent call |
| **Weak context typing** | `self.state["context"]` is untyped dict, easy to miss fields |
| **Limited parallelism** | `max_parallel_agents=1` by default, not utilizing concurrency |
| **Custom file markers** | `### FILE:` format requires regex parsing, fragile |
| **No distributed execution** | Single machine, long timeouts (7200s) |

---

## Data Flow Example (STANDARD workflow)

```
1. User Request
      ↓
2. Clarifier: asks questions → user answers → synthesizes job_scope
      ↓
3. Architect: generates YAML plan (files, exports, dependencies)
      ↓
4. Coder: generates code following plan (### FILE: markers)
      ↓
5. Reviewers (×3 parallel): check code → APPROVED or NEEDS_REVISION
      ↓ (revision loop if rejected, max 3)
6. Documenter: generates README
      ↓
7. Verifier: checks docs match code + runs smoke tests
      ↓
8. Output: project directory with all files
```

---

## File Structure

```
swarm_v5/
├── src/
│   ├── swarm_coordinator_v2.py   # Main coordinator (3,885 lines)
│   ├── plan_executor_v2.py       # YAML plan execution (1,542 lines)
│   ├── agent_base.py             # Shared agent infrastructure
│   ├── architect/architect_agent.py
│   ├── clarifier/clarifier_agent.py
│   ├── coder/coder_agent.py
│   ├── reviewer/reviewer_agent.py
│   ├── tester/tester_agent.py
│   ├── verifier/verifier_agent.py
│   └── ... (other agents)
├── config/config_v2.json         # Multi-model configuration
├── interactive_v2.py             # Terminal CLI
├── service.py                    # Flask/FastAPI service
└── projects/                     # Generated output
```

---

## Key Metrics Tracked

- Per-agent: call count, success rate, avg response time, total tokens
- Per-workflow: total duration, revision count, phase transitions
- Per-task: status history, error details

---

## Recommendations for Redo

1. **Split the coordinator** into smaller modules (state, execution, prompts, I/O)
2. **Replace subprocess with async workers** (FastAPI + message queue)
3. **Use structured output** (JSON/YAML) instead of `### FILE:` markers
4. **Add TypedDict/Pydantic** for context and task schemas
5. **Enable real parallelism** for independent tasks
6. **Consider distributed execution** for scalability
7. **Add observability** (OpenTelemetry, structured logging)
