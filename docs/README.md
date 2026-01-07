# Advanced Multi-Agent Swarm Coordinator v2

Production-ready multi-agent system with parallel execution, dynamic routing, and comprehensive observability.

---

## Features

### 🚀 Core Capabilities
- **Parallel Agent Execution** - Run multiple agents simultaneously
- **9 Specialized Agent Roles** - From architecture to security
- **Dynamic Task Routing** - Automatic dependency resolution
- **Comprehensive Metrics** - Track performance, tokens, response times
- **Flexible Workflows** - Standard, Full, Review-only, or Custom
- **Production Ready** - Error handling, recovery, checkpointing

### 🤖 Agent Roles

| Role | Purpose | Example Use |
|------|---------|-------------|
| **ARCHITECT** | System design, tech stack decisions | Design microservice architecture |
| **CLARIFIER** | Requirements analysis | Extract missing specs from vague requests |
| **CODER** | Implementation | Write clean, documented code |
| **REVIEWER** | Code quality assurance | Find bugs, suggest improvements |
| **TESTER** | Test generation | Create unit, integration, edge case tests |
| **OPTIMIZER** | Performance tuning | Identify bottlenecks, optimize algorithms |
| **DOCUMENTER** | Documentation | API docs, user guides, README files |
| **DEBUGGER** | Bug analysis | Root cause analysis, fix suggestions |
| **SECURITY** | Security audit | Identify vulnerabilities, best practices |

---

## Quick Start

### 1. Installation

```bash
# No dependencies beyond requests
pip install requests --break-system-packages
```

### 2. Configuration

Copy the example config:
```bash
cp config_v2.json config.json
```

Edit `config.json` to match your setup:
- Update URLs for LM Studio (port 1234) or Ollama (port 11434)
- Set model names for each agent role
- Adjust `max_parallel_agents` for your hardware

### 3. Run Interactive Mode

```bash
python interactive_v2.py
```

Select workflow type:
1. **STANDARD** - Fast pipeline (clarify → architect → code → review)
2. **FULL** - All agents for comprehensive development
3. **REVIEW** - 4 parallel code reviews
4. **CUSTOM** - Build your own task graph

### 4. Enter Your Request

```
What would you like to build?
================================================================================
(Type your input. Enter 'END' on a new line when done)
--------------------------------------------------------------------------------
Create a Python class for managing a FIFO queue with these features:
- Thread-safe operations
- Max capacity with blocking
- Timeout support
- Statistics tracking (items in/out, avg wait time)
END
```

---

## Workflow Types

### Standard Workflow
**Best for:** Quick code generation with review
**Duration:** ~2-4 minutes
**Agents:** 5 (Clarifier, Architect, Coder, 3 Reviewers)

```python
coordinator.run_workflow(request, workflow_type="standard")
```

**Task Graph:**
```
Clarifier → Architect → Coder → [Reviewer1, Reviewer2, Reviewer3]
                                  (Parallel)
```

### Full Workflow
**Best for:** Production code with all checks
**Duration:** ~5-8 minutes
**Agents:** 8 (All roles)

```python
coordinator.run_workflow(request, workflow_type="full")
```

**Task Graph:**
```
Clarifier → Architect → Coder → [Reviewer, Security, Tester]
                                  (Parallel)
                     ↓
            [Optimizer, Documenter]
             (Parallel)
```

### Review Workflow
**Best for:** Analyzing existing code
**Duration:** ~1-2 minutes
**Agents:** 4 (Parallel reviewers)

```python
coordinator.run_workflow(code, workflow_type="review_only")
```

**Task Graph:**
```
[Reviewer-Correctness, Reviewer-Security, Reviewer-Performance, Reviewer-Style]
(All parallel)
```

### Custom Workflow
**Best for:** Specialized use cases
**Duration:** Depends on tasks
**Agents:** Your choice

```python
coordinator = SwarmCoordinator()

# Define custom tasks
coordinator.add_task(Task(
    task_id="T001",
    task_type="analysis",
    description="Analyze requirements",
    assigned_role=AgentRole.CLARIFIER,
    status=TaskStatus.PENDING,
    priority=10
))

coordinator.add_task(Task(
    task_id="T002",
    task_type="coding",
    description="Implement solution",
    assigned_role=AgentRole.CODER,
    status=TaskStatus.PENDING,
    priority=8,
    dependencies=["T001"]  # Wait for T001 to complete
))

# Execute
coordinator.run_workflow("", workflow_type="custom")
```

---

## Configuration Reference

### Model Configuration

```json
{
  "model_config": {
    "mode": "multi",  // "single" or "multi"
    
    "single_model": {
      "url": "http://localhost:1234/v1",
      "model": "qwen2.5-coder:7b",
      "api_type": "openai",  // "openai" or "ollama"
      "timeout": 7200
    },
    
    "multi_model": {
      "coder": {
        "url": "http://localhost:1234/v1",
        "model": "openai/gpt-oss-20b",
        "model_name": "GPT-OSS-20B (Display name)",
        "api_type": "openai",
        "timeout": 7200
      },
      "reviewer": {
        "url": "http://localhost:11434",
        "model": "qwen2.5-coder:7b",
        "model_name": "Qwen Coder",
        "api_type": "ollama",
        "timeout": 7200
      }
      // ... other agents
    }
  }
}
```

### Agent Parameters

```json
{
  "agent_parameters": {
    "coder": {
      "temperature": 0.5,     // Lower = more deterministic
      "max_tokens": 6000,     // Max response length
      "top_p": 0.85          // Nucleus sampling
    },
    "reviewer": {
      "temperature": 0.8,     // Higher = more creative/thorough
      "max_tokens": 3000,
      "top_p": 0.95
    }
  }
}
```

### Workflow Settings

```json
{
  "workflow": {
    "max_iterations": 3,           // Max review cycles
    "max_parallel_agents": 4,      // Max concurrent agents
    "enable_parallel": true,       // Enable parallel execution
    "verbose_logging": true        // Detailed logs
  }
}
```

---

## Hardware Recommendations

### Minimal Setup (F17 Laptop)
```json
{
  "workflow": {
    "enable_parallel": false,
    "max_parallel_agents": 1
  }
}
```
- Sequential execution
- One model at a time
- ~6-8GB VRAM required
- Longer execution times

### Optimal Setup (Server with GPUs)
```json
{
  "workflow": {
    "enable_parallel": true,
    "max_parallel_agents": 4
  }
}
```
- Parallel execution
- Multiple models loaded
- ~16-24GB VRAM required
- Faster execution times

### High-Performance Setup
```json
{
  "workflow": {
    "enable_parallel": true,
    "max_parallel_agents": 8
  }
}
```
- Max parallelism
- All agents can run simultaneously
- ~32-48GB VRAM required
- Fastest execution

---

## API Usage

### Basic Usage

```python
from swarm_coordinator_v2 import SwarmCoordinator

coordinator = SwarmCoordinator(config_file="config.json")

# Run standard workflow
coordinator.run_workflow(
    user_request="Create a CSV parser with error handling",
    workflow_type="standard"
)

# Access results
code_task = next(t for t in coordinator.completed_tasks if t.task_type == "coding")
print(code_task.result)

# Get metrics
metrics = coordinator.get_metrics_summary()
print(f"Total tokens used: {sum(m['total_tokens'] for m in metrics['agents'].values())}")
```

### Advanced Usage

```python
from swarm_coordinator_v2 import (
    SwarmCoordinator, Task, AgentRole, TaskStatus
)

coordinator = SwarmCoordinator()

# Build custom workflow for Elara LAO document parsing
coordinator.add_task(Task(
    task_id="T001_analyze",
    task_type="document_analysis",
    description="Analyze LAO document structure",
    assigned_role=AgentRole.ARCHITECT,
    status=TaskStatus.PENDING,
    priority=10,
    metadata={"document_type": "LAO", "count": 287}
))

coordinator.add_task(Task(
    task_id="T002_extract",
    task_type="extraction",
    description="Extract procedure sections",
    assigned_role=AgentRole.CODER,
    status=TaskStatus.PENDING,
    priority=9,
    dependencies=["T001_analyze"]
))

# Run custom workflow
coordinator.run_workflow("", workflow_type="custom")

# Save detailed results
coordinator.save_state("elara_session.json")
```

---

## Management Tools

### Swarm Manager

Analyze and manage workflow executions:

```bash
python swarm_manager.py
```

**Features:**
- List all sessions
- Analyze individual sessions
- Compare sessions
- Export metrics to CSV
- Generate comprehensive reports
- Cleanup old sessions

### Session Analysis

```python
from swarm_manager import SwarmManager

manager = SwarmManager()

# Analyze latest session
latest_session = manager.session_files[0]
manager.analyze_session(latest_session)

# Export all metrics
manager.export_metrics("all_metrics.csv")

# Generate report
manager.generate_report("swarm_report.txt")
```

---

## Output Files

### Generated Files

| File | Content | When Created |
|------|---------|--------------|
| `generated_code.txt` | Final code output | After coding task |
| `generated_tests.txt` | Test suite | After tester task |
| `generated_docs.txt` | Documentation | After documenter task |
| `review_results.txt` | All review feedback | After review tasks |
| `swarm_state_*.json` | Complete session state | Every workflow |
| `swarm_metrics.csv` | Metrics export | Via manager |
| `swarm_report.txt` | Analysis report | Via manager |

### Session State Structure

```json
{
  "workflow_id": "20231205_143022",
  "phase": "complete",
  "iteration": 2,
  "context": {
    "coding_coder": {
      "task_id": "T003_code",
      "result": "...",
      "completed_at": 1234567890.123
    }
  },
  "completed_tasks": [
    {
      "task_id": "T001_clarify",
      "status": "COMPLETED",
      "result": "...",
      "created_at": 1234567880.0,
      "completed_at": 1234567885.5
    }
  ],
  "metrics": {
    "coder": {
      "total_calls": 3,
      "successful_calls": 3,
      "avg_response_time": 12.5,
      "total_tokens": 4500
    }
  }
}
```

---

## Best Practices

### 1. Resource Management

**Monitor VRAM usage:**
```bash
# While swarm is running
watch -n 1 nvidia-smi
```

**Adjust parallelism based on load:**
- 4 agents = ~4x memory usage
- Start conservative, scale up
- Monitor system responsiveness

### 2. Agent Selection

**Match agent to model strength:**
- **Coder role** → Strongest coding model (e.g., GPT-OSS-20B, Qwen Coder)
- **Reviewer role** → Different model for diversity
- **Tester role** → Lighter model (tests are simpler)

### 3. Temperature Tuning

| Task Type | Temperature | Reason |
|-----------|-------------|--------|
| Code generation | 0.3-0.5 | Deterministic, correct |
| Code review | 0.7-0.9 | Creative, thorough |
| Requirements | 0.6-0.8 | Balanced |
| Documentation | 0.6-0.7 | Clear but engaging |

### 4. Timeout Settings

**Adjust based on task complexity:**
- Simple scripts: 60-120s
- Complex code: 300-600s
- Large documents: 600-1200s
- Full workflows: 3600-7200s

### 5. Error Recovery

**Enable checkpointing:**
```json
{
  "features": {
    "enable_checkpointing": true
  }
}
```

**Monitor failed tasks:**
```python
failed = [t for t in coordinator.completed_tasks if t.status == TaskStatus.FAILED]
for task in failed:
    print(f"Failed: {task.task_id} - {task.error}")
```

---

## Petrochemical Use Cases

### LAO Document Parsing (Elara Project)

```python
# Parse 287 LAO procedures
coordinator = SwarmCoordinator()

# Custom workflow for batch processing
for doc_num in range(1, 288):
    coordinator.add_task(Task(
        task_id=f"T{doc_num:03d}_parse",
        task_type="document_parse",
        description=f"Parse LAO document {doc_num}",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=5,
        metadata={"doc_id": doc_num}
    ))

# Process in batches of 10
batch_size = 10
# ... implement batching logic
```

### Process Troubleshooting

```python
# Analyze process issues with multiple agents
request = """
Distillation column showing erratic pressure control.
Recent changes: Feed composition shift, new PID tuning.
Historical data shows correlation with ambient temperature.
"""

coordinator.run_workflow(request, workflow_type="full")

# Security agent checks for safety issues
# Reviewer checks logic
# Debugger analyzes root cause
# Architect suggests system improvements
```

### Procedure Generation

```python
request = """
Generate startup procedure for LAO reactor:
- Safety interlocks verification
- Equipment lineup
- Pressure/temperature ramp rates
- Quality checks at each stage
"""

coordinator.run_workflow(request, workflow_type="standard")

# Reviewer ensures safety compliance
# Documenter creates operator-friendly format
```

---

## Troubleshooting

### Common Issues

**Issue:** "Connection refused" to LM Studio/Ollama
```bash
# Check if server is running
curl http://localhost:1234/v1/models  # LM Studio
curl http://localhost:11434/api/tags  # Ollama
```

**Issue:** Out of memory
```json
{
  "workflow": {
    "enable_parallel": false  // Disable parallel execution
  }
}
```

**Issue:** Tasks stuck in PENDING
- Check dependencies: `task.dependencies`
- Ensure no circular dependencies
- Check logs for errors

**Issue:** Poor code quality
- Increase coder temperature (but not too high)
- Use stronger model for coder role
- Add more reviewers

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

coordinator = SwarmCoordinator()
```

---

## Performance Metrics

### Typical Performance (Standard Workflow)

| Hardware | Parallel | Duration | Memory |
|----------|----------|----------|--------|
| F17 Laptop | No | 8-12 min | 8GB |
| F17 Laptop | Yes (2) | 5-7 min | 12GB |
| Server (P40) | Yes (4) | 2-4 min | 20GB |
| Server (P40) | Yes (8) | 1.5-3 min | 32GB |

### Token Usage

| Workflow | Tasks | Avg Tokens | Cost (if cloud) |
|----------|-------|------------|-----------------|
| Standard | 5 | 15,000 | $0.15 |
| Full | 8 | 35,000 | $0.35 |
| Review | 4 | 8,000 | $0.08 |
| Custom | Varies | Varies | Varies |

*Note: You're running local models so cost is $0*

---

## Roadmap

### Planned Features
- [ ] Streaming responses for real-time feedback
- [ ] Tool calling (agents can use external tools)
- [ ] Persistent memory across sessions
- [ ] Web UI for monitoring
- [ ] Integration with ChromaDB/vector stores
- [ ] Automatic model selection based on task
- [ ] Cost optimization (token usage tracking)
- [ ] A/B testing different agent configurations

---

## License

This is your proprietary system for the petrochemical plant. Keep it internal.

---

## Support

For issues or questions, check:
1. Session logs: `swarm_state_*.json`
2. Metrics: Run `python swarm_manager.py`
3. Migration guide: `MIGRATION.md`

---

**Built for production use in air-gapped petrochemical environments.**
