# Migration Guide: Swarm v1 → v2

## Overview

The v2 swarm coordinator introduces significant architectural improvements while maintaining backward compatibility with your existing config.json format.

---

## Key Improvements in v2

### 1. **Parallel Execution**
- Multiple agents can now execute simultaneously
- Configurable parallelism (`max_parallel_agents`)
- Automatic dependency resolution prevents race conditions

**Before (v1):**
```
Clarifier → Coder → Reviewer1 → Reviewer2 → Reviewer3
(Sequential, ~5-10 minutes)
```

**After (v2):**
```
Clarifier → Architect → Coder → [Reviewer1, Reviewer2, Reviewer3]
                                  (Parallel, ~2-3 minutes)
```

### 2. **Enhanced Agent Specialization**
New specialized agent roles:
- **ARCHITECT** - System design decisions
- **TESTER** - Test generation
- **OPTIMIZER** - Performance optimization
- **DOCUMENTER** - Documentation generation
- **DEBUGGER** - Bug analysis
- **SECURITY** - Security auditing

### 3. **Task-Based Workflow**
- Explicit task DAG (Directed Acyclic Graph)
- Dynamic task routing
- Dependency management
- Better failure recovery

### 4. **Metrics & Observability**
- Per-agent performance tracking
- Token usage monitoring
- Response time analytics
- Success/failure rates

### 5. **Flexible Workflows**
Three built-in workflow types:
- **STANDARD** - Fast, focused (clarify → architect → code → review)
- **FULL** - Comprehensive (all agents)
- **REVIEW** - Code review only (4 parallel reviewers)
- **CUSTOM** - Build your own workflow

---

## Migration Steps

### Step 1: Update Configuration

Your existing `config.json` is compatible! Just rename it:

```bash
cp config.json config_v1_backup.json
cp config_v2.json config.json  # Use the new template
```

**Key changes in config_v2.json:**

```json
{
  "workflow": {
    "max_parallel_agents": 4,      // NEW: Control parallelism
    "enable_parallel": true         // NEW: Enable/disable parallel execution
  },
  "model_config": {
    "multi_model": {
      "architect": { ... },         // NEW: Additional agent roles
      "tester": { ... },
      "optimizer": { ... },
      "documenter": { ... },
      "debugger": { ... },
      "security": { ... }
    }
  }
}
```

### Step 2: Map Your Current Setup

**If you're using LM Studio (port 1234):**
Your setup maps directly. Just add new agent roles to the config.

**If you're using Ollama (port 11434):**
Update the `api_type` for each agent to `"ollama"`.

**Example mapping:**
```
v1 "clarifier" → v2 "clarifier" + "architect"
v1 "coder"     → v2 "coder"
v1 "reviewer"  → v2 "reviewer" + "security" + "tester"
```

### Step 3: Update Your Scripts

**Old way (v1):**
```python
from coordinator import AgentCoordinator

coordinator = AgentCoordinator()
coordinator.run_workflow(user_request)
```

**New way (v2):**
```python
from swarm_coordinator_v2 import SwarmCoordinator

coordinator = SwarmCoordinator(config_file="config.json")
coordinator.run_workflow(user_request, workflow_type="standard")
```

### Step 4: Run Interactive CLI

**v1:**
```bash
python interactive.py
```

**v2:**
```bash
python interactive_v2.py
```

The v2 interface gives you workflow selection, better progress monitoring, and more control.

---

## Feature Comparison

| Feature | v1 | v2 |
|---------|----|----|
| Parallel execution | ❌ | ✅ |
| Task dependencies | ❌ | ✅ |
| Agent specialization | 3 roles | 9 roles |
| Metrics tracking | Basic | Comprehensive |
| Custom workflows | ❌ | ✅ |
| Recovery mechanisms | Basic | Advanced |
| Real-time monitoring | ❌ | ✅ (via manager) |
| Session management | JSON only | JSON + CSV export |

---

## Backward Compatibility

### What's Compatible:
✅ Your existing `config.json` structure
✅ Your LM Studio/Ollama setup
✅ Your model selections
✅ Agent parameters (temperature, max_tokens, etc.)

### What Changed:
⚠ Import path: `coordinator.py` → `swarm_coordinator_v2.py`
⚠ Class name: `AgentCoordinator` → `SwarmCoordinator`
⚠ New required parameter: `workflow_type` in `run_workflow()`

---

## Performance Expectations

### Resource Usage

**v1 (Sequential):**
- CPU: One model at a time
- Memory: One model loaded
- Time: 5-10 minutes for full workflow
- VRAM: ~6-8GB (single model)

**v2 (Parallel - 4 agents):**
- CPU: Up to 4 models simultaneously
- Memory: Multiple models may be loaded
- Time: 2-4 minutes for full workflow
- VRAM: ~12-24GB (multiple models)

### Recommendations

**For F17 Laptop (Limited resources):**
```json
{
  "workflow": {
    "enable_parallel": false,    // Sequential execution
    "max_parallel_agents": 1
  }
}
```

**For Server (256GB RAM, Tesla P40s):**
```json
{
  "workflow": {
    "enable_parallel": true,     // Full parallel
    "max_parallel_agents": 6
  }
}
```

---

## Common Patterns

### Pattern 1: Quick Code Generation
```python
coordinator = SwarmCoordinator()
coordinator.run_workflow(request, workflow_type="standard")
# Uses: Clarifier → Architect → Coder → 3 parallel reviewers
```

### Pattern 2: Comprehensive Development
```python
coordinator = SwarmCoordinator()
coordinator.run_workflow(request, workflow_type="full")
# Uses all 9 agent types with parallel QA
```

### Pattern 3: Code Review Only
```python
coordinator = SwarmCoordinator()
coordinator.run_workflow(code, workflow_type="review_only")
# 4 parallel reviewers: correctness, security, performance, style
```

### Pattern 4: Custom Workflow
```python
# Build your own task graph
coordinator = SwarmCoordinator()
coordinator.add_task(Task(...))
coordinator.add_task(Task(...))
# Execute with custom dependencies
```

---

## Troubleshooting

### Issue: "Too many parallel requests"
**Solution:** Reduce `max_parallel_agents` in config.

### Issue: "Out of memory"
**Solution:** Set `enable_parallel: false` or reduce parallel count.

### Issue: "Model timeout"
**Solution:** Increase `timeout` in model config (default 7200s).

### Issue: "Task blocked"
**Solution:** Check task dependencies - circular dependencies will block workflow.

---

## Management Tools

### Analyze Previous Runs
```bash
python swarm_manager.py
# Select option 2: Analyze session
```

### Export Metrics
```bash
python swarm_manager.py
# Select option 4: Export metrics to CSV
```

### Compare Workflows
```bash
python swarm_manager.py
# Select option 3: Compare sessions
```

---

## Next Steps

1. **Test with Standard Workflow:**
   ```bash
   python interactive_v2.py
   # Select option 1: STANDARD
   ```

2. **Review Metrics:**
   Check the generated `swarm_state_*.json` files to see agent performance.

3. **Optimize Configuration:**
   Adjust parallelism and agent assignments based on your hardware.

4. **Build Custom Workflows:**
   Create task graphs specific to your petrochemical documentation needs.

---

## Getting Help

### Check Session Logs
```bash
# View latest session
python -c "import glob; print(max(glob.glob('swarm_state_*.json')))"
# Analyze with manager
python swarm_manager.py
```

### Enable Verbose Logging
```json
{
  "workflow": {
    "verbose_logging": true
  }
}
```

### Debug Mode
Set environment variable:
```bash
export SWARM_DEBUG=1
python interactive_v2.py
```

---

## Rollback Plan

If you need to revert to v1:

1. Restore config:
   ```bash
   cp config_v1_backup.json config.json
   ```

2. Use original scripts:
   ```bash
   python interactive.py  # v1
   ```

Your v1 and v2 systems can coexist - they use different session files.

---

## Questions?

Common questions and solutions are in the main README.md file.

For Elara project integration, see the specialized workflow examples for document parsing.
