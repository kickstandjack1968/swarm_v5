# SwarmCoordinator V2 - Plan Enforcement Fixes

## Overview

This package contains comprehensive fixes for the SwarmCoordinator multi-agent system. The core issue was that the **YAML plan created by the architect was being treated as suggestions rather than a contract**. These fixes make the plan the single source of truth.

## Files Created

```
swarm_v4/
├── agent_base.py              # NEW - Shared agent infrastructure
├── plan_executor_v2.py        # NEW - Rewritten with plan enforcement
├── swarm_coordinator_patches.py  # Patches for swarm_coordinator_v2.py
└── src/
    ├── architect/
    │   └── architect_agent.py  # UPDATED
    ├── coder/
    │   └── coder_agent.py      # UPDATED
    ├── clarifier/
    │   └── clarifier_agent.py  # UPDATED
    ├── verifier/
    │   └── verifier_agent.py   # UPDATED
    ├── reviewer/
    │   └── reviewer_agent.py   # UPDATED
    └── tester/
        └── tester_agent.py     # UPDATED
```

## Key Changes

### 1. Plan as Single Source of Truth

**Before:** Imports were computed from what previous files *actually* exported.  
**After:** Imports are generated directly from the plan's `imports_from` field.

**Before:** Export names could drift (plan says `AppConfig`, coder outputs `Settings`).  
**After:** Code is **rejected** if exports don't match the plan exactly.

### 2. Mandatory Export Format

**Before:**
```
Requirements:
  - Create a Settings class
  - Handle configuration loading
```

**After:**
```
MANDATORY EXPORTS - CODE REJECTED IF MISSING
============================================
class Settings:  # REQUIRED
    def __init__(self, config_path: str = None):  # REQUIRED
    def get(self, key: str) -> Any:  # REQUIRED

WARNING: If ANY of the above are missing or renamed,
your code will be REJECTED and you must regenerate.
```

### 3. Pre-Generated Import Statements

**Before:** Coder had to figure out imports from context.  
**After:** Exact import statements provided:

```
COPY THIS IMPORT BLOCK TO YOUR FILE:
----------------------------------------
from .config import Settings
from parsers.pdf_parser import PDFProcessor
----------------------------------------
```

### 4. Plan Compliance Verification Gate

New verification step before accepting any file:
- ✓ Syntax validation (AST parse)
- ✓ Export compliance (all planned exports present)
- ✓ Import validation (only imports what's available)
- ✓ Placeholder detection (no `pass`, `...`, `NotImplementedError`)
- ✓ Integration check (call signatures match)

### 5. Auto-Generated `__init__.py`

Files like `parsers/pdf_parser.py` now automatically get `parsers/__init__.py`:

```python
"""Auto-generated __init__.py"""
from .pdf_parser import PDFProcessor
from .docx_parser import DocxProcessor

__all__ = ['PDFProcessor', 'DocxProcessor']
```

### 6. Shared Agent Infrastructure

All agents now use `agent_base.py`:
- Common LLM calling with retries
- Standardized error handling
- Export/signature extraction utilities

## Integration Steps

### Step 1: Copy New Files

```bash
# Copy to your swarm_v4 directory
cp agent_base.py /path/to/swarm_v4/
cp plan_executor_v2.py /path/to/swarm_v4/

# Copy updated agents
cp -r src/* /path/to/swarm_v4/src/
```

### Step 2: Update Imports in swarm_coordinator_v2.py

Find (~line 39):
```python
from plan_executor import (
    PlanExecutor, 
    create_planned_workflow,
    ...
)
```

Replace with:
```python
from plan_executor_v2 import (
    PlanExecutor, 
    create_planned_workflow,
    execute_plan_task,
    ARCHITECT_PLAN_SYSTEM_PROMPT,
    get_architect_plan_prompt,
    extract_yaml_from_response
)
```

### Step 3: Update _update_context Method

Find `_update_context` (~line 3489) and add plan storage:

```python
def _update_context(self, task: Task):
    """Update shared context with task results"""
    context_key = f"{task.task_type}_{task.assigned_role.value}"
    self.state["context"][context_key] = {
        "task_id": task.task_id,
        "result": task.result,
        "completed_at": task.completed_at
    }
    
    # Store latest code
    if task.task_type in ("coding", "revision", "plan_execution") and task.result:
        self.state["context"]["latest_code"] = task.result
    
    # NEW: Store parsed plan
    if task.task_type == "architecture_plan" and task.result:
        try:
            import yaml
            parsed = yaml.safe_load(task.result)
            self.state["context"]["parsed_plan"] = parsed
            self.state["context"]["plan_exports"] = {
                f.get('name'): [e.get('name') for e in f.get('exports', [])]
                for f in parsed.get('files', [])
            }
        except:
            pass
```

### Step 4: Add Init File Generation

At the end of `_save_project_outputs`, add:

```python
# Auto-generate __init__.py files
self._generate_init_files(project_dir)
```

And add this method:

```python
def _generate_init_files(self, project_dir: str):
    """Auto-generate __init__.py for folders with Python files."""
    src_dir = os.path.join(project_dir, "src")
    if not os.path.exists(src_dir):
        return
    
    folders_with_py = set()
    for root, dirs, files in os.walk(src_dir):
        if any(f.endswith('.py') for f in files):
            folders_with_py.add(root)
            parent = root
            while parent != src_dir:
                parent = os.path.dirname(parent)
                if parent:
                    folders_with_py.add(parent)
    
    for folder in folders_with_py:
        init_path = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_path):
            py_files = [f for f in os.listdir(folder) 
                       if f.endswith('.py') and f != '__init__.py']
            
            lines = ['"""Auto-generated __init__.py"""', '']
            for py_file in py_files:
                module = py_file[:-3]
                lines.append(f"from .{module} import *")
            
            with open(init_path, 'w') as f:
                f.write('\n'.join(lines))
```

### Step 5: Test

```python
from swarm_coordinator_v2 import SwarmCoordinator

coordinator = SwarmCoordinator()
coordinator.run_workflow(
    "Create a PDF text extractor CLI tool",
    workflow_type="planned"
)
```

## What Each Fix Addresses

| Issue | Fix Location | Description |
|-------|--------------|-------------|
| 1.1 Export enforcement | `plan_executor_v2._validate_exports_against_plan()` | Rejects files that don't export what plan requires |
| 1.2 Imports from plan | `plan_executor_v2._generate_import_block()` | Generates imports from plan, not actuals |
| 1.3 Mandatory format | `plan_executor_v2._generate_mandatory_exports()` | Formats exports as contract |
| 1.4 Compliance gate | `plan_executor_v2._verify_plan_compliance()` | Comprehensive check before accepting |
| 2.1 Import paths | Pre-generated exact statements | No LLM guessing |
| 2.3 __init__.py | `_generate_init_files()` | Auto-creates for folders |
| 3.1 Job scope | Structured format | Both prose and checkpoints |
| 3.2 Plan in context | `_update_context()` patch | Stores parsed plan |
| 5.1 Coder prompt | `_get_coder_system_prompt()` | References plan-driven development |
| 5.2 Entry checklist | `_generate_integration_checklist()` | Dynamic verification |

## Expected Improvements

After applying these fixes:

1. **Export names will match** - If plan says `PDFProcessor`, that's what you get
2. **Imports will work** - Generated from plan, not guessed
3. **Early failure detection** - Plan mismatch caught at file generation, not runtime
4. **No more cascading failures** - First file drift won't break everything
5. **Cleaner structure** - `__init__.py` files auto-generated

## Troubleshooting

### "Plan compliance failed: Missing export X"
The coder generated code without the required export. The system will automatically retry with explicit error message. If it keeps failing, the LLM may not understand the requirement - check the plan's export specification.

### "Import not found" at runtime
Check that:
1. `__init__.py` exists in all folders
2. The import path matches the file structure
3. The export actually exists (wasn't renamed by coder)

### Agents not found
Ensure agent files are in the correct paths:
```
src/coder/coder_agent.py
src/architect/architect_agent.py
etc.
```

## Architecture Notes

The key insight is that **local LLMs are not reliable at following complex instructions consistently**. By:

1. Generating exact import statements (not asking LLM to figure them out)
2. Formatting exports as mandatory contracts (not suggestions)
3. Rejecting non-compliant code early (not hoping it works out)

We work *with* the LLM's limitations rather than against them.
