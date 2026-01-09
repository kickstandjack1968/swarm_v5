#!/usr/bin/env python3
"""
SwarmCoordinator V2 Patches
===========================

This file contains the fixes that need to be integrated into swarm_coordinator_v2.py.
Apply these changes to fix the issues identified in the diagnosis.

USAGE:
    1. Review each section below
    2. Apply the changes to swarm_coordinator_v2.py
    3. Update the import to use plan_executor_v2 instead of plan_executor
"""

# =============================================================================
# PATCH 1: Update imports to use plan_executor_v2
# =============================================================================
# 
# FIND (around line 39-51):
#     try:
#         from plan_executor import (
#             PlanExecutor, 
#             create_planned_workflow,
#             execute_plan_task,
#             ...
#         )
#     
# REPLACE WITH:
#     try:
#         from plan_executor_v2 import (
#             PlanExecutor, 
#             create_planned_workflow,
#             execute_plan_task,
#             ARCHITECT_PLAN_SYSTEM_PROMPT,
#             get_architect_plan_prompt,
#             extract_yaml_from_response
#         )
#         PLAN_EXECUTOR_AVAILABLE = True
#     except ImportError as e:
#         print(f"Import failed: {e}")
#         PLAN_EXECUTOR_AVAILABLE = False


# =============================================================================
# PATCH 2: Store parsed plan in context (_update_context)
# =============================================================================
#
# FIND the _update_context method (around line 3489):
#
# REPLACE WITH:

def _update_context_v2(self, task):
    """Update shared context with task results - ENHANCED VERSION."""
    context_key = f"{task.task_type}_{task.assigned_role.value}"
    self.state["context"][context_key] = {
        "task_id": task.task_id,
        "result": task.result,
        "completed_at": task.completed_at
    }
    
    # Store latest code separately
    if task.task_type in ("coding", "revision", "plan_execution") and task.result:
        self.state["context"]["latest_code"] = task.result
    
    # NEW: Store parsed plan for downstream access
    if task.task_type == "architecture_plan" and task.result:
        try:
            import yaml
            parsed = yaml.safe_load(task.result)
            self.state["context"]["parsed_plan"] = parsed
            
            # Extract key info for quick access
            self.state["context"]["plan_files"] = [
                f.get('name') for f in parsed.get('files', [])
            ]
            self.state["context"]["plan_exports"] = {
                f.get('name'): [e.get('name') for e in f.get('exports', [])]
                for f in parsed.get('files', [])
            }
        except:
            pass


# =============================================================================
# PATCH 3: Enhanced _verify_coder_files (checks exports, not just file names)
# =============================================================================
#
# FIND _verify_coder_files method (around line 1233):
#
# REPLACE WITH:

def _verify_coder_files_v2(self, architect_result: str, coder_result: str) -> tuple:
    """
    Verify that coder created all files with correct exports.
    Returns (passed, list_of_issues)
    """
    import yaml
    
    issues = []
    
    # Parse architect's plan
    try:
        plan = yaml.safe_load(architect_result)
        if not plan or 'files' not in plan:
            return True, []  # Can't verify without plan
    except:
        return True, []
    
    # Parse coder's output
    actual_files = self._parse_multi_file_output(coder_result)
    
    if not actual_files:
        # Single file output - can't verify multi-file plan
        if len(plan.get('files', [])) > 1:
            return False, ["Expected multiple files but got single output"]
        return True, []
    
    # Check each planned file
    for file_spec in plan.get('files', []):
        filename = file_spec.get('name', '')
        planned_exports = [e.get('name') for e in file_spec.get('exports', [])]
        
        # Check file exists
        if filename not in actual_files:
            issues.append(f"Missing file: {filename}")
            continue
        
        # Check exports exist
        if planned_exports:
            content = actual_files[filename]
            actual_exports = set(self._extract_exports_from_code(content))
            
            for export in planned_exports:
                if export not in actual_exports:
                    issues.append(f"{filename}: Missing export '{export}'")
    
    return len(issues) == 0, issues


# =============================================================================
# PATCH 4: Better coder task handling with plan context
# =============================================================================
#
# In execute_task, for CODER tasks (around line 1802):
#
# AFTER the payload is built, ADD:

def _enhance_coder_payload(self, payload, task):
    """Enhance coder payload with plan context."""
    
    # Get plan exports for context
    plan_exports = self.state["context"].get("plan_exports", {})
    if plan_exports:
        # Add expected exports to user message
        user_msg = payload.get("user_message", "")
        
        exports_section = "\n\nPLAN-MANDATED EXPORTS:\n"
        for filename, exports in plan_exports.items():
            if exports:
                exports_section += f"  {filename}: {', '.join(exports)}\n"
        
        payload["user_message"] = user_msg + exports_section
    
    return payload


# =============================================================================
# PATCH 5: Auto-generate __init__.py files in _save_project_outputs
# =============================================================================
#
# At the END of _save_project_outputs method, ADD:

def _generate_init_files(self, project_dir: str):
    """Auto-generate __init__.py for all folders with Python files."""
    import os
    
    src_dir = os.path.join(project_dir, "src")
    if not os.path.exists(src_dir):
        return
    
    # Find all folders containing .py files
    folders_with_py = set()
    for root, dirs, files in os.walk(src_dir):
        if any(f.endswith('.py') for f in files):
            folders_with_py.add(root)
            # Also add parent folders
            parent = root
            while parent != src_dir:
                parent = os.path.dirname(parent)
                if parent:
                    folders_with_py.add(parent)
    
    # Generate __init__.py for each
    for folder in folders_with_py:
        init_path = os.path.join(folder, "__init__.py")
        if not os.path.exists(init_path):
            # Simple __init__.py with exports
            py_files = [f for f in os.listdir(folder) 
                       if f.endswith('.py') and f != '__init__.py']
            
            lines = ['"""Auto-generated __init__.py"""', '']
            
            # Import from each module
            for py_file in py_files:
                module = py_file[:-3]  # Remove .py
                lines.append(f"from .{module} import *")
            
            with open(init_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"   ✓ Generated: {os.path.relpath(init_path, project_dir)}")


# =============================================================================
# PATCH 6: Structured job_scope in _handle_clarification_interactive
# =============================================================================
#
# After synthesizing job_scope (around line 2400-2418), ADD structure:

def _structure_job_scope(self, user_request: str, questions: str, answers: str, synthesized: str) -> dict:
    """Create structured job scope with both prose and checkpoints."""
    return {
        "prose": synthesized,
        "original_request": user_request,
        "clarification_qa": {
            "questions": questions,
            "answers": answers
        },
        "checkpoints": _extract_requirements_as_checkpoints(synthesized)
    }


def _extract_requirements_as_checkpoints(text: str) -> list:
    """Extract bullet points and numbered items as checkpoints."""
    checkpoints = []
    
    # Find numbered items (1. something)
    import re
    numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+)$', text, re.MULTILINE)
    checkpoints.extend(numbered)
    
    # Find bullet points
    bullets = re.findall(r'^\s*[-*•]\s*(.+)$', text, re.MULTILINE)
    checkpoints.extend(bullets)
    
    return checkpoints


# =============================================================================
# PATCH 7: Update standard workflow to use plan executor
# =============================================================================
#
# Option A: Modify _create_standard_workflow to use planned workflow
# Option B: Keep separate but ensure plan is generated and stored
#
# For Option B, in _create_standard_workflow, after architecture task, ADD:

def _standard_workflow_store_plan(self, user_request: str):
    """Store plan from architect for coder access."""
    # After architect task completes, parse and store the plan
    # This ensures standard workflow also benefits from plan enforcement
    
    # Add a plan parsing step
    architect_task = next(
        (t for t in self.completed_tasks if t.assigned_role.value == "architect"),
        None
    )
    if architect_task and architect_task.result:
        try:
            import yaml
            # Try to extract YAML from architect output
            result = architect_task.result
            if '```yaml' in result or '```yml' in result:
                import re
                match = re.search(r'```ya?ml\s*(.*?)```', result, re.DOTALL)
                if match:
                    result = match.group(1)
            
            plan = yaml.safe_load(result)
            if plan and 'files' in plan:
                self.state["context"]["parsed_plan"] = plan
                self.state["context"]["plan_yaml"] = result
        except:
            pass


# =============================================================================
# PATCH 8: Integration - How to apply all patches
# =============================================================================
"""
INTEGRATION STEPS:

1. Copy agent_base.py to your project root or src/ directory

2. Copy plan_executor_v2.py to your project root (same location as plan_executor.py)

3. Update swarm_coordinator_v2.py imports (Patch 1)

4. Replace _update_context with _update_context_v2 (Patch 2)

5. Replace _verify_coder_files with _verify_coder_files_v2 (Patch 3)

6. In execute_task CODER block, call _enhance_coder_payload (Patch 4)

7. At end of _save_project_outputs, call _generate_init_files (Patch 5)

8. Update job_scope handling to use structured format (Patch 6)

9. Either:
   - Change workflow_type="standard" to use "planned" internally
   - OR add _standard_workflow_store_plan call (Patch 7)

10. Update individual agent files (coder_agent.py, architect_agent.py, etc.)
    to use agent_base.py if desired (optional but recommended)

TESTING:

After applying patches, test with:

    coordinator = SwarmCoordinator()
    coordinator.run_workflow(
        "Create a simple calculator CLI",
        workflow_type="planned"
    )

Expected improvements:
- Exports will match what architect specified
- Imports will be correct (generated from plan, not guessed)
- __init__.py files auto-generated
- Plan compliance verification catches mismatches early
"""


# =============================================================================
# QUICK REFERENCE: Key changes summary
# =============================================================================
"""
FILE: plan_executor_v2.py (NEW)
- _generate_import_block(): Generates imports FROM THE PLAN, not actuals
- _generate_mandatory_exports(): Formats exports as contract, not suggestions
- _validate_exports_against_plan(): Rejects files that don't match plan
- _verify_plan_compliance(): Comprehensive gate before accepting any file
- _generate_init_files(): Auto-creates __init__.py for folders
- Enhanced prompts reference plan-driven development explicitly

FILE: agent_base.py (NEW)
- Shared LLM calling code
- Common error handling
- Export/signature extraction utilities

FILE: swarm_coordinator_v2.py (PATCHES)
- _update_context(): Now stores parsed plan in context
- _verify_coder_files(): Checks exports, not just file existence
- _save_project_outputs(): Calls _generate_init_files()
- Uses plan_executor_v2 instead of plan_executor
"""

if __name__ == "__main__":
    print("This file contains patches - read the docstrings for integration instructions")
