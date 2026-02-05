#!/usr/bin/env python3
"""
Bugfix Workflow Module for SwarmCoordinator v2
==============================================

Handles debugging and fixing bugs in existing projects.
Used by workflow type 7 (BUGFIX) in interactive_v2.py.

Functions:
- analyze_bug: Analyzes the project and bug description
- create_bugfix_workflow: Creates tasks for fixing the bug
- get_bugfix_prompt_interactive: Interactive prompt for bug details
- setup_bugfix_project_directory: Sets up the working directory
"""

import os
import shutil
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Import from swarm_coordinator at module level (lazy import to avoid circular dependency)
# These will be imported when functions are called
AgentRole = None
Task = None
TaskStatus = None

def _ensure_imports():
    """Lazy import to avoid circular dependency with swarm_coordinator_v2"""
    global AgentRole, Task, TaskStatus
    if AgentRole is None:
        from swarm_coordinator_v2 import AgentRole as AR, Task as T, TaskStatus as TS
        AgentRole = AR
        Task = T
        TaskStatus = TS


class BugfixContext:
    """Context object for bugfix workflow - contains all info needed for fixing a bug."""
    def __init__(self, project_path: str, bug_description: str, expected_behavior: str = "", project_info: Dict = None):
        self.project_path = project_path
        self.bug_description = bug_description
        self.expected_behavior = expected_behavior or bug_description[:100]
        self.project_info = project_info or {}


def get_bugfix_prompt_interactive() -> Tuple[str, str]:
    """
    Interactive prompt to get project path and bug description from user.
    Called by interactive_v2.py for workflow 7.
    
    NOTE: This is a minimal version. Use get_bugfix_input() for full BugfixContext.
    
    Returns:
        (project_path, bug_description)
    """
    print("\n" + "=" * 60)
    print("BUGFIX WORKFLOW")
    print("=" * 60)
    
    # Get project path
    while True:
        project_path = input("\nEnter path to project with bug: ").strip()
        if not project_path:
            print("❌ Project path is required")
            continue
        if not os.path.exists(project_path):
            print(f"❌ Path does not exist: {project_path}")
            continue
        if not os.path.isdir(project_path):
            print(f"❌ Path is not a directory: {project_path}")
            continue
        break
    
    # Get bug description
    print("\nDescribe the bug (what's happening, expected behavior, error messages):")
    print("(Type your description. Enter 'END' on a new line when done)")
    print("-" * 60)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    
    bug_description = '\n'.join(lines)
    
    return project_path, bug_description


def analyze_project_for_bugs(project_path: str) -> Dict[str, Any]:
    """
    Scan and analyze a project for debugging.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Dict with project analysis
    """
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    project_info = {
        "path": os.path.abspath(project_path),
        "files": [],
        "structure": "",
    }
    
    # Scan files
    for root, dirs, files in os.walk(project_path):
        # Skip non-essential directories
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules', 
            '.pytest_cache', '.mypy_cache', 'dist', 'build'
        }]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, project_path)
            
            # Skip non-essential files
            if filename.startswith('.') or filename.endswith(('.pyc', '.pyo')):
                continue
            
            try:
                size = os.path.getsize(filepath)
                content = ""
                
                # Read text files under 100KB
                if size < 100 * 1024 and _is_text_file(filename):
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                    except Exception:
                        content = ""
                
                project_info["files"].append({
                    "path": rel_path,
                    "content": content,
                    "size": size
                })
                
            except Exception as e:
                print(f"Warning: Could not process {rel_path}: {e}")
    
    # Build structure
    project_info["structure"] = _build_structure(project_info["files"])
    
    return project_info


def _is_text_file(filename: str) -> bool:
    """Check if file is likely a text file."""
    text_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.md', '.txt', '.rst', '.sh', '.sql', '.xml'
    }
    _, ext = os.path.splitext(filename)
    return ext.lower() in text_extensions


def _build_structure(files: List[Dict]) -> str:
    """Build a tree-like structure string."""
    lines = []
    for f in sorted(files, key=lambda x: x["path"]):
        depth = f["path"].count(os.sep)
        indent = "  " * depth
        name = os.path.basename(f["path"])
        lines.append(f"{indent}├── {name}")
    return "\n".join(lines[:100])  # Limit - increased for larger projects


def setup_bugfix_project_directory(coordinator, source_path: str, bug_description: str) -> str:
    """
    Set up project directory for bugfix workflow.
    Creates a new versioned copy of the project.
    
    SMART FIXES: 
    1. Detects if copying from 'src' dir and aligns structure.
    2. Cleans project name to avoid recursion.
    3. Copies ALL files (docs, tests, README) not just src.
    """
    import re
    
    source_path = os.path.abspath(source_path)
    source_name = os.path.basename(source_path.rstrip('/'))
    
    # LOGIC 1: Handle if user pointed to 'src' folder (move up one level)
    if source_name == 'src':
        real_project_root = os.path.dirname(source_path)
        parent_name = os.path.basename(real_project_root)
        print(f"   ℹ Detected source 'src' folder - using parent root: {real_project_root}")
        source_path = real_project_root # Switch source to the actual root
        raw_name = parent_name
    else:
        raw_name = source_name
        
    # LOGIC 2: Clean the name
    clean_name = re.sub(r'^\d+_', '', raw_name)
    clean_name = re.sub(r'_v\d+$', '', clean_name)
    clean_name = clean_name.replace('_bugfix', '')
    
    project_name = f"{clean_name}_bugfix"
    
    # Create directory using coordinator's method
    project_dir = coordinator._setup_project_directory(project_name, f"Bugfix: {bug_description[:100]}")
    
    print(f"   → Copying full project from: {source_path}")
    print(f"   → To: {project_dir}")
    
    # LOGIC 3: Copy EVERYTHING (smart recursive copy)
    for item in os.listdir(source_path):
        src_item = os.path.join(source_path, item)
        dst_item = os.path.join(project_dir, item)
        
        # Skip system/generated junk, but KEEP docs/tests/readmes
        if item in {'__pycache__', '.git', '.venv', 'venv', '.pytest_cache', '.DS_Store', 'dist', 'build'}:
            continue
        
        # Don't overwrite the new project info we just made
        if item == "PROJECT_INFO.txt":
            continue

        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dst_item)
            
    # Safety: Ensure src/__init__.py exists
    init_file = os.path.join(project_dir, "src", "__init__.py")
    if not os.path.exists(init_file) and os.path.exists(os.path.join(project_dir, "src")):
        with open(init_file, 'w') as f:
            f.write("")

    print(f"   ✓ Copied project to: {project_dir}")
    
    return project_dir


def get_bugfix_prompt(project_info: Dict, bug_description: str) -> str:
    """
    Build the prompt for debugging.
    
    Args:
        project_info: Result from analyze_project_for_bugs()
        bug_description: User's bug description
        
    Returns:
        Formatted prompt string
    """
    lines = [
        "=" * 60,
        "BUG REPORT",
        "=" * 60,
        "",
        bug_description,
        "",
        "=" * 60,
        "PROJECT STRUCTURE",
        "=" * 60,
        "",
        project_info["structure"],
        "",
    ]
    
    # Include relevant file contents
    py_files = [f for f in project_info["files"] if f["path"].endswith('.py') and f["content"]]
    
    total_size = 0
    max_size = 50000
    
    lines.append("=" * 60)
    lines.append("SOURCE CODE")
    lines.append("=" * 60)
    
    for f in py_files:
        if total_size + len(f["content"]) > max_size:
            lines.append(f"\n[Remaining files truncated]")
            break
        
        lines.append(f"\n### FILE: {f['path']} ###")
        lines.append(f["content"])
        total_size += len(f["content"])
    
    lines.extend([
        "",
        "=" * 60,
        "TASK: Find and fix the bug described above.",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def create_bugfix_workflow(coordinator, bugfix_context_or_info, bug_description: str = None):
    """
    Create workflow tasks for fixing a bug.
    
    Args:
        coordinator: SwarmCoordinator instance
        bugfix_context_or_info: BugfixContext object OR project_info Dict (legacy)
        bug_description: Bug description (only used if bugfix_context_or_info is a dict)
    """
    _ensure_imports()
    
    # Handle both BugfixContext object and legacy dict + string args
    if isinstance(bugfix_context_or_info, BugfixContext):
        project_info = bugfix_context_or_info.project_info
        bug_description = bugfix_context_or_info.bug_description
        expected_behavior = bugfix_context_or_info.expected_behavior
        project_path = bugfix_context_or_info.project_path
    else:
        # Legacy: bugfix_context_or_info is project_info dict, bug_description is separate
        project_info = bugfix_context_or_info
        if bug_description is None:
            bug_description = coordinator.state.get("context", {}).get("bug_description", "")
        expected_behavior = bug_description
        project_path = project_info.get("path", "")
    
    # Setup project directory
    if project_path:
        setup_bugfix_project_directory(coordinator, project_path, bug_description)
    
    user_request = get_bugfix_prompt(project_info, bug_description)
    
    # Store in coordinator state
    coordinator.state["context"]["user_request"] = user_request
    coordinator.state["context"]["bug_description"] = bug_description
    coordinator.state["context"]["expected_behavior"] = expected_behavior
    coordinator.state["context"]["project_info"] = project_info
    
    # Task 1: Analyze and diagnose the bug (Debugger)
    coordinator.add_task(Task(
        task_id="T001_diagnose",
        task_type="bug_diagnosis",
        description="Analyze code and diagnose the root cause of the bug",
        assigned_role=AgentRole.DEBUGGER,
        status=TaskStatus.PENDING,
        priority=10,
        metadata={
            "user_request": user_request,
            "bug_description": bug_description,
        }
    ))
    
    # Task 2: Implement the fix (Coder)
    coordinator.add_task(Task(
        task_id="T002_fix",
        task_type="coding",
        description="Implement the bug fix",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=9,
        dependencies=["T001_diagnose"],
        metadata={
            "user_request": user_request,
            "mode": "bugfix"
        }
    ))
    
    # Task 3: Review the fix (Reviewer)
    coordinator.add_task(Task(
        task_id="T003_review",
        task_type="code_review",
        description="Review the bug fix for correctness",
        assigned_role=AgentRole.REVIEWER,
        status=TaskStatus.PENDING,
        priority=8,
        dependencies=["T002_fix"],
        metadata={"user_request": user_request}
    ))
    
    # Task 4: Test the fix (Tester)
    coordinator.add_task(Task(
        task_id="T004_test",
        task_type="test_generation",
        description="Create tests to verify the fix and prevent regression",
        assigned_role=AgentRole.TESTER,
        status=TaskStatus.PENDING,
        priority=7,
        dependencies=["T002_fix"],
        metadata={"user_request": user_request}
    ))
    
    # Task 5: Verify (Verifier)
    coordinator.add_task(Task(
        task_id="T005_verify",
        task_type="verification",
        description="Verify the fix works correctly",
        assigned_role=AgentRole.VERIFIER,
        status=TaskStatus.PENDING,
        priority=6,
        dependencies=["T003_review", "T004_test"],
        metadata={"user_request": user_request}
    ))
    
    print(f"   ✓ Created bugfix workflow with 5 tasks")


def run_bugfix_workflow(coordinator, source_path: str, bug_description: str) -> str:
    """
    Full bugfix workflow setup.
    
    Args:
        coordinator: SwarmCoordinator instance
        source_path: Path to project with bug
        bug_description: Description of the bug
        
    Returns:
        Path to the new project directory
    """
    # 1. Analyze project
    print(f"\n🔍 Analyzing project: {source_path}")
    project_info = analyze_project_for_bugs(source_path)
    print(f"   Found {len(project_info['files'])} files")
    
    # 2. Set up project directory
    print(f"\n📁 Setting up project directory...")
    project_dir = setup_bugfix_project_directory(coordinator, source_path, bug_description)
    
    # 3. Create workflow tasks (pass all 3 args)
    print(f"\n📋 Creating bugfix workflow...")
    create_bugfix_workflow(coordinator, project_info, bug_description)
    
    return project_dir


# Export for interactive_v2.py compatibility
def handle_bugfix_workflow(coordinator) -> str:
    """
    Handle the bugfix workflow from interactive_v2.py.
    Uses get_bugfix_input() to gather all info, then sets up workflow.
    
    Returns:
        The bug description as user_request
    """
    # Use the consolidated input function
    context = get_bugfix_input()
    
    if not context.bug_description.strip():
        print("\n❌ No bug description provided.")
        return ""
    
    # Set up project directory
    setup_bugfix_project_directory(coordinator, context.project_path, context.bug_description)
    
    # Create workflow with BugfixContext
    create_bugfix_workflow(coordinator, context)
    
    # Store path info
    coordinator.state["project_info"]["source_project"] = context.project_path
    
    return context.bug_description


def get_bugfix_input() -> BugfixContext:
    """
    Interactive prompt to gather bugfix info from user.
    Consolidates get_bugfix_prompt_interactive() and previous get_bugfix_input().
    
    Returns:
        BugfixContext with all info needed for bugfix workflow
    """
    print("\n" + "=" * 60)
    print("BUGFIX WORKFLOW")
    print("=" * 60)
    
    # Get project path
    while True:
        project_path = input("\nEnter path to project with bug: ").strip()
        if not project_path:
            print("❌ Project path is required")
            continue
        if not os.path.exists(project_path):
            print(f"❌ Path does not exist: {project_path}")
            continue
        if not os.path.isdir(project_path):
            print(f"❌ Path is not a directory: {project_path}")
            continue
        break
    
    # Get bug description
    print("\nDescribe the bug (what's happening wrong):")
    print("(Type your description. Enter 'END' on a new line when done)")
    print("-" * 60)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    
    bug_description = '\n'.join(lines)
    
    # Get expected behavior
    print("\nWhat should happen instead (expected behavior)?")
    print("(Press Enter to skip, or type answer then 'END')")
    print("-" * 60)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END' or (not lines and not line.strip()):
                break
            lines.append(line)
        except EOFError:
            break
    
    expected_behavior = '\n'.join(lines) if lines else bug_description[:100]
    
    # Analyze project
    print(f"\n🔍 Analyzing project...")
    project_info = analyze_project_for_bugs(project_path)
    print(f"   Found {len(project_info['files'])} files")
    
    return BugfixContext(project_path, bug_description, expected_behavior, project_info)