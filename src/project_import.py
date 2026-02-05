#!/usr/bin/env python3
"""
Project Import Module for SwarmCoordinator v2
=============================================

Handles importing existing projects for modification.
Used by workflow type 6 (IMPORT) in interactive_v2.py.

FIXES INCLUDED:
1. Smart Directory Copy: Detects existing 'src/' vs flat projects to prevent nesting.
2. Full Context: Includes config files (requirements.txt, etc) in prompts.
3. Anti-Refactor Prompt: Strictly instructs agent to ONLY apply fixes.
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


def import_existing_project(project_path: str) -> Dict[str, Any]:
    """
    Scan and analyze an existing project.
    
    Args:
        project_path: Path to the existing project directory
        
    Returns:
        Dict with project analysis:
        {
            "path": str,
            "files": [{"path": str, "content": str, "size": int}, ...],
            "structure": str,  # Tree-like structure
            "summary": str,    # Brief description
            "entry_points": [str, ...],
            "dependencies": [str, ...],
        }
    """
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    if not os.path.isdir(project_path):
        raise ValueError(f"Project path is not a directory: {project_path}")
    
    project_info = {
        "path": os.path.abspath(project_path),
        "files": [],
        "structure": "",
        "summary": "",
        "entry_points": [],
        "dependencies": [],
    }
    
    # Scan files
    all_files = []
    for root, dirs, files in os.walk(project_path):
        # Skip common non-essential directories
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', '.venv', 'venv', 'node_modules', 
            '.pytest_cache', '.mypy_cache', 'dist', 'build', 'egg-info',
            '.eggs', '.tox', '.idea', '.vscode'
        }]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, project_path)
            
            # Skip non-essential files
            if filename.startswith('.') and filename not in {'.env', '.gitignore', '.dockerignore', '.env.example'}:
                continue
            if filename.endswith(('.pyc', '.pyo', '.so', '.o', '.a', '.db', '.sqlite')):
                continue
            
            try:
                size = os.path.getsize(filepath)
                
                # Only read text files under 100KB
                content = ""
                if size < 100 * 1024 and _is_text_file(filename):
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                    except Exception:
                        content = f"[Could not read file: {rel_path}]"
                
                all_files.append({
                    "path": rel_path,
                    "content": content,
                    "size": size
                })
                
            except Exception as e:
                print(f"Warning: Could not process {rel_path}: {e}")
    
    project_info["files"] = all_files
    
    # Build structure tree
    project_info["structure"] = _build_tree_structure(project_path, all_files)
    
    # Detect entry points
    project_info["entry_points"] = _detect_entry_points(all_files)
    
    # Extract dependencies
    project_info["dependencies"] = _extract_dependencies(all_files)
    
    # Generate summary
    project_info["summary"] = _generate_summary(project_info)
    
    return project_info


def _is_text_file(filename: str) -> bool:
    """Check if file is likely a text file based on extension."""
    text_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
        '.md', '.txt', '.rst', '.sh', '.bash', '.zsh',
        '.sql', '.xml', '.env', '.gitignore', '.dockerignore',
        'Dockerfile', 'Makefile', 'requirements.txt', 'setup.py',
        'pyproject.toml', 'package.json', 'tsconfig.json'
    }
    
    _, ext = os.path.splitext(filename)
    return ext.lower() in text_extensions or filename in text_extensions


def _build_tree_structure(base_path: str, files: List[Dict]) -> str:
    """Build a tree-like string representation of the project structure."""
    lines = [os.path.basename(base_path) + "/"]
    
    # Group files by directory
    dirs = set()
    for f in files:
        parts = f["path"].split(os.sep)
        for i in range(len(parts) - 1):
            dirs.add(os.sep.join(parts[:i+1]))
    
    # Sort entries
    entries = sorted(set(list(dirs) + [f["path"] for f in files]))
    
    for entry in entries:
        depth = entry.count(os.sep)
        indent = "  " * depth
        name = os.path.basename(entry)
        
        # Check if it's a directory
        is_dir = entry in dirs
        if is_dir:
            lines.append(f"{indent}├── {name}/")
        else:
            # Find file info
            file_info = next((f for f in files if f["path"] == entry), None)
            size_str = ""
            if file_info:
                size = file_info["size"]
                if size < 1024:
                    size_str = f" ({size}B)"
                elif size < 1024 * 1024:
                    size_str = f" ({size // 1024}KB)"
                else:
                    size_str = f" ({size // (1024*1024)}MB)"
            lines.append(f"{indent}├── {name}{size_str}")
    
    return "\n".join(lines[:200])  # Increased limit from 50 to 200 lines


def _detect_entry_points(files: List[Dict]) -> List[str]:
    """Detect likely entry point files."""
    entry_points = []
    
    entry_patterns = ['main.py', 'app.py', 'run.py', 'server.py', 'cli.py', '__main__.py']
    
    for f in files:
        filename = os.path.basename(f["path"])
        
        # Check common entry point names
        if filename in entry_patterns:
            entry_points.append(f["path"])
            continue
        
        # Check for if __name__ == "__main__" pattern
        if filename.endswith('.py') and f["content"]:
            if '__name__' in f["content"] and '__main__' in f["content"]:
                entry_points.append(f["path"])
    
    return entry_points


def _extract_dependencies(files: List[Dict]) -> List[str]:
    """Extract dependencies from requirements.txt, setup.py, pyproject.toml."""
    dependencies = []
    
    for f in files:
        filename = os.path.basename(f["path"])
        content = f["content"]
        
        if not content:
            continue
        
        if filename == "requirements.txt":
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Strip version specifiers for summary
                    dep = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                    if dep:
                        dependencies.append(dep)
        
        elif filename == "setup.py":
            # Basic extraction - look for install_requires
            import re
            match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                deps_str = match.group(1)
                for dep in re.findall(r'["\']([^"\']+)["\']', deps_str):
                    dep = dep.split('==')[0].split('>=')[0].split('<=')[0]
                    dependencies.append(dep)
        
        elif filename == "pyproject.toml":
            # Basic extraction
            import re
            match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                deps_str = match.group(1)
                for dep in re.findall(r'["\']([^"\']+)["\']', deps_str):
                    dep = dep.split('==')[0].split('>=')[0].split('<=')[0]
                    dependencies.append(dep)
    
    return list(set(dependencies))


def _generate_summary(project_info: Dict) -> str:
    """Generate a brief summary of the project."""
    file_count = len(project_info["files"])
    py_files = [f for f in project_info["files"] if f["path"].endswith('.py')]
    
    lines = [
        f"Project: {os.path.basename(project_info['path'])}",
        f"Files: {file_count} total, {len(py_files)} Python files",
    ]
    
    if project_info["entry_points"]:
        lines.append(f"Entry points: {', '.join(project_info['entry_points'][:3])}")
    
    if project_info["dependencies"]:
        deps = project_info["dependencies"][:5]
        lines.append(f"Dependencies: {', '.join(deps)}" + (" ..." if len(project_info["dependencies"]) > 5 else ""))
    
    return "\n".join(lines)


def setup_import_project_directory(coordinator, source_path: str, modification_request: str) -> str:
    """
    Set up the project directory for an import workflow.
    Smartly handles structured projects (with src/) vs flat scripts.
    
    Args:
        coordinator: SwarmCoordinator instance
        source_path: Path to the source project
        modification_request: What changes are being made
        
    Returns:
        Path to the new project directory
    """
    # Get project name from source path
    source_name = os.path.basename(source_path.rstrip('/'))
    
    # Create project directory using coordinator's method
    project_dir = coordinator._setup_project_directory(source_name, modification_request)
    
    source_abs = os.path.abspath(source_path)
    
    # SMART DETECTION: Does the source already have a 'src' directory?
    has_src_structure = os.path.isdir(os.path.join(source_abs, "src"))
    
    print(f"   ℹ Detected structure: {'Standard (src/ found)' if has_src_structure else 'Flat (migrating to src/)'}")

    for item in os.listdir(source_abs):
        src_item = os.path.join(source_abs, item)
        
        # Skip pycache and other non-essential dirs
        if item in {'__pycache__', '.git', '.venv', 'venv', 'node_modules', '.pytest_cache', 'dist', 'build', '.idea', '.vscode'}:
            continue
            
        # Determine destination
        if has_src_structure:
            # Preserves structure: Copy root items to root items
            # src -> src, tests -> tests, README.md -> README.md
            dst_item = os.path.join(project_dir, item)
        else:
            # Flat structure migration:
            # Move code to src/, but keep config/meta files in root
            if item in ["requirements.txt", "README.md", ".env", ".gitignore", "setup.py", "pyproject.toml", "Dockerfile"]:
                dst_item = os.path.join(project_dir, item)
            else:
                # Assume everything else is source code if it's a flat dir
                # Ensure src exists
                os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
                dst_item = os.path.join(project_dir, "src", item)
        
        # Perform Copy
        if os.path.isdir(src_item):
            # shutil.copytree requires destination to NOT exist, but we might be merging into existing src/
            # So we use dirs_exist_ok=True
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dst_item)
    
    print(f"   ✓ Copied project to: {project_dir}")
    
    return project_dir


def get_import_prompt(project_info: Dict, modification_request: str) -> str:
    """
    Build the prompt for modifying an imported project.
    Now includes ALL text files and STRICT constraints against refactoring.
    
    Args:
        project_info: Result from import_existing_project()
        modification_request: What the user wants to change
        
    Returns:
        Formatted prompt string
    """
    lines = [
        "=" * 60,
        "EXISTING PROJECT TO MODIFY",
        "=" * 60,
        "",
        project_info["summary"],
        "",
        "PROJECT STRUCTURE:",
        project_info["structure"],
        "",
    ]
    
    # FIX: Include ALL text files (configs, requirements, env), not just .py
    files_to_include = [f for f in project_info["files"] if f["content"]]
    
    total_content_size = 0
    # INCREASED LIMIT: 50k -> 150k characters (approx 40k tokens)
    max_content_size = 150000 
    
    lines.append("=" * 60)
    lines.append("EXISTING CODE & CONFIG")
    lines.append("=" * 60)
    
    for f in files_to_include:
        if total_content_size + len(f["content"]) > max_content_size:
            lines.append(f"\n[Remaining {len(files_to_include) - files_to_include.index(f)} files truncated for length]")
            break
        
        lines.append(f"\n### FILE: {f['path']} ###")
        lines.append(f["content"])
        total_content_size += len(f["content"])
    
    # FIX: Add STRICT constraints to prevent unnecessary rewrites
    lines.extend([
        "",
        "=" * 60,
        "MODIFICATION REQUEST",
        "=" * 60,
        "",
        modification_request,
        "",
        "=" * 60,
        "CRITICAL INSTRUCTIONS - READ CAREFULLY",
        "=" * 60,
        "1. TARGETED FIX ONLY: Do NOT refactor, reorganize, or improve code that is not broken.",
        "2. PRESERVE CONTEXT: Keep existing imports, comments, and structure exactly as they are.",
        "3. NO HALLUCINATIONS: Do not invent new dependencies or delete existing config files.",
        "4. FULL OUTPUT REQUIRED: You must output the COMPLETE content of the modified files so they can be saved.",
        "   (Copy the unchanged parts exactly, then apply your fix).",
        "5. CONFIG FILES: If updating a config (like requirements.txt), keep the original entries unless explicitly asked to remove them."
    ])
    
    return "\n".join(lines)


def create_import_workflow(coordinator, project_info: Dict, modification_request: str):
    """
    Create workflow tasks for modifying an imported project.
    
    Args:
        coordinator: SwarmCoordinator instance
        project_info: Result from import_existing_project()
        modification_request: What the user wants to change
    """
    from swarm_coordinator_v2 import AgentRole, Task, TaskStatus
    
    user_request = get_import_prompt(project_info, modification_request)
    
    # Store in coordinator state
    coordinator.state["context"]["user_request"] = user_request
    coordinator.state["context"]["original_modification_request"] = modification_request
    coordinator.state["context"]["imported_project"] = project_info
    
    # Task 1: Analyze what needs to change (Architect)
    coordinator.add_task(Task(
        task_id="T001_analyze",
        task_type="code_analysis",
        description="Analyze existing code and plan modifications",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=10,
        metadata={
            "user_request": user_request,
            "modification_request": modification_request,
            "mode": "modify_existing"
        }
    ))
    
    # Task 2: Implement changes (Coder)
    coordinator.add_task(Task(
        task_id="T002_modify",
        task_type="coding",
        description="Implement the requested modifications",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=9,
        dependencies=["T001_analyze"],
        metadata={
            "user_request": user_request,
            "mode": "modify_existing"
        }
    ))
    
    # Task 3: Review changes (Reviewer)
    coordinator.add_task(Task(
        task_id="T003_review",
        task_type="code_review",
        description="Review the modifications",
        assigned_role=AgentRole.REVIEWER,
        status=TaskStatus.PENDING,
        priority=8,
        dependencies=["T002_modify"],
        metadata={"user_request": user_request}
    ))
    
    # Task 4: Update tests (Tester)
    coordinator.add_task(Task(
        task_id="T004_tests",
        task_type="test_generation",
        description="Update or create tests for modifications",
        assigned_role=AgentRole.TESTER,
        status=TaskStatus.PENDING,
        priority=7,
        dependencies=["T002_modify"],
        metadata={"user_request": user_request}
    ))
    
    # Task 5: Update documentation (Documenter)
    coordinator.add_task(Task(
        task_id="T005_document",
        task_type="documentation",
        description="Update documentation to reflect changes",
        assigned_role=AgentRole.DOCUMENTER,
        status=TaskStatus.PENDING,
        priority=6,
        dependencies=["T002_modify", "T004_tests"],
        metadata={"user_request": user_request}
    ))
    
    # Task 6: Verify (Verifier)
    coordinator.add_task(Task(
        task_id="T006_verify",
        task_type="verification",
        description="Verify modifications work correctly",
        assigned_role=AgentRole.VERIFIER,
        status=TaskStatus.PENDING,
        priority=5,
        dependencies=["T005_document"],
        metadata={"user_request": user_request}
    ))
    
    print(f"   ✓ Created import workflow with 6 tasks")


# Convenience function for interactive_v2.py
def run_import_workflow(coordinator, source_path: str, modification_request: str) -> str:
    """
    Full import workflow setup - call this from interactive_v2.py.
    
    Args:
        coordinator: SwarmCoordinator instance
        source_path: Path to project to import
        modification_request: What to modify
        
    Returns:
        Path to the new project directory
    """
    # 1. Analyze existing project
    print(f"\n📂 Analyzing project: {source_path}")
    project_info = import_existing_project(source_path)
    print(f"   Found {len(project_info['files'])} files")
    
    # 2. Set up new project directory
    print(f"\n📁 Setting up project directory...")
    project_dir = setup_import_project_directory(coordinator, source_path, modification_request)
    
    # 3. Create workflow tasks
    print(f"\n📋 Creating workflow tasks...")
    create_import_workflow(coordinator, project_info, modification_request)
    
    return project_dir


def get_import_prompt_interactive() -> Tuple[str, str]:
    """
    Interactive prompt to get project path and modification request from user.
    Called by interactive_v2.py for workflow 6.
    
    Returns:
        (project_path, modification_request)
    """
    print("\n" + "=" * 60)
    print("IMPORT EXISTING PROJECT")
    print("=" * 60)
    
    # Get project path
    while True:
        project_path = input("\nEnter path to existing project: ").strip()
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
    
    # Get modification request
    print("\nWhat modifications do you want to make?")
    print("(Type your request. Enter 'END' on a new line when done)")
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
    
    modification_request = '\n'.join(lines)
    
    return project_path, modification_request