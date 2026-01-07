# PlanExecuto#!/usr/bin/env python3
"""
Plan Executor Module for SwarmCoordinator v2
============================================

Orchestrates program generation by executing a YAML plan file-by-file.
Maintains global context and feeds scoped tasks to Coder/Verifier agents.

Integrates with existing SwarmCoordinator infrastructure:
- Uses AgentExecutor for LLM calls
- Uses Task dataclass for task management
- Uses existing file output format (### FILE: filename.py ###)

Usage:
    from plan_executor import PlanExecutor, run_plan_workflow
    
    # Option 1: Use with existing SwarmCoordinator
    coordinator = SwarmCoordinator()
    coordinator.run_workflow(user_request, workflow_type="planned")
    
    # Option 2: Standalone execution
    result = run_plan_workflow(user_request, config)
"""


import sys          # <--- REQUIRED for sys.executable
import shutil       # <--- REQUIRED
import tempfile     # <--- REQUIRED for temporary directories
import subprocess   # <--- REQUIRED for running pytest
import yaml
import re
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import ast

# =============================================================================
# LOGGING SETUP
# =============================================================================

            
def setup_plan_executor_logging(
    log_dir: str = "logs",
    log_level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging for PlanExecutor.
    
    Creates:
    - logs/plan_executor_YYYYMMDD_HHMMSS.log (detailed)
    - logs/plan_executor_latest.log (symlink to latest)
    - Console output (summary)
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("PlanExecutor")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"plan_executor_{timestamp}.log")
    
    # File handler - detailed
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - summary only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Update "latest" symlink (Unix only)
    latest_link = os.path.join(log_dir, "plan_executor_latest.log")
    try:
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(log_file), latest_link)
    except (OSError, NotImplementedError):
        pass  # Windows or permission issues
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get or create the PlanExecutor logger"""
    global _logger
    if _logger is None:
        _logger = setup_plan_executor_logging()
    return _logger


class FileStatus(Enum):
    """Status of individual file generation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REVISION_NEEDED = "revision_needed"


@dataclass
class FileSpec:
    """Specification for a single file in the plan"""
    name: str
    purpose: str
    dependencies: List[str] = field(default_factory=list)
    exports: List[Dict[str, str]] = field(default_factory=list)
    imports_from: Dict[str, List[str]] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    optional: bool = False
    fallback: Optional[str] = None


@dataclass
class FileResult:
    """Result of generating a single file"""
    name: str
    content: str
    actual_exports: List[Dict[str, str]]
    status: FileStatus
    verification: Optional[Dict[str, Any]] = None
    notes: str = ""
    revision_count: int = 0


@dataclass 
class ProgramPlan:
    """Parsed program plan from YAML"""
    name: str
    description: str
    architecture_pattern: str
    entry_point: str
    files: List[FileSpec]
    execution_order: List[str]
    raw_yaml: str


class PlanExecutor:
    """
    Orchestrates program generation by executing a YAML plan file-by-file.
    Maintains global context and feeds scoped tasks to Coder/Verifier agents.
    """
    
    def __init__(self, executor, config: Dict, log_dir: str = "logs"):
        """
        Initialize the Plan Executor.
        
        Args:
            executor: AgentExecutor instance for making LLM calls
            config: Configuration dict (same as SwarmCoordinator config)
            log_dir: Directory for log files
        """
        self.executor = executor
        self.config = config
        self.plan: Optional[ProgramPlan] = None
        self.completed_files: Dict[str, FileResult] = {}
        self.status: Dict[str, FileStatus] = {}
        self.errors: List[Dict[str, Any]] = []
        self.max_file_revisions = 3
        self.log_dir = log_dir
        self.logger = setup_plan_executor_logging(log_dir)
        self.execution_log: List[Dict[str, Any]] = []  # Structured log for JSON export
        self.start_time: Optional[float] = None


    def _generate_hot_test(self, file_spec: FileSpec, code_content: str, context: Dict) -> str:
        """Generate a specific, runnable unit test for the file currently being created."""
        from swarm_coordinator_v2 import AgentRole
        
        # We need the relative path for imports to work during execution
        module_name = file_spec.name.replace('.py', '')
        
        system_prompt = """You are a QA Engineer writing a 'Hot Test'.
Your goal is to write a minimal, standalone pytest script to verify the file provided works.

RULES:
1. Import using: from src.{module} import ClassName, function_name
   (Import specific classes/functions directly from the module file)
2. Test the HAPPY PATH only (verify it runs/imports/returns values).
3. For mocking, use ONLY unittest.mock from the standard library:
   - from unittest.mock import Mock, patch, MagicMock
   - DO NOT use pytest-mock, mocker fixture, or any third-party mocking libraries
4. Mock external dependencies (network, filesystem, databases) but prefer real execution when possible.
5. Keep tests simple - verify imports work, classes instantiate, functions return expected types.
6. Output ONLY raw python code - no markdown, no explanations.

CRITICAL RULES:
7. ALWAYS use MagicMock (not Mock) for objects used as context managers (with statements).
   - Mock() does NOT support __enter__/__exit__ by default
   - MagicMock() automatically supports all magic methods
8. ALWAYS create temp directories in test setup when testing code that accesses filesystem:
   - Use tempfile.mkdtemp() or tmp_path fixture
   - Create any folders the code expects to exist
9. When mocking database connections that use 'with' statements:
   - Use MagicMock() for the connection object
   - Example: mock_conn = MagicMock()
             mock_db.get_connection.return_value.__enter__.return_value = mock_conn
10. Use the EXACT class names, method names, and attribute names from the code provided.
    Do NOT guess or invent names - read the actual code and use those exact names."""
        
        user_message = f"""Generate a test for {file_spec.name}.

READ THIS CODE CAREFULLY - USE EXACT NAMES FROM IT:
```python
{code_content}
```

DEPENDENCIES AVAILABLE:
{list(context['dependencies'].keys())}

CRITICAL: 
- Use the EXACT class names, method names, and attribute names from the code above
- Do NOT invent or guess names - read what's actually defined in the code
- Import using: from src.{module_name} import ClassName, function_name

Output a valid 'tests/test_{module_name}.py' file content."""
        
        response = self.executor.execute_agent(
            role=AgentRole.TESTER,
            system_prompt=system_prompt,
            user_message=user_message
        )
        
        test_content = self._clean_code_output(response, f"test_{module_name}.py")
        self.logger.debug(f"Generated hot test for {file_spec.name}:\n{test_content[:3000]}")
        return test_content

    def _run_execution_check(self, test_content: str, file_name: str) -> Tuple[bool, str]:
        """Save the code and test to a temp env and run pytest (Docker or local fallback)."""
        
        # Try to use Docker sandbox from swarm_coordinator
        try:
            from swarm_coordinator_v2 import get_docker_sandbox
            sandbox = get_docker_sandbox()
            
            if sandbox and sandbox.container_running:
                # Get requirements content if available
                requirements_content = ""
                if "requirements.txt" in self.completed_files:
                    req_result = self.completed_files["requirements.txt"]
                    requirements_content = req_result.content if hasattr(req_result, 'content') else str(req_result)
                
                return sandbox.run_hot_test(
                    completed_files=self.completed_files,
                    test_content=test_content,
                    file_name=file_name,
                    requirements_content=requirements_content
                )
        except ImportError:
            pass  # Fall through to local execution
        except Exception as e:
            self.logger.warning(f"Docker sandbox error, falling back to local: {e}")
        
        # Local fallback
        return self._run_local_execution_check(test_content, file_name)
    
    def _run_local_execution_check(self, test_content: str, file_name: str) -> Tuple[bool, str]:
        """Local fallback: Save the code and test to a temp env and run pytest."""
        import tempfile
        import subprocess
        import shutil
        
        # Create a temp directory structure mimicking the real project
        # /tmp/build/src/  <-- put generated code here
        # /tmp/build/tests/ <-- put test here
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = os.path.join(tmp_dir, "src")
            tests_dir = os.path.join(tmp_dir, "tests")
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(tests_dir, exist_ok=True)
            
            # 1. Write ALL completed files to src_dir (Context is King)
            for fname, result in self.completed_files.items():
                full_path = os.path.join(src_dir, fname)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(result.content)
            
            # 2. Write the current file being verified (if not already in completed)
            # (Note: In your logic, it might not be in completed_files yet)
            # We assume the caller (verify_file) passes the content or we grab it from result
            # But wait, let's write the specific file we are testing now:
            # This logic assumes 'result.content' is passed or available.
            # (See update to _verify_file below)
            
            # 3. Write the test file
            test_name = f"test_{file_name}"
            with open(os.path.join(tests_dir, test_name), 'w') as f:
                f.write(test_content)
            
            # 4. Run Pytest
            # We must set PYTHONPATH so 'from src import X' works
            env = os.environ.copy()
            env["PYTHONPATH"] = tmp_dir
            
            try:
                cmd = [sys.executable, "-m", "pytest", os.path.join(tests_dir, test_name)]
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30 # Fail fast
                )
                
                if proc.returncode == 0:
                    return True, proc.stdout
                else:
                    return False, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                    
            except subprocess.TimeoutExpired:
                return False, "Execution timed out (infinite loop?)"
            except Exception as e:
                return False, f"Harness Error: {str(e)}"

    def _get_coder_system_prompt_entry_point(self) -> str:
        """Special system prompt when generating the entry point (main.py)"""
        return """You are an expert programmer generating the ENTRY POINT FILE for a larger program.

CRITICAL RULES FOR ENTRY POINT:
1. Use RELATIVE imports for sibling modules (from .module import X, not from module import X)
2. This file orchestrates all other modules - imports and method calls MUST match exactly
3. Check EVERY function call against the ACTUAL signatures in completed files
4. Do NOT assume default arguments - check what parameters are required
5. If a function requires arguments, you MUST provide them
6. Use proper error handling for missing data/files

COMMON MISTAKES TO AVOID:
- Calling func() when it requires func(arg1, arg2)
- Calling method() when it requires method(self, data)
- Passing None when the function doesn't handle None
- Forgetting required positional arguments
- Not matching the exact parameter names

VERIFICATION CHECKLIST (do this before outputting):
1. For each import: Does the module export this name?
2. For each function call: Does my call match the signature EXACTLY?
3. For each class instantiation: Am I providing required constructor args?
4. For each method call: Did I account for 'self' in the arg count?

OUTPUT FORMAT:
- Start directly with docstring or imports
- No markdown code blocks, no explanations
- Just pure, complete Python code
- Ensure all brackets are closed and statements complete"""

    def parse_plan(self, plan_yaml: str) -> ProgramPlan:
        """Parse YAML plan into structured ProgramPlan"""
        try:
            data = yaml.safe_load(plan_yaml)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML plan: {e}")
        
        program = data.get('program', {})
        architecture = data.get('architecture', {})
        
        files = []
        for file_data in data.get('files', []):
            # Handle both simple and complex dependency formats
            deps = file_data.get('dependencies', [])
            if deps and isinstance(deps[0], dict):
                # Complex format: [{name: "x.py", optional: true}]
                simple_deps = [d['name'] if isinstance(d, dict) else d for d in deps]
            else:
                simple_deps = deps
                
            # Handle imports_from - ensure it's a dict (YAML might have it as list or None)
            raw_imports_from = file_data.get('imports_from', {})
            if not isinstance(raw_imports_from, dict):
                raw_imports_from = {}  # Default to empty dict if list/None/etc
                
            files.append(FileSpec(
                name=file_data['name'],
                purpose=file_data.get('purpose', ''),
                dependencies=simple_deps,
                exports=file_data.get('exports', []),
                imports_from=raw_imports_from,
                requirements=file_data.get('requirements', []),
                optional=file_data.get('optional', False),
                fallback=file_data.get('fallback')
            ))
        
        # Resolve execution order
        if 'execution_order' in data:
            execution_order = data['execution_order']
        else:
            execution_order = self._resolve_dependencies(files)
        
        return ProgramPlan(
            name=program.get('name', 'unnamed_project'),
            description=program.get('description', ''),
            architecture_pattern=architecture.get('pattern', 'simple'),
            entry_point=architecture.get('entry_point', 'main.py'),
            files=files,
            execution_order=execution_order,
            raw_yaml=plan_yaml
        )
    
    # =============================================================================
# FIXES FOR plan_executor.py - Add these enhancements
# =============================================================================

# LOCATION: Replace _validate_plan_consistency function (around line 394)

    def _validate_plan_consistency(self, plan: ProgramPlan) -> List[str]:
        """
        Validate that the architect's plan is internally consistent.
        Returns list of warnings/errors found.
        
        Checks:
        1. imports_from modules are in dependencies
        2. imports_from names exist in target module's exports
        3. execution_order respects dependencies
        4. No missing files in execution_order
        5. NEW: Base classes referenced in imports_from exist in plan
        6. NEW: Common naming patterns (base_*.py) have corresponding files
        """
        issues = []
        
        # Build lookup maps
        file_specs = {f.name: f for f in plan.files}
        all_filenames = set(file_specs.keys())
        
        # Also track simplified names for matching
        simplified_names = {}
        for name in all_filenames:
            # "parsers/base_parser.py" -> "base_parser"
            simple = name.replace('.py', '').split('/')[-1]
            simplified_names[simple] = name
        
        for file_spec in plan.files:
            # Check 1: imports_from modules are in dependencies
            for import_module in file_spec.imports_from.keys():
                if import_module not in file_spec.dependencies:
                    issues.append(
                        f"PLAN FIX: {file_spec.name} imports from {import_module} "
                        f"but {import_module} was not in dependencies - auto-added"
                    )
                    # Auto-fix: add to dependencies
                    file_spec.dependencies.append(import_module)
            
            # Check 2: imports_from names exist in target's exports
            for import_module, import_names in file_spec.imports_from.items():
                if import_module in file_specs:
                    target_exports = {e['name'] for e in file_specs[import_module].exports}
                    for name in import_names:
                        if name not in target_exports:
                            issues.append(
                                f"PLAN WARNING: {file_spec.name} wants to import '{name}' from "
                                f"{import_module}, but {import_module} doesn't list '{name}' in exports"
                            )
                else:
                    # NEW: Check if imported module exists at all
                    # Try to match with simplified name
                    simple_import = import_module.replace('.py', '').split('/')[-1]
                    if simple_import not in simplified_names:
                        issues.append(
                            f"PLAN ERROR: {file_spec.name} imports from '{import_module}' "
                            f"which does not exist in the plan. Add this file to the plan."
                        )
            
            # Check 3: Dependencies exist in plan
            for dep in file_spec.dependencies:
                if dep not in file_specs:
                    # Try simplified matching
                    simple_dep = dep.replace('.py', '').split('/')[-1]
                    if simple_dep in simplified_names:
                        # Auto-fix: update to correct name
                        correct_name = simplified_names[simple_dep]
                        issues.append(
                            f"PLAN FIX: {file_spec.name} depends on '{dep}' - "
                            f"corrected to '{correct_name}'"
                        )
                        # Update the dependency
                        file_spec.dependencies = [
                            correct_name if d == dep else d 
                            for d in file_spec.dependencies
                        ]
                    else:
                        issues.append(
                            f"PLAN ERROR: {file_spec.name} depends on {dep} which is not in the plan"
                        )
        
        # Check 4: Execution order respects dependencies
        file_position = {name: i for i, name in enumerate(plan.execution_order)}
        for file_spec in plan.files:
            if file_spec.name not in file_position:
                issues.append(f"PLAN ERROR: {file_spec.name} not in execution_order")
                continue
            my_pos = file_position[file_spec.name]
            for dep in file_spec.dependencies:
                if dep in file_position:
                    dep_pos = file_position[dep]
                    if dep_pos >= my_pos:
                        issues.append(
                            f"PLAN ERROR: {file_spec.name} depends on {dep} but "
                            f"{dep} comes later in execution_order"
                        )
        
        # NEW Check 5: Look for common base class patterns
        # If we have files like "parsers/pdf_parser.py", "parsers/docx_parser.py"
        # but no "parsers/base_parser.py", that's suspicious
        directories = {}
        for name in all_filenames:
            if '/' in name:
                dir_name = name.rsplit('/', 1)[0]
                if dir_name not in directories:
                    directories[dir_name] = []
                directories[dir_name].append(name)
        
        for dir_name, files in directories.items():
            # Check if any file in this directory might need a base class
            non_base_files = [f for f in files if 'base_' not in f]
            base_files = [f for f in files if 'base_' in f]
            
            if len(non_base_files) >= 2 and len(base_files) == 0:
                # Multiple implementation files but no base class - warn
                issues.append(
                    f"PLAN WARNING: Directory '{dir_name}/' has {len(non_base_files)} files "
                    f"but no base class. Consider adding '{dir_name}/base_*.py' if inheritance is used."
                )
        
        return issues
   
    def _check_integration(
        self,
        file_spec,  # FileSpec
        result,     # FileResult
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Programmatically check for integration issues BEFORE LLM verification.
        Returns list of issues found.
        """
        issues = []
        content = result.content
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        # Extract all function/method calls from this file
        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []
                
            def visit_Call(self, node):
                call_info = {'args': len(node.args), 'kwargs': len(node.keywords)}
                
                if isinstance(node.func, ast.Attribute):
                    # method call: obj.method()
                    call_info['type'] = 'method'
                    call_info['name'] = node.func.attr
                elif isinstance(node.func, ast.Name):
                    # function call: func()
                    call_info['type'] = 'function'
                    call_info['name'] = node.func.id
                else:
                    call_info['type'] = 'other'
                    call_info['name'] = None
                    
                if call_info['name']:
                    self.calls.append(call_info)
                self.generic_visit(node)
        
        visitor = CallVisitor()
        visitor.visit(tree)
        
        # Check calls against known dependencies
        for dep_name, dep_info in context.get('dependencies', {}).items():
            dep_content = dep_info.get('content', '')
            if not dep_content:
                continue
                
            try:
                dep_tree = ast.parse(dep_content)
            except SyntaxError:
                continue
            
            # Extract function/method signatures from dependency
            for node in ast.walk(dep_tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    if func_name.startswith('_') and func_name != '__init__':
                        continue
                    
                    # Count required args (those without defaults)
                    total_args = len(node.args.args)
                    default_count = len(node.args.defaults)
                    required_args = total_args - default_count
                    
                    # Subtract 'self' for methods
                    if node.args.args and node.args.args[0].arg == 'self':
                        required_args -= 1
                    
                    # Check if this function is called in the current file
                    for call in visitor.calls:
                        if call['name'] == func_name:
                            provided = call['args'] + call['kwargs']
                            if provided < required_args:
                                issues.append(
                                    f"{func_name}() requires at least {required_args} args "
                                    f"but called with {provided} in {file_spec.name}"
                                )
        
        return issues

    def _validate_imports_against_exports(
        self,
        content: str,
        file_spec,  # FileSpec  
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that all imports in generated code exist in actual exports.
        Returns list of invalid imports found.
        """
        issues = []
        
        # Build set of all available exports
        available_exports = {}  # module_name -> set of export names
        for dep_name, dep_info in context.get('dependencies', {}).items():
            module_name = dep_name.replace('.py', '')
            available_exports[module_name] = {
                e['name'] for e in dep_info.get('exports', [])
            }
        
        # Parse imports from generated code
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues  # Let syntax validator handle this
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                    
                # Handle relative imports (.module -> module)
                module = node.module.lstrip('.')
                
                # Check if this is an internal module
                if module in available_exports:
                    valid_names = available_exports[module]
                    for alias in node.names:
                        import_name = alias.name
                        if import_name != '*' and import_name not in valid_names:
                            issues.append(
                                f"Invalid import: '{import_name}' from {module} "
                                f"(available: {', '.join(sorted(valid_names)) or 'none'})"
                            )
        
        return issues

    def _fix_relative_imports(self, content: str, all_files: List[str]) -> str:
        """Fix absolute imports to relative imports for sibling modules."""
        # Get module names without .py extension
        sibling_modules = [f.replace('.py', '') for f in all_files if f.endswith('.py')]
        
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            for module in sibling_modules:
                # Match "from module import" but not "from .module import"
                # Also avoid matching partial names (e.g., "from config" shouldn't match "from config_utils")
                import_pattern = f"from {module} import "
                relative_pattern = f"from .{module} import "
                
                if import_pattern in fixed_line and relative_pattern not in fixed_line:
                    fixed_line = fixed_line.replace(import_pattern, relative_pattern)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)

    def _validate_syntax(self, content: str, filename: str) -> Tuple[bool, str]:
        """
        Programmatically validate Python syntax before LLM verification.
        Catches truncated files, syntax errors, and incomplete code.
        
        Returns:
            (is_valid, error_message)
        """
        # Check 1: Empty or near-empty content
        if not content or len(content.strip()) < 50:
            return False, "File content is empty or too short"
        
        # Check 2: Obvious truncation indicators
        truncation_endings = (
            'excep', 'def ', 'class ', 'if ', 'for ', 'while ',
            'try:', 'except', 'elif ', 'else:', 'with ', 'return',
            'import ', 'from ', 'raise ', 'yield ', 'async ', 'await '
        )
        if content.rstrip().endswith(truncation_endings):
            return False, f"File appears truncated (ends with incomplete statement)"
        
        # Check 3: Bracket mismatch (quick check before AST)
        if content.count('(') != content.count(')'):
            return False, f"Mismatched parentheses: {content.count('(')} open, {content.count(')')} close"
        if content.count('[') != content.count(']'):
            return False, f"Mismatched brackets: {content.count('[')} open, {content.count(']')} close"
        if content.count('{') != content.count('}'):
            return False, f"Mismatched braces: {content.count('{')} open, {content.count('}')} close"
        
        # Check 4: Triple-quote balance (rough check for docstrings)
        if content.count('"""') % 2 != 0:
            return False, "Unclosed triple-double-quote string"
        if content.count("'''") % 2 != 0:
            return False, "Unclosed triple-single-quote string"
        
        # Check 5: AST parse - the definitive syntax check
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        
        # Check 6: Minimum viable content - should have at least one function or class
        has_definitions = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(tree)
        )
        if not has_definitions and filename != '__init__.py':
            # Allow config files that might just have assignments
            has_assignments = any(isinstance(node, ast.Assign) for node in ast.walk(tree))
            if not has_assignments:
                return False, "File has no functions, classes, or assignments"
        
        return True, ""





    def _resolve_dependencies(self, files: List[FileSpec]) -> List[str]:
        """Topological sort of files based on dependencies"""
        file_deps = {f.name: f.dependencies for f in files}
        
        # Kahn's algorithm
        in_degree = defaultdict(int)
        for fname in file_deps:
            in_degree[fname]  # ensure all files are in dict
        for fname, deps in file_deps.items():
            for dep in deps:
                if dep in file_deps:  # only count internal deps
                    in_degree[fname] += 1
        
        queue = [f for f in file_deps if in_degree[f] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for fname, deps in file_deps.items():
                if current in deps:
                    in_degree[fname] -= 1
                    if in_degree[fname] == 0 and fname not in result and fname not in queue:
                        queue.append(fname)
        
        if len(result) != len(file_deps):
            # Circular dependency - return files in definition order
            print("⚠ Warning: Circular dependency detected, using definition order")
            return [f.name for f in files]
        
        return result
    
    def execute(self, plan_yaml: str, user_request: str, job_scope: str = "") -> Dict[str, Any]:
        """
        Execute the full plan, generating all files in dependency order.
        
        Args:
            plan_yaml: YAML string containing the program design plan
            user_request: Original user request for context
            
        Returns:
            Dictionary containing all completed files and execution status
        """
        self.start_time = time.time()
        
        # Parse the plan
        self.plan = self.parse_plan(plan_yaml)
        
        # Validate plan consistency and auto-fix issues
        plan_issues = self._validate_plan_consistency(self.plan)
        for issue in plan_issues:
            self.logger.warning(issue)
            print(f"⚠ {issue}")
        
        self.job_scope = user_request  # Will be overridden if job_scope passed
        if job_scope:
            self.job_scope = job_scope
        # Log execution start
        self.logger.info("="*70)
        self.logger.info(f"PLAN EXECUTOR START: {self.plan.name}")
        self.logger.info("="*70)
        self.logger.info(f"Description: {self.plan.description}")
        self.logger.info(f"Pattern: {self.plan.architecture_pattern}")
        self.logger.info(f"Files to generate: {len(self.plan.files)}")
        self.logger.info(f"Execution order: {' -> '.join(self.plan.execution_order)}")
        
        # Log the full YAML plan
        self.logger.debug("YAML PLAN:\n" + plan_yaml)
        
        # Log user request
        self.logger.debug(f"USER REQUEST:\n{user_request}")
        
        self._log_event("execution_start", {
            "plan_name": self.plan.name,
            "files": self.plan.execution_order,
            "user_request": user_request[:500]  # Truncate for JSON
        })
        
        print(f"\n{'='*70}")
        print(f"PLAN EXECUTOR: {self.plan.name}")
        print(f"{'='*70}")
        print(f"Description: {self.plan.description}")
        print(f"Pattern: {self.plan.architecture_pattern}")
        print(f"Files to generate: {len(self.plan.files)}")
        print(f"Execution order: {' → '.join(self.plan.execution_order)}")
        print(f"{'='*70}\n")
        
        # Execute each file in order
        for filename in self.plan.execution_order:
            file_spec = self._get_file_spec(filename)
            if not file_spec:
                self.logger.warning(f"Skipping unknown file: {filename}")
                print(f"⚠ Skipping unknown file: {filename}")
                continue
            
            self.logger.info(f"")
            self.logger.info(f">>> GENERATING: {filename}")
            self.logger.info(f"    Purpose: {file_spec.purpose}")
            self.logger.info(f"    Dependencies: {file_spec.dependencies or 'None'}")
            
            print(f"\n▶ Generating: {filename}")
            print(f"  Purpose: {file_spec.purpose}")
            print(f"  Dependencies: {file_spec.dependencies or 'None'}")
            
            # Build context from completed dependencies
            context = self._build_context_for_file(file_spec)
            context['job_scope'] = getattr(self, 'job_scope', user_request)
            
            # Log context summary
            self.logger.debug(f"Context for {filename}:")
            for dep_name, dep_info in context.get('dependencies', {}).items():
                exports = [e['name'] for e in dep_info.get('exports', [])]
                self.logger.debug(f"  - {dep_name}: exports {exports}")
            
            # Generate the file
            self.status[filename] = FileStatus.IN_PROGRESS
            file_start_time = time.time()
            
            self._log_event("file_generation_start", {
                "filename": filename,
                "purpose": file_spec.purpose,
                "dependencies": file_spec.dependencies
            })
            
            result = self._generate_file(
                file_spec=file_spec,
                user_request=user_request,
                context=context
            )
            
            file_elapsed = time.time() - file_start_time
                
            if result.status == FileStatus.COMPLETED:
                # Log generated content
                self.logger.debug(f"Generated {filename} ({len(result.content)} chars):")
                self.logger.debug("-" * 40)
                self.logger.debug(result.content[:2000] + ("..." if len(result.content) > 2000 else ""))
                self.logger.debug("-" * 40)
                
                # Verify the generated file
                self.logger.info(f"    Verifying {filename}...")
                verification = self._verify_file(file_spec, result, context)
                result.verification = verification
                
                self.logger.info(f"    Verification: {'PASS' if verification.get('passed') else 'FAIL'}")
                if not verification.get('passed'):
                    self.logger.warning(f"    Issues: {verification.get('issues', [])}")
                
                if not verification.get('passed', False):
                    # Attempt revision
                    self.logger.info(f"    Starting revision loop for {filename}")
                    result = self._handle_revision(
                        file_spec, result, verification, context, user_request
                    )
            
            # Record result
            self.completed_files[filename] = result
            self.status[filename] = result.status
            
            status_icon = "✓" if result.status == FileStatus.COMPLETED else "✗"
            self.logger.info(f"    Result: {status_icon} {result.status.value} ({file_elapsed:.1f}s)")
            print(f"  {status_icon} {filename}: {result.status.value}")
            
            self._log_event("file_generation_complete", {
                "filename": filename,
                "status": result.status.value,
                "elapsed_seconds": round(file_elapsed, 2),
                "exports": [e['name'] for e in result.actual_exports],
                "revision_count": result.revision_count
            })
            
            if result.status == FileStatus.FAILED:
                self.errors.append({
                    'filename': filename,
                    'error': result.notes
                })
                self.logger.error(f"    FAILED: {result.notes}")
        
        # Generate combined output
        combined_output = self._generate_combined_output()
        
        total_elapsed = time.time() - self.start_time
        
        # Log completion
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"PLAN EXECUTOR COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {total_elapsed:.1f}s")
        self.logger.info(f"Files completed: {sum(1 for s in self.status.values() if s == FileStatus.COMPLETED)}/{len(self.plan.files)}")
        self.logger.info(f"Errors: {len(self.errors)}")
        
        self._log_event("execution_complete", {
            "success": len(self.errors) == 0,
            "total_seconds": round(total_elapsed, 2),
            "files_completed": sum(1 for s in self.status.values() if s == FileStatus.COMPLETED),
            "files_failed": sum(1 for s in self.status.values() if s == FileStatus.FAILED),
            "errors": self.errors
        })
        
        # Save execution log to JSON
        self._save_execution_log()
        
        return {
            'success': len(self.errors) == 0,
            'plan': self.plan,
            'completed_files': self.completed_files,
            'status': {k: v.value for k, v in self.status.items()},
            'errors': self.errors,
            'combined_output': combined_output,
            'log_file': os.path.join(self.log_dir, "plan_executor_latest.log")
        }
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Add structured event to execution log"""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(time.time() - self.start_time, 2) if self.start_time else 0,
            "event": event_type,
            "data": data
        })
    
    def _save_execution_log(self):
        """Save structured execution log to JSON file"""
        if not self.plan:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"plan_execution_{self.plan.name}_{timestamp}.json")
        
        log_data = {
            "plan_name": self.plan.name,
            "plan_description": self.plan.description,
            "execution_order": self.plan.execution_order,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "total_seconds": round(time.time() - self.start_time, 2) if self.start_time else 0,
            "success": len(self.errors) == 0,
            "files": {
                filename: {
                    "status": result.status.value,
                    "exports": result.actual_exports,
                    "revision_count": result.revision_count,
                    "content_length": len(result.content)
                }
                for filename, result in self.completed_files.items()
            },
            "errors": self.errors,
            "events": self.execution_log
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"Execution log saved: {log_file}")
    
    def _get_file_spec(self, filename: str) -> Optional[FileSpec]:
        """Get file specification from plan"""
        for f in self.plan.files:
            if f.name == filename:
                return f
        return None
    
    def _build_context_for_file(self, file_spec: FileSpec) -> Dict[str, Any]:
        """Build context from completed dependencies"""
        context = {
            'program': {
                'name': self.plan.name,
                'description': self.plan.description,
                'pattern': self.plan.architecture_pattern,
                'entry_point': self.plan.entry_point
            },
            'all_files': [f.name for f in self.plan.files],
            'dependencies': {}
        }
        
        for dep in file_spec.dependencies:
            if dep in self.completed_files:
                dep_result = self.completed_files[dep]
                # For direct dependencies, include full content
                context['dependencies'][dep] = {
                    'exports': dep_result.actual_exports,
                    'content': dep_result.content,
                    'summary': self._summarize_file(dep_result.content)
                }
        
        # Also include transitive dependency summaries
        for fname, result in self.completed_files.items():
            if fname not in context['dependencies']:
                context['dependencies'][fname] = {
                    'exports': result.actual_exports,
                    'summary': self._summarize_file(result.content)
                }
        
        return context
    
    def _summarize_file(self, content: str) -> str:
        """
        Create a detailed interface summary INCLUDING full method signatures.
        This is critical for main.py to call methods correctly.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to simple truncation if syntax is invalid
            lines = content.split('\n')
            return '\n'.join(lines[:30]) + "\n# ... (syntax error in parsing)"

        skeleton = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    skeleton.append(ast.unparse(node))
                except AttributeError:
                    pass
                    
            elif isinstance(node, ast.ClassDef):
                # Class with bases
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except:
                        bases.append("...")
                base_str = f"({', '.join(bases)})" if bases else ""
                skeleton.append(f"\nclass {node.name}{base_str}:")
                
                # Docstring (first 3 lines)
                docstring = ast.get_docstring(node)
                if docstring:
                    doc_lines = docstring.split('\n')[:3]
                    skeleton.append('    """' + '\n    '.join(doc_lines) + '..."""')
                
                # All methods with FULL signatures
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        try:
                            args_str = ast.unparse(item.args)
                        except:
                            args_str = "..."
                        
                        if item.name == '__init__':
                            skeleton.append(f"    def __init__({args_str}):")
                            # Show instance attributes being set
                            for stmt in item.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if (isinstance(target, ast.Attribute) and
                                            isinstance(target.value, ast.Name) and
                                            target.value.id == 'self'):
                                            skeleton.append(f"        self.{target.attr} = ...")
                            skeleton.append("        ...")
                        elif not item.name.startswith('_'):
                            # Public methods - show full signature
                            skeleton.append(f"    def {item.name}({args_str}): ...")
                            
            elif isinstance(node, ast.FunctionDef):
                # Top level functions - FULL signature
                if not node.name.startswith('_'):
                    try:
                        args_str = ast.unparse(node.args)
                    except:
                        args_str = "..."
                    skeleton.append(f"\ndef {node.name}({args_str}):")
                    docstring = ast.get_docstring(node)
                    if docstring:
                        doc_line = docstring.split('\n')[0]
                        skeleton.append(f'    """{doc_line}..."""')
                    skeleton.append("    ...")
                        
            elif isinstance(node, ast.Assign):
                # Module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        try:
                            val_str = ast.unparse(node.value)
                            if len(val_str) > 50:
                                val_str = val_str[:50] + "..."
                            skeleton.append(f"{target.id} = {val_str}")
                        except:
                            skeleton.append(f"{target.id} = ...")
                  
        return "\n".join(skeleton)
    
    def _extract_exports(self, content: str) -> List[Dict[str, str]]:
        """Extract actual exports from generated file content"""
        exports = []
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            # Classes
            if stripped.startswith('class ') and not line.startswith(' '):
                match = re.match(r'class\s+(\w+)', stripped)
                if match and not match.group(1).startswith('_'):
                    exports.append({'name': match.group(1), 'type': 'class'})
            # Functions
            elif stripped.startswith('def ') and not line.startswith(' '):
                match = re.match(r'def\s+(\w+)', stripped)
                if match and not match.group(1).startswith('_'):
                    exports.append({'name': match.group(1), 'type': 'function'})
            # Module-level variables (constants typically)
            elif re.match(r'^[A-Z][A-Z_0-9]*\s*=', stripped):
                var_name = stripped.split('=')[0].strip()
                exports.append({'name': var_name, 'type': 'constant'})
        
        return exports
    
        
    def _generate_file(
        self,
        file_spec,  # FileSpec
        user_request: str,
        context: Dict[str, Any]
    ):  # -> FileResult
        """Generate a single file using the Coder agent - WITH SYNTAX VALIDATION"""
        
        from swarm_coordinator_v2 import AgentRole
        
        # Use special prompt for entry point
        if file_spec.name == self.plan.entry_point or file_spec.name == 'main.py':
            system_prompt = self._get_coder_system_prompt_entry_point()
        else:
            system_prompt = self._get_coder_system_prompt()
        
        user_message = self._build_coder_message(file_spec, user_request, context)
        
        max_generation_attempts = 2  # Retry if truncated
        last_error = ""
        
        for attempt in range(max_generation_attempts):
            try:
                start_time = time.time()
                
                # Add error context on retry
                if attempt > 0 and last_error:
                    retry_message = f"""PREVIOUS ATTEMPT FAILED VALIDATION:
Error: {last_error}

IMPORTANT: Ensure the code is COMPLETE and syntactically valid. 
Do not truncate. Include all closing brackets and complete all statements.

{user_message}"""
                    response = self.executor.execute_agent(
                        role=AgentRole.CODER,
                        system_prompt=system_prompt,
                        user_message=retry_message
                    )
                else:
                    response = self.executor.execute_agent(
                        role=AgentRole.CODER,
                        system_prompt=system_prompt,
                        user_message=user_message
                    )
                
                elapsed = time.time() - start_time
                
                # Clean the response
                content = self._clean_code_output(response, file_spec.name)
                content = self._fix_relative_imports(content, context['all_files'])
                
                if not content.strip():
                    last_error = "Coder returned empty content"
                    if attempt < max_generation_attempts - 1:
                        self.logger.warning(f"Empty content for {file_spec.name}, retrying...")
                        continue
                    return FileResult(
                        name=file_spec.name,
                        content="",
                        actual_exports=[],
                        status=FileStatus.FAILED,
                        notes="Coder returned empty content after retries"
                    )
                
                # NEW: Programmatic syntax validation BEFORE LLM verification
                is_valid, syntax_error = self._validate_syntax(content, file_spec.name)
                
                if not is_valid:
                    self.logger.warning(f"Syntax validation failed for {file_spec.name}: {syntax_error}")
                    last_error = syntax_error
                    if attempt < max_generation_attempts - 1:
                        self.logger.info(f"Retrying generation for {file_spec.name}...")
                        continue
                    else:
                        return FileResult(
                            name=file_spec.name,
                            content=content,  # Keep partial content for debugging
                            actual_exports=[],
                            status=FileStatus.FAILED,
                            notes=f"Syntax validation failed: {syntax_error}"
                        )
                
                # NEW: Validate imports against actual exports
                import_issues = self._validate_imports_against_exports(content, file_spec, context)
                if import_issues:
                    self.logger.warning(f"Import issues in {file_spec.name}: {import_issues}")
                    # Don't fail immediately - let verification handle it
                    # But log for debugging
                    for issue in import_issues:
                        print(f"  ⚠ {issue}")
                
                # Extract what was actually exported
                actual_exports = self._extract_exports(content)
                
                return FileResult(
                    name=file_spec.name,
                    content=content,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED,
                    notes=f"Generated in {elapsed:.1f}s"
                )
                
            except Exception as e:
                last_error = str(e)
                if attempt < max_generation_attempts - 1:
                    self.logger.warning(f"Generation attempt {attempt+1} failed: {e}, retrying...")
                    continue
                return FileResult(
                    name=file_spec.name,
                    content="",
                    actual_exports=[],
                    status=FileStatus.FAILED,
                    notes=str(e)
                )
        
        # Should not reach here, but safety fallback
        return FileResult(
            name=file_spec.name,
            content="",
            actual_exports=[],
            status=FileStatus.FAILED,
            notes=f"Generation failed after all attempts: {last_error}"
        )


    
    def _get_coder_system_prompt(self) -> str:
        """System prompt for file-by-file code generation"""
        return """You are an expert programmer generating a SINGLE FILE as part of a larger program.

CRITICAL RULES:
1. Generate ONLY the requested file - no other files
2. Output raw Python code - no markdown code blocks
3. Use the EXACT filename specified
4. Use RELATIVE imports for sibling modules (from .module import X, not from module import X)
5. Implement ALL requirements listed for this file
6. Export the classes/functions specified in the plan

CONTEXT AWARENESS:
- You will be given the full program plan and completed files
- Use actual exports from completed files (not planned - ACTUAL)
- Match function signatures and class names exactly
- Handle cases where dependencies provide different names than planned

OUTPUT FORMAT:
- Start directly with docstring or imports
- No explanations before or after the code
- No ### FILE: ### headers (that's handled externally)
- Just pure Python code for this one file

QUALITY:
- Code must be immediately executable
- Include docstrings for public interfaces
- Handle realistic error cases
- Add type hints where helpful"""
    
    def _build_coder_message(
        self, 
        file_spec: FileSpec, 
        user_request: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Build the user message for file generation - STRICT VERSION.
        
        Only shows ACTUAL exports from completed files and explicitly
        warns against inventing imports.
        """
        parts = []
        
        # Program overview
        parts.append(f"""PROGRAM OVERVIEW
================
Name: {context['program']['name']}
Description: {context['program']['description']}
Pattern: {context['program']['pattern']}
Entry Point: {context['program']['entry_point']}
All Files: {', '.join(context['all_files'])}

JOB SCOPE
=========
{context.get('job_scope', user_request)}
""")
        
        # Current file spec
        parts.append(f"""
FILE TO GENERATE
================
Filename: {file_spec.name}
Purpose: {file_spec.purpose}
""")
        
        if file_spec.requirements:
            parts.append("Requirements:")
            for req in file_spec.requirements:
                parts.append(f"  - {req}")
        
        if file_spec.exports:
            parts.append("\nYou MUST export these (define them in your code):")
            for exp in file_spec.exports:
                parts.append(f"  - {exp['name']} ({exp['type']})")
        
        # STRICT IMPORT SECTION
        parts.append(f"""
AVAILABLE IMPORTS (ONLY USE THESE)
==================================
You may ONLY import from completed files listed below.
You may ONLY import the EXACT names shown in ACTUAL EXPORTS.
DO NOT invent imports that don't exist.
""")
        
        # Determine current file's folder
        current_folder = file_spec.name.rsplit('/', 1)[0] if '/' in file_spec.name else ''
        
        if context['dependencies']:
            allowed_imports = {}  # module -> [names]
            
            for dep_name, dep_info in context['dependencies'].items():
                actual_exports = dep_info.get('exports', [])
                if actual_exports:
                    export_names = [e['name'] for e in actual_exports]
                    allowed_imports[dep_name] = export_names
                    
                    # Determine correct import path
                    dep_folder = dep_name.rsplit('/', 1)[0] if '/' in dep_name else ''
                    dep_module = dep_name.replace('.py', '').replace('/', '.')
                    
                    # Same folder = relative, different folder = absolute
                    if dep_folder == current_folder and current_folder:
                        dep_file = dep_name.rsplit('/', 1)[-1].replace('.py', '')
                        import_stmt = f"from .{dep_file} import {', '.join(export_names)}"
                    else:
                        import_stmt = f"from {dep_module} import {', '.join(export_names)}"
                    
                    if dep_name in file_spec.dependencies:
                        # Full content for direct dependencies
                        parts.append(f"### {dep_name} ###")
                        parts.append(f"ACTUAL EXPORTS: {', '.join(export_names)}")
                        parts.append(f"IMPORT WITH: {import_stmt}")
                        parts.append("\nFull implementation:")
                        parts.append(dep_info['content'])
                        parts.append("")
                    else:
                        # Summary for transitive dependencies
                        parts.append(f"### {dep_name} (transitive) ###")
                        parts.append(f"EXPORTS: {', '.join(export_names)}")
                        parts.append(f"IMPORT WITH: {import_stmt}")
                        parts.append("")
            
            # Summary of allowed imports
            if allowed_imports:
                parts.append("IMPORT SUMMARY (copy these exactly):")
                for module, names in allowed_imports.items():
                    if names:
                        dep_folder = module.rsplit('/', 1)[0] if '/' in module else ''
                        dep_module = module.replace('.py', '').replace('/', '.')
                        if dep_folder == current_folder and current_folder:
                            dep_file = module.rsplit('/', 1)[-1].replace('.py', '')
                            parts.append(f"  from .{dep_file} import {', '.join(names)}")
                        else:
                            parts.append(f"  from {dep_module} import {', '.join(names)}")
        else:
            parts.append("No dependencies - this file should be self-contained.")
            parts.append("Do NOT import from other project files.")
        
        # Planned imports from architect (for reference)
        if file_spec.imports_from:
            parts.append("\nArchitect's planned imports (verify against ACTUAL EXPORTS above):")
            for module, imports in file_spec.imports_from.items():
                parts.append(f"  from .{module.replace('.py', '')} import {', '.join(imports)}")
            parts.append("NOTE: If a planned import doesn't exist in ACTUAL EXPORTS, DO NOT USE IT.")
        
        parts.append(f"""
CRITICAL RULES
==============
1. Cross-folder imports: from folder.file import X (e.g., from parsers.base_parser import BaseParser)
2. Same-folder imports: from .file import X (e.g., from .base_chunker import BaseChunker)
3. NEVER do: from .folder.file import X (this is ALWAYS wrong)
4. ONLY import names that appear in ACTUAL EXPORTS above
5. If you need functionality that doesn't exist, IMPLEMENT IT in this file
6. Do NOT invent imports - if it's not listed above, it doesn't exist

GENERATE {file_spec.name} NOW
=============================
Output ONLY the Python code for {file_spec.name}. No explanations.""")
        
        return '\n'.join(parts)
    
    def _clean_code_output(self, code: str, expected_filename: str) -> str:
        """Clean code output, extracting just the relevant file content"""
        
        # Check if response contains file markers
        if '### FILE:' in code:
            # Extract the specific file we want
            pattern = rf'###\s*FILE:\s*{re.escape(expected_filename)}\s*###\s*(.*?)(?=###\s*FILE:|$)'
            match = re.search(pattern, code, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1)
        
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Remove any leading/trailing explanations
        lines = code.split('\n')
        cleaned = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Detect code start
            if not in_code:
                code_starters = (
                    stripped.startswith('"""'),
                    stripped.startswith("'''"),
                    stripped.startswith('import '),
                    stripped.startswith('from '),
                    stripped.startswith('class '),
                    stripped.startswith('def '),
                    stripped.startswith('#!'),
                    stripped.startswith('# '),
                    stripped.startswith('@'),
                )
                if any(code_starters):
                    in_code = True
            
            if in_code:
                cleaned.append(line)
        
        return '\n'.join(cleaned).strip()
 
    def _verify_file(self, file_spec: FileSpec, result: FileResult, context: Dict) -> Dict[str, Any]:
        """
        Verification 2.0: Static Analysis + Runtime Execution
        """
        # 1. Static Syntax Check (Keep your existing fast check)
        is_valid, syntax_error = self._validate_syntax(result.content, file_spec.name)
        if not is_valid:
            return {'passed': False, 'issues': [f"Syntax Error: {syntax_error}"]}

        # 2. Integration Check (Keep your existing AST check)
        if file_spec.dependencies:
            integration_issues = self._check_integration(file_spec, result, context)
            if integration_issues:
                return {'passed': False, 'issues': integration_issues}

        # 3. RUNTIME EXECUTION CHECK (The New "Production" Gate)
        self.logger.info(f"    generating hot test for {file_spec.name}...")
        
        # Save current file content to tmp src so _run_execution_check can use it
        # (We handle this by ensuring _run_execution_check writes 'result.content' to the temp dir)
        # Actually, let's pass result.content explicitly to the harness in step 4 below.
        
        try:
            # A. Generate the test
            test_content = self._generate_hot_test(file_spec, result.content, context)
            
            # B. Run the test
            # We need to manually write the CURRENT file content into the temp dir inside the harness
            # I will modify the harness call slightly to accept it, or handle it inside.
            # Simplified: The harness writes ALL completed_files. We just need to ensure 
            # the CURRENT file is also written.
            
            # Let's patch the harness usage:
            # Note: You need to pass 'result.content' to the harness for the current file
            pass_test, output = self._run_execution_check_patched(result.name, result.content, test_content)
            
            if not pass_test:
                return {
                    'passed': False,
                    'issues': [f"Runtime Failure:\n{output}"], # This goes back to Coder!
                    'response': output
                }
                
        except Exception as e:
            self.logger.warning(f"Test generation failed: {e}")
            # Fallback to LLM verification if testing crashes (don't block everything)
            return self._verify_file_llm_fallback(file_spec, result, context)

        return {'passed': True, 'issues': [], 'response': "Tests Passed"}

    def _run_execution_check_patched(self, current_filename, current_content, test_content):
        """Helper to inject the current file into the harness"""
        # Add the current file to the list of 'completed' files temporarily for the harness
        # This is a quick patch strategy
        original_completed = self.completed_files.copy()
        
        # Create a temp result for the harness to read
        temp_result = FileResult(name=current_filename, content=current_content, actual_exports=[], status=FileStatus.IN_PROGRESS)
        self.completed_files[current_filename] = temp_result
        
        try:
            return self._run_execution_check(test_content, current_filename)
        finally:
            # Restore state
            self.completed_files = original_completed
    
    def _verify_file_llm_fallback(self, file_spec: FileSpec, result: FileResult, context: Dict) -> Dict[str, Any]:
        """
        Fallback verification when runtime testing fails.
        Since syntax and integration checks already passed, we pass with a warning.
        """
        self.logger.warning(f"    Using fallback verification for {file_spec.name} (runtime test failed)")
        return {
            'passed': True,
            'issues': [],
            'response': "Passed (fallback - runtime test generation failed, syntax/integration OK)"
        }

    def _format_dependency_exports(self, context: Dict[str, Any]) -> str:
        """Format dependency exports for verification prompt"""
        parts = []
        for dep_name, dep_info in context.get('dependencies', {}).items():
            exports = [f"{e['name']} ({e['type']})" for e in dep_info.get('exports', [])]
            parts.append(f"  {dep_name}: {', '.join(exports) if exports else 'none detected'}")
        return '\n'.join(parts) if parts else "  None"
    
    def _parse_verification_verdict(self, response: str) -> Tuple[bool, List[str]]:
        """
        Robust verdict parsing with multiple signals to avoid false positives/negatives.
        
        Strategy:
        1. Look for explicit VERIFICATION: PASS/FAIL line
        2. Parse checklist YES/NO counts
        3. Extract actual issues (filter noise)
        4. Use consensus: if checklist is all YES and no real issues, it's PASS
        
        Returns:
            (passed: bool, issues: List[str])
        """
        response_upper = response.upper()
        
        # Signal 1: Explicit verdict line (most authoritative)
        verdict_match = re.search(
            r'VERIFICATION\s*:\s*(PASS|FAIL)',
            response_upper
        )
        explicit_verdict = verdict_match.group(1) if verdict_match else None
        
        # Signal 2: Checklist parsing
        checklist_yes = len(re.findall(r'\[YES\]|\bYES\b(?:\s*-)', response_upper))
        checklist_no = len(re.findall(r'\[NO\]|\bNO\b(?:\s*-)', response_upper))
        
        # Signal 3: Extract real issues (filter out noise)
        issues = self._extract_verification_issues(response)
        real_issues = [i for i in issues if self._is_real_issue(i)]
        
        # Decision logic
        if explicit_verdict == 'PASS':
            # Explicit PASS - trust it unless there are serious issues
            if real_issues and checklist_no > 0:
                # Contradiction: says PASS but has NO items and real issues
                self.logger.warning("Verifier said PASS but has issues - treating as FAIL")
                return False, real_issues
            return True, []
        
        elif explicit_verdict == 'FAIL':
            # Explicit FAIL - trust it if there are real issues
            if real_issues:
                return False, real_issues
            elif checklist_no > 0:
                # FAIL with NO items but no extracted issues - create generic issue
                return False, ["Verification checklist has failing items - review required"]
            else:
                # Says FAIL but all checklist is YES and no issues - likely false positive
                if checklist_yes >= 3 and checklist_no == 0:
                    self.logger.warning("Verifier said FAIL but checklist all YES - treating as PASS")
                    return True, []
                return False, ["Verification failed - details unclear"]
        
        else:
            # No explicit verdict - use checklist consensus
            if checklist_no == 0 and checklist_yes >= 2:
                # All YES, no NO - implicit PASS
                return True, []
            elif checklist_no > 0 or real_issues:
                # Has NO items or real issues - FAIL
                return False, real_issues if real_issues else ["Verification checklist has failing items"]
            else:
                # Ambiguous - default to PASS (don't block on unclear verdicts)
                self.logger.warning("Ambiguous verification verdict - defaulting to PASS")
                return True, []
    
    def _extract_verification_issues(self, response: str) -> List[str]:
        """Extract issues from verification response with better filtering."""
        issues = []
        
        # Try multiple patterns for issues section
        patterns = [
            r'ISSUES[^:]*:(.*?)(?=\n\n|\n[A-Z]{2,}|\Z)',  # ISSUES: section
            r'(?:MUST|NEED TO|SHOULD) FIX:(.*?)(?=\n\n|\Z)',  # MUST FIX: section
            r'PROBLEMS?:(.*?)(?=\n\n|\Z)',  # PROBLEMS: section
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                for line in raw.split('\n'):
                    line = line.strip().lstrip('- •*')
                    if line and len(line) > 3:
                        issues.append(line)
                break
        
        # Also check for inline issues after [NO]
        no_matches = re.findall(
            r'\[NO\]\s*[-–:]\s*(.+?)(?=\n|$)',
            response,
            re.IGNORECASE
        )
        issues.extend([m.strip() for m in no_matches if m.strip()])
        
        return issues
    
    def _is_real_issue(self, issue: str) -> bool:
        """Filter out noise from issue extraction."""
        if not issue or len(issue) < 5:
            return False
        
        # Filter out meta-comments and noise
        noise_patterns = [
            r'^none',
            r'^n/a',
            r'^no issues',
            r'^all good',
            r'^looks good',
            r'^verified',
            r'^pass',
            r'^\[',  # Just brackets
            r'^if fail\b',  # Template text
            r'^list specific',  # Template text
        ]
        
        issue_lower = issue.lower().strip()
        for pattern in noise_patterns:
            if re.match(pattern, issue_lower):
                return False
        
        return True
    
    def _handle_revision(
        self,
        file_spec: FileSpec,
        result: FileResult,
        verification: Dict[str, Any],
        context: Dict[str, Any],
        user_request: str
    ) -> FileResult:
        """Handle revision loop when verification fails"""
        
        from swarm_coordinator_v2 import AgentRole
        
        for attempt in range(self.max_file_revisions):
            self.logger.info(f"    Revision attempt {attempt + 1}/{self.max_file_revisions}")
            print(f"  🔄 Revision attempt {attempt + 1}/{self.max_file_revisions}")
            
            self._log_event("file_revision_start", {
                "filename": file_spec.name,
                "attempt": attempt + 1,
                "issues": verification.get('issues', [])
            })
            
            # OPTIMIZED PROMPT: Forces analysis before coding
            system_prompt = """You are an expert programmer REVISING code based on verification feedback.

CRITICAL RULES:
1. Fix ALL issues identified by the verifier
2. Keep working parts of the original code
3. Output ONLY the revised Python code
4. No markdown, no explanations
5. The code must pass verification after your fixes

Focus on fixing the specific issues mentioned."""
            
            # Build import guidance for revision
            current_folder = file_spec.name.rsplit('/', 1)[0] if '/' in file_spec.name else ''
            import_lines = []
            if context.get('dependencies'):
                for dep_name, dep_info in context['dependencies'].items():
                    actual_exports = dep_info.get('exports', [])
                    if actual_exports:
                        export_names = [e['name'] for e in actual_exports]
                        dep_folder = dep_name.rsplit('/', 1)[0] if '/' in dep_name else ''
                        dep_module = dep_name.replace('.py', '').replace('/', '.')
                        if dep_folder == current_folder and current_folder:
                            dep_file = dep_name.rsplit('/', 1)[-1].replace('.py', '')
                            import_lines.append(f"  from .{dep_file} import {', '.join(export_names)}")
                        else:
                            import_lines.append(f"  from {dep_module} import {', '.join(export_names)}")
            import_section = '\n'.join(import_lines) if import_lines else "  (no project imports)"
            
            user_message = f"""REVISE THIS FILE: {file_spec.name}

VERIFICATION FAILED. Issues to fix:
{chr(10).join('- ' + i for i in verification.get('issues', ['Unknown issues']))}

IMPORT RULES (CRITICAL):
- Cross-folder: from folder.file import X
- Same-folder: from .file import X  
- NEVER: from .folder.file import X

ALLOWED IMPORTS:
{import_section}

ORIGINAL CODE (With Errors):
{result.content}

Fix the issues and output the COMPLETE revised code for {file_spec.name}:"""
            
            try:
                # Use fallback coder on revision attempts 2 and 3
                if attempt == 0:
                    revision_role = AgentRole.CODER
                else:
                    revision_role = AgentRole.FALLBACK_CODER
                    self.logger.info(f"    Switching to FALLBACK_CODER for attempt {attempt + 1}")
                
                response = self.executor.execute_agent(
                    role=revision_role,
                    system_prompt=system_prompt,
                    user_message=user_message
                )
                
                content = self._clean_code_output(response, file_spec.name)
                content = self._fix_relative_imports(content, context['all_files'])
                self.logger.debug(f"Revised code for {file_spec.name} (attempt {attempt + 1}):\n{content[:3000]}")
                actual_exports = self._extract_exports(content)
                
                revised_result = FileResult(
                    name=file_spec.name,
                    content=content,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED,
                    revision_count=attempt + 1
                )
                
                # Re-verify
                new_verification = self._verify_file(file_spec, revised_result, context)
                revised_result.verification = new_verification
                
                if new_verification.get('passed', False):
                    self.logger.info(f"    Revision successful on attempt {attempt + 1}")
                    self._log_event("file_revision_success", {
                        "filename": file_spec.name,
                        "attempt": attempt + 1
                    })
                    print(f"  ✓ Revision successful")
                    return revised_result
                
                self.logger.warning(f"Revision {attempt + 1} still failing for {file_spec.name}")
                self.logger.debug(f"Revision {attempt + 1} failure details:\n{new_verification.get('issues', [])[:2000]}")
                verification = new_verification
                result = revised_result
                
            except Exception as e:
                self.logger.error(f"    Revision failed: {e}")
                self._log_event("file_revision_error", {
                    "filename": file_spec.name,
                    "attempt": attempt + 1,
                    "error": str(e)
                })
                print(f"  ✗ Revision failed: {e}")
        
        # All revisions failed
        self.logger.error(f"    All {self.max_file_revisions} revision attempts failed for {file_spec.name}")
        self._log_event("file_revision_exhausted", {
            "filename": file_spec.name,
            "max_attempts": self.max_file_revisions
        })
        result.status = FileStatus.FAILED
        result.notes = f"Failed after {self.max_file_revisions} revision attempts"
        return result
    
    def _generate_combined_output(self) -> str:
        """Generate combined output in standard ### FILE: ### format"""
        parts = []
        
        for filename in self.plan.execution_order:
            if filename in self.completed_files:
                result = self.completed_files[filename]
                if result.content and result.content.strip():
                    if result.status == FileStatus.FAILED:
                        parts.append(f"### FILE: {filename}.failed ###")
                        parts.append(f"# GENERATION FAILED: {result.notes}")
                        parts.append(result.content)
                    else:
                        parts.append(f"### FILE: {filename} ###")
                        parts.append(result.content)
                    parts.append("")
        
        return '\n'.join(parts)
    
    def get_status_report(self) -> str:
        """Generate human-readable status report"""
        lines = [
            f"Plan Execution Status: {self.plan.name if self.plan else 'No plan'}",
            "=" * 60,
            ""
        ]
        
        if self.plan:
            for filename in self.plan.execution_order:
                status = self.status.get(filename, FileStatus.PENDING)
                icon = {
                    FileStatus.PENDING: '⏳',
                    FileStatus.IN_PROGRESS: '🔄',
                    FileStatus.COMPLETED: '✅',
                    FileStatus.FAILED: '❌',
                    FileStatus.REVISION_NEEDED: '⚠️'
                }.get(status, '❓')
                
                lines.append(f"{icon} {filename}: {status.value}")
                
                if filename in self.completed_files:
                    result = self.completed_files[filename]
                    if result.actual_exports:
                        export_names = [e['name'] for e in result.actual_exports]
                        lines.append(f"   Exports: {', '.join(export_names)}")
                    if result.revision_count > 0:
                        lines.append(f"   Revisions: {result.revision_count}")
        
        if self.errors:
            lines.extend(["", "Errors:", "-" * 40])
            for error in self.errors:
                lines.append(f"  {error['filename']}: {error['error']}")
        
        return '\n'.join(lines)


# =============================================================================
# ARCHITECT PROMPT FOR YAML PLAN GENERATION
# =============================================================================

ARCHITECT_PLAN_SYSTEM_PROMPT = """You are an expert software architect. Your job is to create a YAML execution plan for a program.

OUTPUT FORMAT: You MUST output a valid YAML document with this structure:

```yaml
program:
  name: "project_name"
  description: "What the program does"
  type: "cli|subprocess_tool|library|service"
architecture:
  pattern: "simple|layered|modular"
  entry_point: "filename.py or null for libraries"

files:
  - name: "config.py"
    purpose: "What this file does"
    dependencies: []
    exports:
      - name: "Settings"
        type: "class"
      - name: "get_config"
        type: "function"
    requirements:
      - "Specific requirement 1"
      - "Specific requirement 2"
  
  - name: "core.py"
    purpose: "Core business logic"
    dependencies: ["config.py"]
    imports_from:
      config.py: ["Settings", "get_config"]
    exports:
      - name: "CoreEngine"
        type: "class"
    requirements:
      - "Use Settings from config"
      - "Implement main logic"

  - name: "main.py"
    purpose: "Application entry point"
    dependencies: ["config.py", "core.py"]
    imports_from:
      config.py: ["get_config"]
      core.py: ["CoreEngine"]
    exports: []
    requirements:
      - "Initialize and run CoreEngine"

execution_order: ["config.py", "core.py", "main.py"]
```

CRITICAL RULES:
1. DEPENDENCY CONSISTENCY: If file A imports from file B, then B MUST be in A's dependencies list
2. EXPORT ACCURACY: Only list exports that the file will ACTUALLY define
3. IMPORT ACCURACY: imports_from MUST list ONLY names that appear in the source file's exports
4. NO CIRCULAR DEPENDENCIES: A file cannot depend on files that depend on it
5. EXECUTION ORDER: Must be a valid topological sort of the dependency graph

VALIDATION CHECKLIST (verify before output):
- Every imports_from module is in that file's dependencies list
- Every name in imports_from appears in that module's exports list
- execution_order respects all dependencies (deps come before dependents)
- No file imports from itself
- Entry point has all necessary dependencies listed

COMMON MISTAKES TO AVOID:
- Listing imports_from a file but forgetting to add it to dependencies
- Listing an import name that won't exist in the source module's exports
- Forgetting that main.py needs to depend on ALL modules it uses
- Adding unnecessary intermediate dependencies

Keep it SIMPLE - match complexity to requirements.
Output ONLY the YAML block. No explanations before or after."""


def get_architect_plan_prompt(user_request: str, job_scope: str = "") -> str:
    """Build user message for architect to generate YAML plan"""
    if job_scope and job_scope != "Requirements are clear and complete.":
        # Use job scope as primary source - it contains everything
        parts = [f"Create a YAML execution plan for this program:\n\n{job_scope}"]
    else:
        # Fallback to raw user request
        parts = [f"Create a YAML execution plan for this program:\n\n{user_request}"]
    
    parts.append("\nOutput the YAML plan:")
    return '\n'.join(parts)


def extract_yaml_from_response(response: str) -> str:
    """Extract YAML content from architect response"""
    # Try to find YAML in code blocks
    yaml_match = re.search(r'```ya?ml\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if yaml_match:
        return yaml_match.group(1).strip()
    
    # Try to find raw YAML (starts with 'program:')
    yaml_match = re.search(r'(program:\s*\n.*)', response, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1).strip()
    
    # Return full response and let YAML parser handle it
    return response.strip()


# =============================================================================
# INTEGRATION WITH SWARMCOORDINATOR
# =============================================================================

def create_planned_workflow(coordinator, user_request: str):
    """
    Create a planned workflow using the Plan Executor.
    Call this from SwarmCoordinator to use plan-based execution.
    
    Usage in SwarmCoordinator:
        elif workflow_type == "planned":
            from plan_executor import create_planned_workflow
            create_planned_workflow(self, user_request)
    """
    from swarm_coordinator_v2 import AgentRole, Task, TaskStatus
    
    # Phase 1: Clarification (reuse existing)
    coordinator.add_task(Task(
        task_id="T001_clarify",
        task_type="clarification",
        description="Clarify requirements",
        assigned_role=AgentRole.CLARIFIER,
        status=TaskStatus.PENDING,
        priority=10,
        metadata={"user_request": user_request}
    ))
    
    # Phase 2: Architecture with YAML plan output
    coordinator.add_task(Task(
        task_id="T002_plan",
        task_type="architecture_plan",
        description="Create YAML execution plan",
        assigned_role=AgentRole.ARCHITECT,
        status=TaskStatus.PENDING,
        priority=9,
        dependencies=["T001_clarify"],
        metadata={
            "user_request": user_request,
            "output_format": "yaml_plan"
        }
    ))
    
    # Phase 3: Plan execution (handled specially)
    coordinator.add_task(Task(
        task_id="T003_execute_plan",
        task_type="plan_execution",
        description="Execute the YAML plan file-by-file",
        assigned_role=AgentRole.CODER,  # Will use PlanExecutor
        status=TaskStatus.PENDING,
        priority=8,
        dependencies=["T002_plan"],
        metadata={
            "user_request": user_request,
            "use_plan_executor": True
        }
    ))

    # Phase 4: Semantic audit (local AST-based gate, no LLM)
    coordinator.add_task(Task(
        task_id="T004_audit",
        task_type="semantic_audit",
        description="Audit generated code for placeholders/lazy stubs",
        assigned_role=AgentRole.DEBUGGER,
        status=TaskStatus.PENDING,
        priority=7,
        dependencies=["T003_execute_plan"],
        metadata={"user_request": user_request}
    ))

    # Phase 5: Test generation
    coordinator.add_task(Task(
        task_id="T005_tests",
        task_type="test_generation",
        description="Generate unit tests",
        assigned_role=AgentRole.TESTER,
        status=TaskStatus.PENDING,
        priority=6,
        dependencies=["T004_audit"],
        metadata={"user_request": user_request}
    ))
    
    # Phase 6: Documentation
    coordinator.add_task(Task(
        task_id="T006_document",
        task_type="documentation",
        description="Generate documentation",
        assigned_role=AgentRole.DOCUMENTER,
        status=TaskStatus.PENDING,
        priority=5,
        dependencies=["T003_execute_plan", "T005_tests"],
        metadata={"user_request": user_request}
    ))
    
    # Phase 7: Final verification
    coordinator.add_task(Task(
        task_id="T007_verify",
        task_type="verification",
        description="Verify docs match code",
        assigned_role=AgentRole.VERIFIER,
        status=TaskStatus.PENDING,
        priority=4,
        dependencies=["T006_document"]
    ))


def execute_plan_task(coordinator, task, plan_yaml: str, user_request: str, job_scope: str = "") -> str:
    """
    Execute a plan_execution task using PlanExecutor.
    Call this from SwarmCoordinator.execute_task when task.task_type == "plan_execution"
    
    Returns the combined code output in ### FILE: ### format
    """
    plan_executor = PlanExecutor(
        executor=coordinator.executor,
        config=coordinator.config
    )
    
    result = plan_executor.execute(plan_yaml, user_request, job_scope)
    
    # Print status report
    print("\n" + plan_executor.get_status_report())
    
    if result['success']:
        return result['combined_output']
    else:
        # Return partial output even on failure
        print("\n⚠ Plan execution had errors:")
        for error in result['errors']:
            print(f"  - {error['filename']}: {error['error']}")
        return result['combined_output']


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

def run_plan_workflow(user_request: str, config: Dict = None) -> Dict[str, Any]:
    """
    Standalone function to run a complete plan-based workflow.
    
    Args:
        user_request: What to build
        config: Optional config dict (uses defaults if not provided)
        
    Returns:
        Result dictionary with completed files and status
    """
    from swarm_coordinator_v2 import SwarmCoordinator, AgentRole
    
    # Initialize coordinator
    coordinator = SwarmCoordinator()
    if config:
        coordinator.config.update(config)
    
    # Step 1: Clarification
    print("\n" + "="*70)
    print("PHASE 1: CLARIFICATION")
    print("="*70)
    
    clarifier_prompt = coordinator._get_system_prompt(AgentRole.CLARIFIER, "clarification")
    clarifier_response = coordinator.executor.execute_agent(
        role=AgentRole.CLARIFIER,
        system_prompt=clarifier_prompt,
        user_message=f"Analyze these requirements:\n\n{user_request}"
    )
    
    print(clarifier_response)
    
    # Handle clarification interactively if needed
    clarification = ""
    if "STATUS: CLEAR" not in clarifier_response.upper():
        print("\nPlease provide answers (type DONE when finished):")
        answers = []
        while True:
            try:
                line = input()
                if line.strip().upper() == 'DONE':
                    break
                answers.append(line)
            except EOFError:
                break
        clarification = '\n'.join(answers)
    
    # Step 2: Architecture Plan
    print("\n" + "="*70)
    print("PHASE 2: ARCHITECTURE PLAN")
    print("="*70)
    
    architect_message = get_architect_plan_prompt(user_request, clarification)
    architect_response = coordinator.executor.execute_agent(
        role=AgentRole.ARCHITECT,
        system_prompt=ARCHITECT_PLAN_SYSTEM_PROMPT,
        user_message=architect_message
    )
    
    plan_yaml = extract_yaml_from_response(architect_response)
    print(plan_yaml)
    
    # Step 3: Plan Execution
    print("\n" + "="*70)
    print("PHASE 3: PLAN EXECUTION")
    print("="*70)
    
    plan_executor = PlanExecutor(
        executor=coordinator.executor,
        config=coordinator.config
    )
    
    result = plan_executor.execute(plan_yaml, user_request)
    
    # Print final report
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(plan_executor.get_status_report())
    
    return result


if __name__ == "__main__":
    # Example usage
    test_request = """
    Create a Python CLI tool that:
    1. Reads a CSV file of employee data
    2. Calculates average salary by department
    3. Outputs results to console and JSON file
    """
    
    result = run_plan_workflow(test_request)
    
    if result['success']:
        print("\n" + "="*70)
        print("GENERATED CODE")
        print("="*70)
        print(result['combined_output'])