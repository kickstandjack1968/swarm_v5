#!/usr/bin/env python3
"""
Plan Executor v2 - With Full Plan Enforcement
==============================================

MAJOR CHANGES FROM v1:
1. YAML plan is the single source of truth - exports enforced, not suggested
2. Import statements are pre-generated from plan, not computed from actuals
3. Plan compliance verification gate before accepting any file
4. Structured requirements with mandatory export signatures
5. Auto-generation of __init__.py for subfolders
6. Dynamic integration checklists for entry points

Usage:
    from plan_executor_v2 import PlanExecutor, create_planned_workflow
    
    executor = PlanExecutor(agent_executor, config)
    result = executor.execute(plan_yaml, user_request)
"""

import sys
import os
import re
import ast
import time
import json
import yaml
import logging
import tempfile
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging for PlanExecutor."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("PlanExecutor")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"plan_executor_{timestamp}.log")
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # Symlink to latest
    latest_link = os.path.join(log_dir, "plan_executor_latest.log")
    try:
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(log_file), latest_link)
    except (OSError, NotImplementedError):
        pass
    
    return logger


# =============================================================================
# DATA CLASSES
# =============================================================================

class FileStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PLAN_MISMATCH = "plan_mismatch"


@dataclass
class ExportSpec:
    """Specification for an exported item."""
    name: str
    type: str  # "class", "function", "constant"
    methods: Dict[str, Dict] = field(default_factory=dict)  # For classes: method signatures


@dataclass
class FileSpec:
    """Specification for a single file in the plan."""
    name: str
    purpose: str
    dependencies: List[str] = field(default_factory=list)
    exports: List[ExportSpec] = field(default_factory=list)
    imports_from: Dict[str, List[str]] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class FileResult:
    """Result of generating a single file."""
    name: str
    content: str
    actual_exports: List[Dict[str, str]]
    status: FileStatus
    plan_compliance: Dict[str, Any] = field(default_factory=dict)
    verification: Optional[Dict[str, Any]] = None
    notes: str = ""
    revision_count: int = 0


@dataclass
class ProgramPlan:
    """Parsed program plan from YAML."""
    name: str
    description: str
    architecture_pattern: str
    entry_point: str
    files: List[FileSpec]
    execution_order: List[str]
    raw_yaml: str


# =============================================================================
# STUB DETECTION HELPERS
# =============================================================================

_SIMPLE_ACCESSOR_PREFIXES = ("get_", "is_", "has_", "to_", "as_")
_SIMPLE_ACCESSOR_NAMES = frozenset({
    "__repr__", "__str__", "__hash__", "__len__", "__bool__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__iter__", "__next__", "__contains__", "__getitem__",
    "__format__", "__int__", "__float__", "__index__",
    "name", "value", "key", "items", "keys", "values",
})


def _is_simple_accessor(name: str) -> bool:
    """Return True if the method name suggests a legitimately simple body."""
    if name in _SIMPLE_ACCESSOR_NAMES:
        return True
    if name.startswith(_SIMPLE_ACCESSOR_PREFIXES):
        return True
    return False


def _has_property_decorator(node) -> bool:
    """Return True if the function node has a @property decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "property":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in ("getter", "setter", "deleter"):
            return True
    return False


# =============================================================================
# PLAN EXECUTOR
# =============================================================================

class PlanExecutor:
    """
    Executes YAML plans with strict enforcement.
    
    The YAML plan is the single source of truth:
    - Exports must match the plan exactly
    - Imports are generated from the plan, not from actuals
    - Files that don't comply with the plan are rejected
    """
    
    def __init__(self, executor, config: Dict, log_dir: str = "logs", project_dir: str = ""):
        self.executor = executor
        self.config = config
        self.plan: Optional[ProgramPlan] = None
        self.completed_files: Dict[str, FileResult] = {}
        self.status: Dict[str, FileStatus] = {}
        self.errors: List[Dict[str, Any]] = []
        self.max_file_revisions = 5   # for structural issues
        self.max_issue_retries = 10   # per-method surgical retry cap
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir)
        self.job_scope = ""
        self.start_time: Optional[float] = None
        self.project_dir = project_dir
        self._file_counter = 0

    def _log_prompt(self, step: str, payload: Dict):
        """Save prompt payload to project's prompt_logs/ directory."""
        if not self.project_dir:
            return
        log_dir = os.path.join(self.project_dir, "prompt_logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{timestamp}_coder_{step}.json"

        log_entry = {
            "step": step,
            "agent": "coder",
            "timestamp": datetime.now().isoformat(),
        }
        for key in ("system_prompt", "user_message", "result", "file_name",
                     "compliance_issues", "revision_attempt"):
            if key in payload:
                log_entry[key] = payload[key]

        log_path = os.path.join(log_dir, filename)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)
            print(f"  [LOG] {filename}")
        except Exception as e:
            print(f"  [LOG] Warning: could not save prompt log: {e}")

    # =========================================================================
    # PLAN PARSING
    # =========================================================================
    
    def parse_plan(self, plan_yaml: str) -> ProgramPlan:
        """Parse YAML plan into structured ProgramPlan."""
        try:
            data = yaml.safe_load(plan_yaml)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML plan: {e}")
        
        program = data.get('program', {})
        if isinstance(program, str):
            program = {"name": program, "description": program, "type": "cli"}
        
        architecture = data.get('architecture', {})
        if isinstance(architecture, str):
            architecture = {"pattern": architecture, "entry_point": "main.py"}
        
        files = []
        for file_data in data.get('files', []):
            # Parse exports into ExportSpec objects
            exports = []
            for exp in file_data.get('exports', []):
                if isinstance(exp, str):
                    exports.append(ExportSpec(name=exp, type="unknown"))
                elif isinstance(exp, dict):
                    exports.append(ExportSpec(
                        name=exp.get('name', ''),
                        type=exp.get('type', 'unknown'),
                        methods=exp.get('methods', {})
                    ))
            
            # Parse dependencies
            deps = file_data.get('dependencies', [])
            if deps and isinstance(deps[0], dict):
                deps = [d['name'] if isinstance(d, dict) else d for d in deps]
            
            # Parse imports_from
            imports_from = file_data.get('imports_from', {})
            if not isinstance(imports_from, dict):
                imports_from = {}
            
            files.append(FileSpec(
                name=file_data['name'],
                purpose=file_data.get('purpose', ''),
                dependencies=deps,
                exports=exports,
                imports_from=imports_from,
                requirements=file_data.get('requirements', []),
                optional=file_data.get('optional', False)
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
    
    def _resolve_dependencies(self, files: List[FileSpec]) -> List[str]:
        """Topological sort of files based on dependencies."""
        file_deps = {f.name: f.dependencies for f in files}
        
        in_degree = defaultdict(int)
        for fname in file_deps:
            in_degree[fname]
        for fname, deps in file_deps.items():
            for dep in deps:
                if dep in file_deps:
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
            self.logger.warning("Circular dependency detected, using definition order")
            return [f.name for f in files]
        
        return result
    
    # =========================================================================
    # IMPORT GENERATION (FROM PLAN, NOT ACTUALS)
    # =========================================================================
    
    def _generate_import_block(self, file_spec: FileSpec) -> str:
        """
        Generate exact import statements from the PLAN.
        
        This is the key fix: imports come from the plan's imports_from field,
        NOT from what previous files actually exported.
        """
        if not file_spec.imports_from:
            return ""
        
        lines = []
        current_folder = file_spec.name.rsplit('/', 1)[0] if '/' in file_spec.name else ''
        
        for source_file, import_names in file_spec.imports_from.items():
            if not import_names:
                continue

            # Compute correct relative import path based on folder depth
            source_folder = source_file.rsplit('/', 1)[0] if '/' in source_file else ''
            source_base = source_file.rsplit('/', 1)[-1].replace('.py', '') if '/' in source_file else source_file.replace('.py', '')

            if source_folder == current_folder:
                # Same folder (both root, or both in same subfolder)
                dots = '.'
                import_stmt = f"from {dots}{source_base} import {', '.join(import_names)}"
            else:
                # Different folders — compute relative path
                # Split into path components
                current_parts = current_folder.split('/') if current_folder else []
                source_parts = source_folder.split('/') if source_folder else []

                # Find common prefix
                common = 0
                for a, b in zip(current_parts, source_parts):
                    if a == b:
                        common += 1
                    else:
                        break

                # Go up from current to common ancestor
                ups = len(current_parts) - common
                # Then go down to source
                down_parts = source_parts[common:]

                if ups == 0:
                    # Source is in a subfolder of current's folder
                    module_path = '.'.join(down_parts + [source_base])
                    import_stmt = f"from .{module_path} import {', '.join(import_names)}"
                else:
                    dots = '.' * (ups + 1)  # +1 because level=1 is same package
                    if down_parts:
                        module_path = '.'.join(down_parts + [source_base])
                        import_stmt = f"from {dots}{module_path} import {', '.join(import_names)}"
                    else:
                        import_stmt = f"from {dots}{source_base} import {', '.join(import_names)}"

            lines.append(import_stmt)
        
        return '\n'.join(lines)
    
    def _generate_import_documentation(self, file_spec: FileSpec) -> str:
        """
        Generate detailed import documentation for the coder prompt.
        Shows exactly what's available and how to import it.
        """
        if not file_spec.imports_from and not file_spec.dependencies:
            return "This file has no dependencies. Do NOT import from other project files."
        
        parts = []
        current_folder = file_spec.name.rsplit('/', 1)[0] if '/' in file_spec.name else ''
        
        parts.append("=" * 60)
        parts.append("AVAILABLE IMPORTS - COPY THESE EXACTLY")
        parts.append("=" * 60)
        parts.append("")
        
        # Generate import block
        import_block = self._generate_import_block(file_spec)
        if import_block:
            parts.append("COPY THIS IMPORT BLOCK TO YOUR FILE:")
            parts.append("-" * 40)
            parts.append(import_block)
            parts.append("-" * 40)
            parts.append("")
        
        # Document what each import provides
        parts.append("WHAT EACH IMPORT PROVIDES:")
        for source_file, import_names in file_spec.imports_from.items():
            source_spec = self._get_file_spec(source_file)
            if source_spec:
                parts.append(f"\nFrom {source_file}:")
                for exp in source_spec.exports:
                    if exp.name in import_names:
                        parts.append(f"  - {exp.name} ({exp.type})")
                        if exp.methods:
                            for method_name, method_info in exp.methods.items():
                                args = method_info.get('args', [])
                                parts.append(f"      .{method_name}({', '.join(args)})")
        
        # Add completed file signatures for reference
        parts.append("")
        parts.append("ACTUAL SIGNATURES FROM COMPLETED FILES:")
        for dep_name in file_spec.dependencies:
            if dep_name in self.completed_files:
                result = self.completed_files[dep_name]
                signatures = self._extract_signatures(result.content)
                if signatures:
                    parts.append(f"\n--- {dep_name} ---")
                    parts.append(signatures[:2000])
        
        return '\n'.join(parts)
    
    # =========================================================================
    # MANDATORY EXPORT FORMAT
    # =========================================================================
    
    def _generate_mandatory_exports(self, file_spec: FileSpec) -> str:
        """
        Generate mandatory export specification that will cause rejection if not met.
        """
        if not file_spec.exports:
            return ""
        
        parts = []
        parts.append("=" * 60)
        parts.append("MANDATORY EXPORTS - CODE REJECTED IF MISSING")
        parts.append("=" * 60)
        parts.append("")
        parts.append("Your code MUST define these EXACTLY as specified:")
        parts.append("")
        
        for exp in file_spec.exports:
            if exp.type == "class":
                parts.append(f"class {exp.name}:  # REQUIRED")
                if exp.methods:
                    for method_name, method_info in exp.methods.items():
                        args = method_info.get('args', ['self'])
                        returns = method_info.get('returns', '')
                        ret_hint = f" -> {returns}" if returns else ""
                        parts.append(f"    def {method_name}({', '.join(args)}){ret_hint}:  # REQUIRED")
                parts.append("")
            elif exp.type == "function":
                parts.append(f"def {exp.name}(...):  # REQUIRED")
            elif exp.type == "constant":
                parts.append(f"{exp.name} = ...  # REQUIRED")
        
        parts.append("")
        parts.append("WARNING: If ANY of the above are missing or renamed,")
        parts.append("your code will be REJECTED and you must regenerate.")
        parts.append("")
        
        return '\n'.join(parts)
    
    # =========================================================================
    # PLAN COMPLIANCE VERIFICATION
    # =========================================================================
    
    def _validate_exports_against_plan(
        self, 
        file_spec: FileSpec, 
        actual_exports: List[Dict[str, str]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that actual exports match the plan.
        
        Returns:
            (passed, list of issues)
        """
        if not file_spec.exports:
            return True, []
        
        issues = []
        actual_names = {e['name'] for e in actual_exports}
        
        for planned_export in file_spec.exports:
            if planned_export.name not in actual_names:
                issues.append(
                    f"MISSING EXPORT: Plan requires '{planned_export.name}' ({planned_export.type}) "
                    f"but it was not found in the generated code."
                )
        
        # Check for unexpected exports (warning only)
        planned_names = {e.name for e in file_spec.exports}
        for actual in actual_exports:
            if actual['name'] not in planned_names:
                # This is just informational, not a failure
                self.logger.debug(f"Extra export in {file_spec.name}: {actual['name']}")
        
        return len(issues) == 0, issues
    
    def _validate_syntax(self, content: str, filename: str) -> Tuple[bool, str]:
        """Validate Python syntax."""
        if not content or len(content.strip()) < 20:
            return False, "File content is empty or too short"

        # AST parse first — gives the most useful error with line numbers
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

        return True, ""
    
    def _validate_imports_against_plan(
        self, 
        content: str, 
        file_spec: FileSpec
    ) -> List[str]:
        """
        Validate that imports match what the plan says should be imported.
        """
        issues = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues
        
        # Build expected imports from plan
        expected_imports = {}  # module -> set of names
        for source_file, names in file_spec.imports_from.items():
            module = source_file.replace('.py', '').replace('/', '.')
            # Also handle relative import form
            base_module = source_file.replace('.py', '').split('/')[-1]
            expected_imports[module] = set(names)
            expected_imports[base_module] = set(names)
        
        # Check actual imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                
                module = node.module.lstrip('.')
                
                # Check if importing from a project file
                for expected_module, expected_names in expected_imports.items():
                    if module == expected_module or module.endswith('.' + expected_module):
                        for alias in node.names:
                            if alias.name not in expected_names and alias.name != '*':
                                issues.append(
                                    f"Importing '{alias.name}' from {module}, but plan only allows: "
                                    f"{', '.join(sorted(expected_names))}"
                                )
        
        return issues

    def _validate_import_paths_resolve(
        self,
        content: str,
        file_spec: FileSpec,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that relative imports in generated code point to files
        that actually exist in the plan. Catches `from ..config import X`
        when config.py is at root level (not reachable via ..).
        """
        issues = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues

        # Build set of all plan file paths (normalized, no .py)
        plan_files = set()
        all_files = context.get("all_files", {})
        for f in all_files:
            plan_files.add(f)
            plan_files.add(f.replace('.py', ''))
            # Also add base module name
            plan_files.add(f.replace('.py', '').split('/')[-1])

        # Determine this file's depth in the directory tree
        file_parts = file_spec.name.replace('\\', '/').split('/')
        file_depth = len(file_parts) - 1  # e.g. "core/engine.py" -> depth 1

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level is None or node.level == 0:
                continue

            # Relative import: check if the dot level makes sense
            if node.level > file_depth + 1:
                issues.append(
                    f"Relative import 'from {'.' * node.level}{node.module or ''}' "
                    f"goes {node.level} levels up, but '{file_spec.name}' is only "
                    f"{file_depth} level(s) deep — this import cannot resolve"
                )
                continue

            # Resolve the target module path
            if node.module:
                # Walk up from the file's directory
                parts = list(file_parts[:-1])  # directory parts
                for _ in range(node.level - 1):
                    if parts:
                        parts.pop()
                module_parts = node.module.split('.')
                target_path = '/'.join(parts + module_parts)

                # Check if target exists in plan files
                target_py = target_path + '.py'
                target_init = target_path + '/__init__.py'
                if (target_py not in plan_files
                        and target_path not in plan_files
                        and target_init not in plan_files
                        and target_path.split('/')[-1] not in plan_files):
                    issues.append(
                        f"Relative import 'from {'.' * node.level}{node.module}' "
                        f"resolves to '{target_py}' which is not in the plan"
                    )

        return issues

    def _verify_plan_compliance(
        self,
        file_spec: FileSpec,
        result: FileResult,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive plan compliance check.
        
        This is the main verification gate that enforces the plan.
        """
        compliance = {
            "passed": True,
            "issues": [],
            "warnings": []
        }
        
        # 1. Syntax check
        is_valid, syntax_error = self._validate_syntax(result.content, file_spec.name)
        if not is_valid:
            compliance["passed"] = False
            compliance["issues"].append(f"Syntax: {syntax_error}")
            return compliance
        
        # 2. Export compliance (CRITICAL)
        exports_ok, export_issues = self._validate_exports_against_plan(
            file_spec, result.actual_exports
        )
        if not exports_ok:
            compliance["passed"] = False
            compliance["issues"].extend(export_issues)
        
        # 3. Import compliance (CRITICAL — plan imports are mandatory)
        import_issues = self._validate_imports_against_plan(result.content, file_spec)
        if import_issues and file_spec.imports_from:
            compliance["passed"] = False
            compliance["issues"].extend(import_issues)

        # 3b. Validate import paths resolve to real plan files
        path_issues = self._validate_import_paths_resolve(result.content, file_spec, context)
        if path_issues:
            compliance["passed"] = False
            compliance["issues"].extend(path_issues)
        
        # 4. Placeholder detection
        placeholder_issues = self._detect_placeholders(result.content)
        if placeholder_issues:
            compliance["passed"] = False
            compliance["issues"].extend(placeholder_issues)
        
        # 5. Integration check (call signatures + phantom imports)
        if file_spec.dependencies:
            integration_warnings, integration_errors = self._check_integration(file_spec, result, context)
            if integration_warnings:
                compliance["warnings"].extend(integration_warnings)
            if integration_errors:
                compliance["passed"] = False
                compliance["issues"].extend(integration_errors)

        return compliance
    
    def _detect_placeholders(self, content: str) -> List[str]:
        """Detect lazy placeholder implementations including subtle stubs."""
        issues = []

        # --- Regex-based checks (pre-AST) ---

        if re.search(r'raise\s+NotImplementedError', content, re.MULTILINE):
            issues.append("Contains NotImplementedError")

        if re.search(r'#.*\bplaceholder\b', content, re.IGNORECASE | re.MULTILINE):
            issues.append("Contains 'placeholder' comment")
        if re.search(r'#.*\bsimulat', content, re.IGNORECASE | re.MULTILINE):
            issues.append("Contains 'simulate/simulated' comment")

        # --- AST-based detection ---

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues

        # Build map of class_name -> set of self.X attributes used in __init__
        class_init_attrs: Dict[str, Set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    attrs = set()
                    for sub in ast.walk(item):
                        if (isinstance(sub, ast.Attribute)
                                and isinstance(sub.value, ast.Name)
                                and sub.value.id == "self"):
                            attrs.add(sub.attr)
                    class_init_attrs[node.name] = attrs

        # Walk all classes and their methods
        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef):
                continue

            cls_self_attrs = class_init_attrs.get(cls_node.name, set())

            for node in cls_node.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                # Skip allowlisted names
                if _is_simple_accessor(node.name) or _has_property_decorator(node):
                    continue

                body = node.body
                real_body = [
                    n for n in body
                    if not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.Str)))
                ]

                arg_names = {a.arg for a in node.args.args if a.arg != "self"}
                is_method = any(a.arg == "self" for a in node.args.args)

                # --- Check 1: pass-only ---
                if len(real_body) == 1 and isinstance(real_body[0], ast.Pass):
                    issues.append(f"Stub method [{cls_node.name}.{node.name}]: body is only `pass`")
                    continue

                # --- Check 2: ellipsis-only ---
                if len(real_body) == 1 and isinstance(real_body[0], ast.Expr):
                    if isinstance(real_body[0].value, ast.Constant) and real_body[0].value.value is ...:
                        issues.append(f"Stub method [{cls_node.name}.{node.name}]: body is only `...`")
                        continue

                # --- Check 3: hardcoded dict return ignoring all params ---
                if len(real_body) == 1 and isinstance(real_body[0], ast.Return):
                    ret_val = real_body[0].value
                    if isinstance(ret_val, ast.Dict) and len(ret_val.keys) > 0 and arg_names:
                        source_seg = ast.get_source_segment(content, real_body[0]) or ""
                        if not any(arg in source_seg for arg in arg_names):
                            issues.append(
                                f"Stub method [{cls_node.name}.{node.name}]: "
                                f"returns hardcoded dict, ignoring parameters {arg_names}"
                            )
                            continue

                # For remaining checks, look at single-return bodies
                if len(real_body) != 1 or not isinstance(real_body[0], ast.Return):
                    # Multi-statement body — check resource-ignoring (Check 7) separately
                    pass
                else:
                    ret_val = real_body[0].value
                    if ret_val is None:
                        # `return None` or bare `return` — could be legit for setters
                        pass

                    # --- Check 4: echo stub (f-string that only formats params back) ---
                    elif isinstance(ret_val, ast.JoinedStr) and arg_names:
                        # Gather all Name references inside the f-string
                        fstring_names = set()
                        for val in ast.walk(ret_val):
                            if isinstance(val, ast.Name):
                                fstring_names.add(val.id)
                        # If the f-string only references params (no self, no function calls)
                        has_calls = any(isinstance(v, ast.Call) for v in ast.walk(ret_val))
                        uses_self = "self" in fstring_names
                        if not has_calls and not uses_self and fstring_names and fstring_names.issubset(arg_names):
                            issues.append(
                                f"Stub method [{cls_node.name}.{node.name}]: "
                                f"echo stub — f-string only formats parameters back"
                            )
                            continue

                    # --- Check 5: hardcoded string return ---
                    elif isinstance(ret_val, ast.Constant) and isinstance(ret_val.value, str):
                        if is_method and arg_names and not node.name.startswith("_"):
                            issues.append(
                                f"Stub method [{cls_node.name}.{node.name}]: "
                                f"returns hardcoded string \"{ret_val.value[:60]}\" "
                                f"— should compute from parameters"
                            )
                            continue

                    # --- Check 6: truncation stub (returns a slice of an arg) ---
                    elif isinstance(ret_val, ast.Subscript) and isinstance(ret_val.slice, ast.Slice):
                        slice_src = ast.get_source_segment(content, ret_val) or ""
                        # e.g. text[:200] or data[:100]
                        target_name = None
                        if isinstance(ret_val.value, ast.Name):
                            target_name = ret_val.value.id
                        if target_name and target_name in arg_names:
                            issues.append(
                                f"Stub method [{cls_node.name}.{node.name}]: "
                                f"truncation stub — returns `{slice_src}` instead of processing"
                            )
                            continue

                # --- Check 7: single-line complex (1 actual line, 2+ params, no self refs) ---
                # Only flag truly trivial bodies — skip try/except, if/else, for, with, while
                # (those are single AST nodes but contain real logic)
                if (len(real_body) == 1 and len(arg_names) >= 2 and is_method
                        and not node.name.startswith("__")
                        and not isinstance(real_body[0], (ast.Try, ast.If, ast.For,
                                                          ast.While, ast.With, ast.AsyncWith,
                                                          ast.AsyncFor))):
                    body_src = ast.get_source_segment(content, real_body[0]) or ""
                    if "self" not in body_src:
                        issues.append(
                            f"Stub method [{cls_node.name}.{node.name}]: "
                            f"single-statement body with {len(arg_names)} params, never references self"
                        )
                        continue

                # --- Check 8: resource-ignoring method ---
                _ACTION_PREFIXES = ("handle_", "process_", "generate_", "execute_",
                                    "run_", "send_", "fetch_", "build_", "create_",
                                    "analyze_", "compute_", "transform_", "parse_")
                if (is_method and any(node.name.startswith(p) for p in _ACTION_PREFIXES)
                        and cls_self_attrs):
                    # Check if the method body ever references self.X (direct use)
                    # OR calls self.other_method() (delegation — that method may use deps)
                    uses_self_attr = False
                    delegates_to_self_method = False
                    for sub in ast.walk(node):
                        if (isinstance(sub, ast.Attribute)
                                and isinstance(sub.value, ast.Name)
                                and sub.value.id == "self"):
                            if sub.attr in cls_self_attrs:
                                uses_self_attr = True
                                break
                            # self.some_method() call — counts as delegation
                            # (the called method likely uses class resources)
                            if isinstance(sub.ctx, ast.Load):
                                delegates_to_self_method = True
                    if not uses_self_attr and not delegates_to_self_method:
                        issues.append(
                            f"Stub method [{cls_node.name}.{node.name}]: "
                            f"action method never uses class dependencies "
                            f"(self has: {', '.join(sorted(list(cls_self_attrs)[:5]))}...)"
                        )

        # Also check top-level functions (not inside a class)
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_simple_accessor(node.name):
                continue

            body = node.body
            real_body = [
                n for n in body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.Str)))
            ]

            if len(real_body) == 1 and isinstance(real_body[0], ast.Pass):
                issues.append(f"Stub function [{node.name}]: body is only `pass`")
            elif len(real_body) == 1 and isinstance(real_body[0], ast.Expr):
                if isinstance(real_body[0].value, ast.Constant) and real_body[0].value.value is ...:
                    issues.append(f"Stub function [{node.name}]: body is only `...`")

        return issues

    # =========================================================================
    # SURGICAL REVISION HELPERS
    # =========================================================================

    @staticmethod
    def _extract_method_source(content: str, method_name: str, class_name: Optional[str] = None) -> Optional[str]:
        """Extract a single method's full source (including decorators) from file content."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        lines = content.splitlines()

        for node in ast.walk(tree):
            if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        start = item.lineno
                        if item.decorator_list:
                            start = item.decorator_list[0].lineno
                        end = item.end_lineno
                        return "\n".join(lines[start - 1:end])
            elif not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    # Only top-level
                    if any(node is child for child in ast.iter_child_nodes(tree)):
                        start = node.lineno
                        if node.decorator_list:
                            start = node.decorator_list[0].lineno
                        end = node.end_lineno
                        return "\n".join(lines[start - 1:end])

        return None

    @staticmethod
    def _replace_method_source(content: str, method_name: str, new_method: str, class_name: Optional[str] = None) -> str:
        """Splice a fixed method back into the file, replacing the original."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content

        lines = content.splitlines()
        target_node = None

        for node in ast.walk(tree):
            if class_name and isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        target_node = item
                        break
            elif not class_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    if any(node is child for child in ast.iter_child_nodes(tree)):
                        target_node = node
                        break

        if target_node is None:
            return content

        start = target_node.lineno
        if target_node.decorator_list:
            start = target_node.decorator_list[0].lineno
        end = target_node.end_lineno

        # Determine the original indentation
        original_first_line = lines[start - 1]
        original_indent = len(original_first_line) - len(original_first_line.lstrip())

        # Determine the replacement's indentation
        new_lines = new_method.splitlines()
        if new_lines:
            first_new = new_lines[0]
            new_indent = len(first_new) - len(first_new.lstrip())
        else:
            new_indent = 0

        indent_diff = original_indent - new_indent

        # Re-indent the replacement to match original
        adjusted_lines = []
        for line in new_lines:
            if line.strip() == "":
                adjusted_lines.append("")
            elif indent_diff > 0:
                adjusted_lines.append(" " * indent_diff + line)
            elif indent_diff < 0:
                # Remove leading spaces
                remove = abs(indent_diff)
                if line[:remove].strip() == "":
                    adjusted_lines.append(line[remove:])
                else:
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)

        # Splice
        result_lines = lines[:start - 1] + adjusted_lines + lines[end:]
        return "\n".join(result_lines)

    def _build_surgical_prompt(
        self,
        file_spec: FileSpec,
        method_name: str,
        method_source: str,
        issue: str,
        context: Dict[str, Any],
        class_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Build a focused prompt for fixing a single broken method."""
        sys_prompt = """You are an expert programmer fixing ONE specific method.

OUTPUT RULES:
1. Output ONLY the fixed method — no class wrapper, no imports, no markdown
2. Start directly with the def line (or decorator if applicable)
3. Preserve the exact method signature (name, parameters)
4. The method must be a complete, self-contained replacement

QUALITY RULES:
1. Implement REAL logic — no stubs, no placeholders, no hardcoded returns
2. Use self.X dependencies when they are available
3. Handle error cases appropriately
4. If an external service is unavailable, implement a working stdlib alternative"""

        # Gather class dependency info
        dep_info = ""
        if class_name:
            dep_info = self._get_class_dependency_info(file_spec, class_name, context)

        usr_msg = f"""FIX THIS METHOD in class {class_name or '(top-level)'}:

ISSUE: {issue}

BROKEN METHOD:
```python
{method_source}
```
"""
        if dep_info:
            usr_msg += f"""
CLASS CONTEXT (init + dependency signatures):
{dep_info}
"""

        usr_msg += f"""
FILE PURPOSE: {file_spec.purpose}

Output ONLY the fixed method (starting with def/async def). No class wrapper. No imports."""

        return sys_prompt, usr_msg

    def _get_class_dependency_info(
        self,
        file_spec: FileSpec,
        class_name: str,
        context: Dict[str, Any],
    ) -> str:
        """Extract ONLY the class __init__ and the specific dependencies it uses.

        Keeps context minimal — only what the method actually needs to reference.
        """
        parts = []

        # Get the file's current content
        own_content = ""
        if file_spec.name in self.completed_files:
            own_content = self.completed_files[file_spec.name].content

        if not own_content:
            return ""

        # Extract __init__ for the class — this shows what self.X is available
        init_src = self._extract_method_source(own_content, "__init__", class_name)
        if init_src:
            parts.append(f"# {class_name}.__init__:")
            parts.append(init_src)
            parts.append("")

        # Figure out which dependency types this class actually uses
        # by scanning __init__ for type annotations and assignments
        used_types: Set[str] = set()
        if init_src:
            try:
                # Parse just the init to find type references
                # Wrap in a dummy class so it parses
                dummy = f"class _Tmp:\n" + "\n".join(
                    "    " + line if not line.startswith(" ") else line
                    for line in init_src.splitlines()
                )
                try:
                    init_tree = ast.parse(dummy)
                except SyntaxError:
                    init_tree = None

                if init_tree:
                    for node in ast.walk(init_tree):
                        # Type annotations: self.x: SomeType = ...
                        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name):
                            used_types.add(node.annotation.id)
                        # Parameter annotations: def __init__(self, x: SomeType)
                        if isinstance(node, ast.arg) and node.annotation:
                            if isinstance(node.annotation, ast.Name):
                                used_types.add(node.annotation.id)
            except Exception:
                pass

        # Only include dependency signatures for types this class actually references
        for dep_name, dep_info in context.get("dependencies", {}).items():
            if dep_name not in file_spec.dependencies:
                continue
            summary = dep_info.get("summary", "")
            if not summary:
                continue
            # Only include if __init__ references a type from this dependency
            dep_exports = dep_info.get("exports", [])
            dep_names = {
                (e.get("name", "") if isinstance(e, dict) else str(e))
                for e in dep_exports
            }
            if used_types & dep_names:
                parts.append(f"# From {dep_name}:")
                parts.append(summary[:1500])
                parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _categorize_issues(issues: List[str]) -> Dict[str, List[str]]:
        """Split issues into structural vs implementation (method-specific).

        Issues containing [ClassName.method_name] are implementation issues
        that can be fixed surgically. Everything else is structural.
        """
        structural = []
        implementation = []

        method_pattern = re.compile(r'\[(\w+)\.(\w+)\]')

        for issue in issues:
            m = method_pattern.search(issue)
            if m:
                implementation.append(issue)
            else:
                structural.append(issue)

        return {"structural": structural, "implementation": implementation}

    # =========================================================================
    # SYNTAX AUTO-REPAIR
    # =========================================================================

    @staticmethod
    def _auto_fix_syntax(content: str) -> Tuple[str, bool]:
        """Try to fix bracket/paren syntax errors deterministically using tokenize.

        Walks the token stream, tracks a bracket stack, and inserts missing
        closers at the best-guess position.  Zero LLM cost.

        Returns (content, was_fixed).  If unfixable, returns the original.
        """
        import tokenize
        import io

        # Quick check — already valid?
        try:
            ast.parse(content)
            return content, False
        except SyntaxError:
            pass

        OPENERS = {"(": ")", "[": "]", "{": "}"}
        CLOSERS = {v: k for k, v in OPENERS.items()}

        current = content

        # Allow up to 10 rounds of single-bracket fixes
        for _round in range(10):
            try:
                ast.parse(current)
                return current, True  # fixed
            except SyntaxError as e:
                error_line = e.lineno or 0
                error_msg = e.msg or ""

            # Tokenize as far as possible (broken code will raise partway)
            stack: List[Tuple[str, int]] = []  # (bracket_char, line_no)
            tokens = []
            try:
                for tok in tokenize.generate_tokens(io.StringIO(current).readline):
                    tokens.append(tok)
                    if tok.type == tokenize.OP:
                        if tok.string in OPENERS:
                            stack.append((tok.string, tok.start[0]))
                        elif tok.string in CLOSERS:
                            if stack and stack[-1][0] == CLOSERS[tok.string]:
                                stack.pop()
            except tokenize.TokenError:
                pass  # expected — file is broken

            if not stack:
                break  # no unclosed brackets detected, can't fix deterministically

            lines = current.splitlines()

            # Fix the deepest (most recent) unclosed bracket first
            unclosed_char, unclosed_line = stack[-1]
            closer = OPENERS[unclosed_char]

            # Strategy: insert closer before the error line (or at the
            # end of the block that contains the opener).
            # Find the right indentation — match the opener's line indent.
            if 0 < unclosed_line <= len(lines):
                opener_line_text = lines[unclosed_line - 1]
                indent = len(opener_line_text) - len(opener_line_text.lstrip())
            else:
                indent = 0

            # Insert before the error line if possible, else after the opener line
            insert_idx = min(error_line - 1, len(lines)) if error_line > 0 else unclosed_line
            # Clamp
            insert_idx = max(0, min(insert_idx, len(lines)))

            lines.insert(insert_idx, " " * indent + closer)
            current = "\n".join(lines)

        # Final check
        try:
            ast.parse(current)
            return current, True
        except SyntaxError:
            return content, False  # give up, return original

    def _surgical_syntax_fix(
        self,
        content: str,
        syntax_error: str,
        file_spec: "FileSpec",
    ) -> Optional[str]:
        """Extract ~40 lines around a syntax error, ask a fast model to fix it,
        splice the result back.  Cheap fallback when deterministic fix fails.

        Returns the fixed content, or None if it couldn't be fixed.
        """
        from swarm_coordinator_v2 import AgentRole

        # Parse line number from error string: "Syntax error at line 208: ..."
        m = re.match(r"Syntax error at line (\d+): (.+)", syntax_error)
        if not m:
            return None
        error_line = int(m.group(1))
        error_msg = m.group(2)

        lines = content.splitlines()
        total = len(lines)

        # Extract a window: 20 lines before, 20 lines after
        win_start = max(0, error_line - 21)
        win_end = min(total, error_line + 20)
        window_lines = lines[win_start:win_end]

        # Add line numbers for the model's reference
        numbered = []
        for i, line in enumerate(window_lines, start=win_start + 1):
            numbered.append(f"{i:4d} | {line}")
        window_text = "\n".join(numbered)

        sys_prompt = """You are a syntax-repair specialist. You receive a code fragment with a syntax error and must fix it.

OUTPUT RULES:
1. Output ONLY the fixed code lines — same line range, same indentation
2. Do NOT add line numbers — output raw Python code only
3. Do NOT add markdown code blocks
4. Do NOT add explanations
5. Preserve all logic and variable names — only fix the syntax"""

        usr_msg = f"""SYNTAX ERROR at line {error_line}: {error_msg}

CODE (lines {win_start + 1}-{win_end}):
{window_text}

Output the fixed version of these {len(window_lines)} lines. Raw Python only, no line numbers:"""

        try:
            surg_step = f"PLAN_FILE_{self._file_counter}_SYNTAX_FIX"
            self._log_prompt(surg_step, {
                "system_prompt": sys_prompt,
                "user_message": usr_msg,
                "file_name": file_spec.name,
            })

            response = self.executor.execute_agent(
                role=AgentRole.FALLBACK_CODER,
                system_prompt=sys_prompt,
                user_message=usr_msg,
            )

            self._log_prompt(f"{surg_step}_RESULT", {
                "file_name": file_spec.name,
                "result": response[:4000] if len(response) > 4000 else response,
            })

            # Clean response — strip markdown fences
            fixed_text = re.sub(r"```python\s*", "", response)
            fixed_text = re.sub(r"```\s*", "", fixed_text).strip()

            # Splice back
            fixed_lines = fixed_text.splitlines()
            patched_lines = lines[:win_start] + fixed_lines + lines[win_end:]
            patched = "\n".join(patched_lines)

            # Verify it parses
            try:
                ast.parse(patched)
                self.logger.info(
                    f"Surgical syntax fix succeeded for {file_spec.name} "
                    f"(error was line {error_line})"
                )
                return patched
            except SyntaxError:
                self.logger.warning(
                    f"Surgical syntax fix did not resolve error for {file_spec.name}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Surgical syntax fix failed: {e}")
            return None

    def _check_integration(
        self,
        file_spec: FileSpec,
        result: FileResult,
        context: Dict[str, Any]
    ) -> tuple:
        """
        Check cross-file integration: constructor arg counts, phantom imports.

        Returns (warnings: List[str], errors: List[str]).
        Errors trigger revision; warnings are informational.
        """
        warnings = []
        errors = []

        try:
            tree = ast.parse(result.content)
        except SyntaxError:
            return warnings, errors

        # --- 1. Build constructor signature map from completed dependencies ---
        # Maps class_name -> number of required __init__ args (excluding self)
        constructor_args = {}
        for dep_name, dep_info in context.get('dependencies', {}).items():
            dep_content = dep_info.get('content', '')
            if not dep_content:
                continue
            try:
                dep_tree = ast.parse(dep_content)
            except SyntaxError:
                continue

            for node in ast.walk(dep_tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == '__init__':
                            args = item.args
                            all_args = [a.arg for a in args.args if a.arg != 'self']
                            num_defaults = len(args.defaults)
                            required = len(all_args) - num_defaults
                            total = len(all_args)
                            constructor_args[node.name] = {
                                'required': required,
                                'total': total,
                                'arg_names': all_args,
                                'source': dep_name
                            }

        # --- 2. Check constructor calls in current file ---
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                class_name = node.func.id
                if class_name in constructor_args:
                    sig = constructor_args[class_name]
                    provided = len(node.args) + len(node.keywords)
                    if provided < sig['required']:
                        errors.append(
                            f"Constructor mismatch: {class_name}() called with {provided} args "
                            f"but requires {sig['required']} ({', '.join(sig['arg_names'][:sig['required']])}). "
                            f"See {sig['source']} for the actual __init__ signature."
                        )
                    elif provided > sig['total']:
                        errors.append(
                            f"Constructor mismatch: {class_name}() called with {provided} args "
                            f"but accepts at most {sig['total']} ({', '.join(sig['arg_names'])}). "
                            f"See {sig['source']} for the actual __init__ signature."
                        )

        # --- 3. Detect phantom imports (imports from modules not in the plan) ---
        plan_file_names = set()
        if self.plan:
            for f in self.plan.files:
                plan_file_names.add(f.name)
                # Also add the module path variants
                # e.g. "core/ingestion.py" -> modules "core.ingestion", "core", "ingestion"
                module_path = f.name.replace('.py', '').replace('/', '.')
                plan_file_names.add(module_path)
                # Add parent packages
                parts = module_path.split('.')
                for i in range(len(parts)):
                    plan_file_names.add('.'.join(parts[:i+1]))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                # Skip stdlib and third-party imports (no dots = likely stdlib/third-party)
                # Focus on relative imports (level > 0) that reference project files
                if node.level and node.level > 0:
                    # This is a relative import like "from ..event_bus import X"
                    # Reconstruct what file this would reference
                    current_dir = file_spec.name.rsplit('/', 1)[0] if '/' in file_spec.name else ''
                    # Go up 'level' directories
                    parts = current_dir.split('/') if current_dir else []
                    up = node.level - 1  # level=1 is same package, level=2 is parent
                    if up > 0:
                        parts = parts[:-up] if len(parts) >= up else []
                    # Build target module path
                    if parts:
                        target = '/'.join(parts) + '/' + module.replace('.', '/') + '.py'
                    else:
                        target = module.replace('.', '/') + '.py'

                    # Check if this module exists in the plan
                    target_as_module = module
                    found = False
                    for plan_file in (self.plan.files if self.plan else []):
                        pf_module = plan_file.name.replace('.py', '').rsplit('/', 1)[-1]
                        pf_full = plan_file.name.replace('.py', '').replace('/', '.')
                        if target_as_module == pf_module or target_as_module == pf_full:
                            found = True
                            break
                        if target == plan_file.name:
                            found = True
                            break
                        # Check __init__.py for package imports
                        if plan_file.name.endswith('__init__.py'):
                            pkg = plan_file.name.replace('/__init__.py', '').replace('/', '.')
                            if target_as_module == pkg:
                                found = True
                                break

                    # Also check if it's a config or standard relative import
                    if not found and target_as_module not in ('config', '__init__'):
                        imported_names = [alias.name for alias in node.names] if node.names else []
                        errors.append(
                            f"Phantom import: 'from {'.' * node.level}{module} import {', '.join(imported_names)}' — "
                            f"module '{module}' does not exist in the project plan. "
                            f"Only import from files in the plan: {', '.join(f.name for f in (self.plan.files if self.plan else []))}"
                        )

        # --- 4. Extract class attributes from dependencies ---
        # Maps class_name -> set of known attribute names
        # Sources: dataclass fields, __init__ self.X assignments, class-level annotations
        class_attributes = {}
        for dep_name, dep_info in context.get('dependencies', {}).items():
            dep_content = dep_info.get('content', '')
            if not dep_content:
                continue
            try:
                dep_tree = ast.parse(dep_content)
            except SyntaxError:
                continue

            for node in ast.walk(dep_tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                attrs = set()

                # Class-level annotated assignments (e.g. dataclass fields, class vars)
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        attrs.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                attrs.add(t.id)

                # __init__ / __post_init__ / __new__ self.X assignments
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in ('__init__', '__post_init__', '__new__'):
                        for sub_node in ast.walk(item):
                            if (isinstance(sub_node, ast.Assign)):
                                for t in sub_node.targets:
                                    if (isinstance(t, ast.Attribute)
                                            and isinstance(t.value, ast.Name)
                                            and t.value.id == 'self'):
                                        attrs.add(t.attr)
                            elif isinstance(sub_node, ast.AnnAssign):
                                if (isinstance(sub_node.target, ast.Attribute)
                                        and isinstance(sub_node.target.value, ast.Name)
                                        and sub_node.target.value.id == 'self'):
                                    attrs.add(sub_node.target.attr)

                # Include all methods as accessible attributes (public AND private)
                # Downstream files may legitimately call private methods on dependencies
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Skip dunder methods except __init__, __call__, __getitem__, __len__, __iter__
                        if item.name.startswith('__') and item.name.endswith('__'):
                            if item.name not in ('__init__', '__call__', '__getitem__',
                                                  '__len__', '__iter__', '__contains__',
                                                  '__enter__', '__exit__', '__str__', '__repr__'):
                                continue
                        attrs.add(item.name)

                if attrs:
                    class_attributes[node.name] = attrs

        # --- 5. Validate constructor keyword args ---
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                class_name = node.func.id
                if class_name in constructor_args and node.keywords:
                    known_params = set(constructor_args[class_name]['arg_names'])
                    for kw in node.keywords:
                        if kw.arg and kw.arg not in known_params:
                            # Also check dataclass fields (they can be used as keyword args)
                            known_fields = class_attributes.get(class_name, set())
                            if kw.arg not in known_fields:
                                errors.append(
                                    f"Unknown keyword arg: {class_name}({kw.arg}=...) — "
                                    f"'{kw.arg}' is not a parameter of __init__ "
                                    f"(valid: {', '.join(sorted(known_params))}). "
                                    f"See {constructor_args[class_name]['source']}."
                                )

        # --- 6. Validate attribute access on dependency objects ---
        # Map self.X -> type name via __init__ parameter annotations
        self_attr_types = {}  # Maps attribute name -> class type name
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    # Build param -> type mapping from annotations
                    param_types = {}
                    for arg in item.args.args:
                        if arg.arg == 'self':
                            continue
                        if arg.annotation:
                            type_name = None
                            if isinstance(arg.annotation, ast.Name):
                                type_name = arg.annotation.id
                            elif isinstance(arg.annotation, ast.Attribute):
                                type_name = arg.annotation.attr
                            if type_name:
                                param_types[arg.arg] = type_name

                    # Find self.X = param assignments to map self.X -> type
                    for sub_node in ast.walk(item):
                        if isinstance(sub_node, ast.Assign):
                            for t in sub_node.targets:
                                if (isinstance(t, ast.Attribute)
                                        and isinstance(t.value, ast.Name)
                                        and t.value.id == 'self'):
                                    # Check if RHS is a known-typed parameter
                                    if isinstance(sub_node.value, ast.Name):
                                        rhs = sub_node.value.id
                                        if rhs in param_types:
                                            self_attr_types[t.attr] = param_types[rhs]

        # Now check all self.X.Y accesses where X maps to a known type
        seen_attr_errors = set()  # Deduplicate: (attr_name, accessed)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == 'self'):
                attr_name = node.value.attr  # X in self.X.Y
                accessed = node.attr          # Y in self.X.Y
                if attr_name in self_attr_types:
                    type_name = self_attr_types[attr_name]
                    if type_name in class_attributes:
                        known_attrs = class_attributes[type_name]
                        error_key = (attr_name, accessed)
                        if accessed not in known_attrs and error_key not in seen_attr_errors:
                            seen_attr_errors.add(error_key)
                            errors.append(
                                f"Attribute mismatch: self.{attr_name}.{accessed} — "
                                f"'{accessed}' not found in {type_name} "
                                f"(available: {', '.join(sorted(known_attrs)[:10])}). "
                                f"Check the {type_name} class definition in its source file."
                            )

        return warnings, errors

    # =========================================================================
    # FILE GENERATION
    # =========================================================================
    
    def _get_file_spec(self, filename: str) -> Optional[FileSpec]:
        """Get file specification from plan."""
        if not self.plan:
            return None
        for f in self.plan.files:
            if f.name == filename:
                return f
        return None
    
    def _build_context_for_file(self, file_spec: FileSpec) -> Dict[str, Any]:
        """Build context from completed dependencies."""
        context = {
            'program': {
                'name': self.plan.name,
                'description': self.plan.description,
                'pattern': self.plan.architecture_pattern,
                'entry_point': self.plan.entry_point
            },
            'all_files': [f.name for f in self.plan.files],
            'dependencies': {},
            'job_scope': self.job_scope
        }
        
        # Direct dependencies get full content
        for dep in file_spec.dependencies:
            if dep in self.completed_files:
                result = self.completed_files[dep]
                context['dependencies'][dep] = {
                    'exports': result.actual_exports,
                    'content': result.content,
                    'summary': self._extract_signatures(result.content)
                }
        
        # Transitive dependencies get summaries only
        for fname, result in self.completed_files.items():
            if fname not in context['dependencies']:
                context['dependencies'][fname] = {
                    'exports': result.actual_exports,
                    'summary': self._extract_signatures(result.content)
                }
        
        return context
    
    def _get_coder_system_prompt(self, is_entry_point: bool = False) -> str:
        """Get system prompt for coder agent."""
        
        base_prompt = """You are an expert programmer generating code as part of a STRUCTURED PLAN.

CRITICAL: This is PLAN-DRIVEN development. The plan specifies EXACTLY what you must create.

YOUR CONTRACT:
1. You MUST export the EXACT names specified in MANDATORY EXPORTS
2. You MUST use the EXACT import statements provided
3. You MUST implement ALL requirements listed
4. Your code will be REJECTED if it doesn't comply with the plan

OUTPUT RULES:
1. Output ONLY valid, executable Python code
2. Start directly with docstring or imports
3. NO markdown code blocks (```)
4. NO explanations before or after the code

QUALITY RULES:
1. Code must be immediately executable
2. Handle realistic error cases
3. Add docstrings for public interfaces
4. ALL public methods MUST have return type annotations (-> str, -> dict, -> list, -> None, etc.)
   This is CRITICAL — downstream files depend on knowing what your methods return.
   BAD:  def get_summary(self):
   GOOD: def get_summary(self) -> str:
   BAD:  def discover_logs(self):
   GOOD: def discover_logs(self) -> list[dict]:

INTEGRATION RULES (VIOLATIONS CAUSE REJECTION):
1. When calling a class constructor, match the EXACT arguments from ACTUAL SIGNATURES section
2. ONLY import from files listed in the plan — do NOT invent modules that don't exist
3. When accessing attributes/methods on imported objects, only use ones visible in ACTUAL SIGNATURES
4. Constructor keyword args must match the actual __init__ parameter names — do NOT invent field names
5. When accessing self.X.Y, ensure Y is an actual attribute/method of X's class (check ACTUAL SIGNATURES)

ANTI-STUB RULES (VIOLATIONS CAUSE REJECTION):
1. NO NotImplementedError — implement real logic instead
2. NO methods whose ONLY statement is pass — implement real logic
3. NO methods whose ONLY statement is ellipsis (...) — implement real logic
4. NO "placeholder" or "simulate" comments — these indicate unfinished code
5. NO hardcoded return values that ignore all function parameters
6. NO echo stubs — f-strings that just format parameters back (e.g. return f"Response to: {text}")
7. NO hardcoded string returns for methods that should compute a result (e.g. return "Bot is running.")
8. NO truncation stubs — returning a slice of input instead of processing it (e.g. return text[:200])
9. Methods with access to self.X dependencies MUST actually use them — do not ignore class resources
10. If a feature needs an external service/model not available, implement a working alternative using stdlib
   Example: Instead of "# would use CLIP model", implement keyword-based classification with regex
11. If you cannot fully implement something, implement the BEST working version possible with available tools

CODE SAFETY RULES (prevent syntax disasters):
1. NEVER embed Python code inside f-strings (e.g. for subprocess -c or exec calls)
   BAD:  subprocess.run([sys.executable, "-c", f"import ast\\nerrors.append({{'key': f'value {{x}}'}})"])
   GOOD: Write the helper script to a temp file using a regular triple-quoted string, then execute the file
2. For subprocess calls that run Python code, ALWAYS write to a temp file first:
   script = textwrap.dedent(\"\"\"\\
       import ast
       import json
       # ... your code here, no f-string nesting issues ...
   \"\"\")
   script_path = Path(temp_dir) / "helper.py"
   script_path.write_text(script)
   subprocess.run([sys.executable, str(script_path)], ...)
3. Pass dynamic values to subprocess scripts via command-line args, environment variables, or temp JSON files — NOT via string interpolation
"""

        if is_entry_point:
            base_prompt += """

ENTRY POINT SPECIFIC RULES:
This file orchestrates all other modules. Extra care required:

1. Import statements are provided - COPY THEM EXACTLY
2. Every function call must match the ACTUAL signature from dependencies
3. Check the ACTUAL SIGNATURES section carefully before calling any function
4. If a function requires arguments, you MUST provide them
5. Use proper error handling for missing data/files

COMMON ENTRY POINT MISTAKES:
- Calling func() when signature is func(arg1, arg2)
- Passing wrong argument types
- Not handling None/missing data cases
- Forgetting to initialize required objects
"""
        
        return base_prompt
    
    def _build_coder_message(
        self,
        file_spec: FileSpec,
        user_request: str,
        context: Dict[str, Any]
    ) -> str:
        """Build the complete prompt for file generation."""
        
        parts = []
        
        # Program overview
        parts.append(f"""PROGRAM: {context['program']['name']}
{'-' * 60}
Description: {context['program']['description']}
Pattern: {context['program']['pattern']}
Entry Point: {context['program']['entry_point']}
All Files: {', '.join(context['all_files'])}
""")
        
        # Job scope / requirements
        parts.append(f"""
JOB SCOPE
{'=' * 60}
{context.get('job_scope', user_request)}
""")
        
        # Current file info
        parts.append(f"""
FILE TO GENERATE: {file_spec.name}
{'=' * 60}
Purpose: {file_spec.purpose}
""")
        
        # Requirements as checklist
        if file_spec.requirements:
            parts.append("\nREQUIREMENTS CHECKLIST (implement ALL):")
            for i, req in enumerate(file_spec.requirements, 1):
                parts.append(f"  [{i}] {req}")
        
        # Mandatory exports
        mandatory_exports = self._generate_mandatory_exports(file_spec)
        if mandatory_exports:
            parts.append("")
            parts.append(mandatory_exports)
        
        # Import documentation
        import_docs = self._generate_import_documentation(file_spec)
        if import_docs:
            parts.append("")
            parts.append(import_docs)
        
        # Integration checklist for entry point
        if file_spec.name == self.plan.entry_point or file_spec.name == 'main.py':
            parts.append(self._generate_integration_checklist(file_spec, context))
        
        # Final instruction
        parts.append(f"""
{'=' * 60}
GENERATE {file_spec.name} NOW
{'=' * 60}
Output ONLY the Python code. No explanations.
Remember: Your code will be REJECTED if mandatory exports are missing.
""")
        
        return '\n'.join(parts)
    
    def _generate_integration_checklist(
        self,
        file_spec: FileSpec,
        context: Dict[str, Any]
    ) -> str:
        """Generate dynamic integration checklist for entry point."""
        
        parts = []
        parts.append("")
        parts.append("=" * 60)
        parts.append("INTEGRATION CHECKLIST - VERIFY BEFORE OUTPUT")
        parts.append("=" * 60)
        parts.append("")
        
        for dep_name, dep_info in context.get('dependencies', {}).items():
            if dep_name not in file_spec.dependencies:
                continue
            
            exports = dep_info.get('exports', [])
            if not exports:
                continue
            
            parts.append(f"From {dep_name}:")
            for exp in exports:
                exp_name = exp.get('name', exp) if isinstance(exp, dict) else exp
                exp_type = exp.get('type', 'unknown') if isinstance(exp, dict) else 'unknown'
                parts.append(f"  [ ] Did you import {exp_name}?")
                
                if exp_type == 'class':
                    # Try to find __init__ signature
                    content = dep_info.get('content', '')
                    if content:
                        try:
                            from agent_base import extract_class_info
                            classes = extract_class_info(content)
                            if exp_name in classes:
                                init_info = classes[exp_name].get('methods', {}).get('__init__', {})
                                args = init_info.get('args', ['self'])
                                if len(args) > 1:
                                    parts.append(f"      [ ] {exp_name}() requires: {', '.join(args[1:])}")
                        except:
                            pass
            parts.append("")
        
        return '\n'.join(parts)
    
    def _extract_exports(self, content: str) -> List[Dict[str, str]]:
        """Extract actual exports from generated code."""
        try:
            from agent_base import extract_exports_from_code
            return extract_exports_from_code(content)
        except ImportError:
            # Fallback implementation
            exports = []
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('class ') and not line.startswith(' '):
                    match = re.match(r'class\s+(\w+)', stripped)
                    if match and not match.group(1).startswith('_'):
                        exports.append({'name': match.group(1), 'type': 'class'})
                elif stripped.startswith('def ') and not line.startswith(' '):
                    match = re.match(r'def\s+(\w+)', stripped)
                    if match and not match.group(1).startswith('_'):
                        exports.append({'name': match.group(1), 'type': 'function'})
            return exports
    
    def _extract_signatures(self, content: str) -> str:
        """Extract function/class signatures for context."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content[:500]
        
        lines = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                try:
                    lines.append(ast.unparse(node))
                except:
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
                lines.append(f"\nclass {node.name}{base_str}:")

                # Class-level fields (dataclass fields, annotated assignments)
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        try:
                            ann_str = ast.unparse(item.annotation)
                            if item.value is not None:
                                val_str = ast.unparse(item.value)
                                lines.append(f"    {item.target.id}: {ann_str} = {val_str}")
                            else:
                                lines.append(f"    {item.target.id}: {ann_str}")
                        except:
                            lines.append(f"    {item.target.id}: ...")

                # Methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        try:
                            args_str = ast.unparse(item.args)
                        except:
                            args_str = "..."
                        ret_str = self._infer_return_type(item)
                        ret_hint = f" -> {ret_str}" if ret_str else ""
                        lines.append(f"    def {item.name}({args_str}){ret_hint}: ...")

            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    try:
                        args_str = ast.unparse(node.args)
                    except:
                        args_str = "..."
                    ret_str = self._infer_return_type(node)
                    ret_hint = f" -> {ret_str}" if ret_str else ""
                    lines.append(f"\ndef {node.name}({args_str}){ret_hint}: ...")
        
        return '\n'.join(lines)

    @staticmethod
    def _infer_return_type(func_node) -> str:
        """
        Get return type for a function — from annotation first, then inferred from code.
        Returns a type string like 'str', 'dict', 'list[dict]', 'None', or '' if unknown.
        """
        # 1. Use explicit annotation if present
        if func_node.returns:
            try:
                return ast.unparse(func_node.returns)
            except:
                pass

        # 2. Infer from return statements in the body
        # Skip __init__ (always returns None implicitly)
        if func_node.name == '__init__':
            return ''

        returns = []
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                returns.append(node.value)

        if not returns:
            # No return statements with values — void method
            # Only flag this for non-dunder methods (dunders like __init__ are expected)
            if not func_node.name.startswith('__'):
                return 'None'
            return ''

        # Infer from the last return statement (most representative)
        last_ret = returns[-1]

        # Dict literal or dict()
        if isinstance(last_ret, ast.Dict):
            return 'dict'
        if isinstance(last_ret, ast.Call) and isinstance(last_ret.func, ast.Name):
            if last_ret.func.id == 'dict':
                return 'dict'

        # List literal or list comprehension
        if isinstance(last_ret, ast.List):
            return 'list'
        if isinstance(last_ret, ast.ListComp):
            return 'list'
        if isinstance(last_ret, ast.Call) and isinstance(last_ret.func, ast.Name):
            if last_ret.func.id == 'list':
                return 'list'
            if last_ret.func.id == 'sorted':
                return 'list'

        # String operations
        if isinstance(last_ret, ast.JoinedStr):  # f-string
            return 'str'
        if isinstance(last_ret, ast.Constant) and isinstance(last_ret.value, str):
            return 'str'
        # str.join(), str.format(), etc.
        if (isinstance(last_ret, ast.Call)
                and isinstance(last_ret.func, ast.Attribute)
                and last_ret.func.attr in ('join', 'format', 'strip', 'lower', 'upper', 'replace')):
            return 'str'

        # Bool
        if isinstance(last_ret, ast.Constant) and isinstance(last_ret.value, bool):
            return 'bool'

        # Int/float
        if isinstance(last_ret, ast.Constant) and isinstance(last_ret.value, int):
            return 'int'
        if isinstance(last_ret, ast.Constant) and isinstance(last_ret.value, float):
            return 'float'

        # Tuple
        if isinstance(last_ret, ast.Tuple):
            return 'tuple'

        # Set
        if isinstance(last_ret, ast.Set) or isinstance(last_ret, ast.SetComp):
            return 'set'

        # json.dumps -> str
        if (isinstance(last_ret, ast.Call)
                and isinstance(last_ret.func, ast.Attribute)
                and last_ret.func.attr == 'dumps'):
            return 'str'

        # json.loads -> dict or list
        if (isinstance(last_ret, ast.Call)
                and isinstance(last_ret.func, ast.Attribute)
                and last_ret.func.attr == 'loads'):
            return 'dict | list'

        # len() -> int
        if (isinstance(last_ret, ast.Call)
                and isinstance(last_ret.func, ast.Name)
                and last_ret.func.id == 'len'):
            return 'int'

        return ''

    def _clean_code_output(self, code: str, expected_filename: str) -> str:
        """Clean LLM output to extract just the code."""
        
        # Handle file markers
        if '### FILE:' in code:
            pattern = rf'###\s*FILE:\s*{re.escape(expected_filename)}\s*###\s*(.*?)(?=###\s*FILE:|$)'
            match = re.search(pattern, code, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1)
        
        # Remove markdown
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Find code start
        lines = code.split('\n')
        cleaned = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
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
    
    def _generate_file(
        self,
        file_spec: FileSpec,
        user_request: str,
        context: Dict[str, Any]
    ) -> FileResult:
        """Generate a single file with plan compliance enforcement."""
        
        from swarm_coordinator_v2 import AgentRole
        
        is_entry_point = (file_spec.name == self.plan.entry_point or 
                         file_spec.name == 'main.py')
        
        system_prompt = self._get_coder_system_prompt(is_entry_point)
        user_message = self._build_coder_message(file_spec, user_request, context)

        self._file_counter += 1
        file_n = self._file_counter

        max_attempts = 2
        last_error = ""

        for attempt in range(max_attempts):
            try:
                # Add error context on retry
                if attempt > 0 and last_error:
                    retry_message = f"""PREVIOUS ATTEMPT FAILED:
{last_error}

You MUST fix this issue. The plan requires specific exports.

{user_message}"""
                    step = f"PLAN_FILE_{file_n}_RETRY_{attempt}"
                    self._log_prompt(step, {
                        "system_prompt": system_prompt,
                        "user_message": retry_message,
                        "file_name": file_spec.name,
                    })
                    response = self.executor.execute_agent(
                        role=AgentRole.CODER,
                        system_prompt=system_prompt,
                        user_message=retry_message
                    )
                else:
                    step = f"PLAN_FILE_{file_n}"
                    self._log_prompt(step, {
                        "system_prompt": system_prompt,
                        "user_message": user_message,
                        "file_name": file_spec.name,
                    })
                    response = self.executor.execute_agent(
                        role=AgentRole.CODER,
                        system_prompt=system_prompt,
                        user_message=user_message
                    )
                
                # Log result
                self._log_prompt(f"{step}_RESULT", {
                    "file_name": file_spec.name,
                    "result": response[:8000] if len(response) > 8000 else response,
                })

                # Clean response
                content = self._clean_code_output(response, file_spec.name)

                if not content.strip():
                    last_error = "Generated code is empty"
                    continue

                # Extract actual exports
                actual_exports = self._extract_exports(content)
                
                # Create result
                result = FileResult(
                    name=file_spec.name,
                    content=content,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED
                )
                
                # Verify plan compliance
                compliance = self._verify_plan_compliance(file_spec, result, context)
                result.plan_compliance = compliance
                
                if not compliance['passed']:
                    last_error = "PLAN COMPLIANCE FAILURE:\n" + '\n'.join(compliance['issues'])
                    self.logger.warning(f"Plan compliance failed for {file_spec.name}: {compliance['issues']}")
                    
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        result.status = FileStatus.PLAN_MISMATCH
                        result.notes = last_error
                
                return result
                
            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts - 1:
                    continue
                
                return FileResult(
                    name=file_spec.name,
                    content="",
                    actual_exports=[],
                    status=FileStatus.FAILED,
                    notes=str(e)
                )
        
        return FileResult(
            name=file_spec.name,
            content="",
            actual_exports=[],
            status=FileStatus.FAILED,
            notes=f"All attempts failed: {last_error}"
        )
    
    def _handle_revision(
        self,
        file_spec: FileSpec,
        result: FileResult,
        context: Dict[str, Any],
        user_request: str
    ) -> FileResult:
        """Handle revision when plan compliance fails.

        Phase 0  — Syntax auto-repair (deterministic tokenize fix, then
                   surgical fast-model fix). Zero/cheap cost.
        Phase 1  — Structural issues (missing exports, wrong imports, integration):
                   broad whole-file revision, up to max_file_revisions attempts.
        Phase 2  — Implementation issues (stub methods detected by _detect_placeholders):
                   surgical per-method fix, up to max_issue_retries per method.
        Phase 3  — Final full compliance verification.
        """
        from swarm_coordinator_v2 import AgentRole

        # Rotate through different coder models on retries
        _CODER_ROTATION = [
            AgentRole.CODER,
            AgentRole.CODER_2,
            AgentRole.CODER_3,
            AgentRole.CODER_4,
        ]

        total_revisions = 0

        # =================================================================
        # Phase 0: Syntax auto-repair (before wasting a full revision)
        # =================================================================
        issues = result.plan_compliance.get("issues", [])
        syntax_issues = [i for i in issues if i.startswith("Syntax")]
        if syntax_issues:
            self.logger.info(f"Phase 0: syntax auto-repair for {file_spec.name}")

            # Layer 2: deterministic tokenize bracket fix
            fixed_content, was_fixed = self._auto_fix_syntax(result.content)
            if was_fixed:
                self.logger.info(f"Deterministic syntax fix succeeded for {file_spec.name}")
                actual_exports = self._extract_exports(fixed_content)
                result = FileResult(
                    name=file_spec.name,
                    content=fixed_content,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED,
                    revision_count=result.revision_count,
                )
                compliance = self._verify_plan_compliance(file_spec, result, context)
                result.plan_compliance = compliance
                if compliance["passed"]:
                    return result
                # Syntax fixed but other issues remain — fall through to Phase 1
            else:
                # Layer 3: surgical syntax fix with fast model
                for syntax_issue in syntax_issues:
                    patched = self._surgical_syntax_fix(
                        result.content, syntax_issue, file_spec
                    )
                    if patched:
                        actual_exports = self._extract_exports(patched)
                        result = FileResult(
                            name=file_spec.name,
                            content=patched,
                            actual_exports=actual_exports,
                            status=FileStatus.COMPLETED,
                            revision_count=result.revision_count,
                        )
                        compliance = self._verify_plan_compliance(file_spec, result, context)
                        result.plan_compliance = compliance
                        if compliance["passed"]:
                            return result
                        break  # syntax fixed, fall through to Phase 1

        # =================================================================
        # Phase 1: Structural issues (broad revision)
        # =================================================================
        categorized = self._categorize_issues(result.plan_compliance.get("issues", []))
        structural_issues = categorized["structural"]

        structural_attempt = 0
        while structural_issues and structural_attempt < self.max_file_revisions:
            structural_attempt += 1
            total_revisions += 1
            self.logger.info(
                f"Structural revision {structural_attempt}/{self.max_file_revisions} "
                f"for {file_spec.name} ({len(structural_issues)} issues)"
            )

            system_prompt = """You are an expert programmer REVISING code to match a plan.

The previous code was REJECTED because it doesn't match the required plan.
You MUST fix the issues listed below.

OUTPUT RULES:
1. Output ONLY valid, executable Python code
2. NO markdown code blocks (```)
3. Fix ALL issues mentioned
4. Keep working parts of the original code

ANTI-STUB RULES (VIOLATIONS CAUSE REJECTION):
1. NO NotImplementedError — implement real logic instead
2. NO methods whose ONLY statement is pass — implement real logic
3. NO methods whose ONLY statement is ellipsis (...) — implement real logic
4. NO "placeholder" or "simulate" comments — these indicate unfinished code
5. NO hardcoded return values that ignore all function parameters
6. NO echo stubs — f-strings that just format parameters back without logic
7. NO hardcoded string returns for methods that should compute a result
8. NO truncation stubs — returning sliced input instead of processing it
9. Methods with self.X dependencies MUST actually use them
10. If a feature needs an external service not available, implement a working alternative using stdlib

INTEGRATION RULES (VIOLATIONS CAUSE REJECTION):
1. Constructor calls MUST match the EXACT arguments from the dependency's __init__ signature
2. ONLY import from files listed in the plan — do NOT invent modules that don't exist
3. Check FULL FILE CONTEXT below for actual signatures of dependencies
4. Constructor keyword args must match actual __init__ parameter names — do NOT invent field names
5. When accessing self.X.Y, Y must be an actual attribute/method of X's class (check FULL FILE CONTEXT)"""

            mandatory_exports = self._generate_mandatory_exports(file_spec)
            import_block = self._generate_import_block(file_spec)

            # Build focused dependency signatures — only direct deps, signatures only
            dep_sigs = []
            for dep_name in file_spec.dependencies:
                dep_info = context.get("dependencies", {}).get(dep_name, {})
                summary = dep_info.get("summary", "")
                if summary:
                    dep_sigs.append(f"--- {dep_name} ---\n{summary[:1500]}")
            dep_section = "\n".join(dep_sigs) if dep_sigs else "(no dependencies)"

            user_message = f"""REVISE THIS FILE: {file_spec.name}
Purpose: {file_spec.purpose}

STRUCTURAL FAILURES (MUST FIX):
{chr(10).join('- ' + i for i in structural_issues)}

{mandatory_exports}

REQUIRED IMPORTS (copy exactly):
{import_block if import_block else "(no imports required)"}

DEPENDENCY SIGNATURES (for correct constructor calls and attribute access):
{dep_section}

ORIGINAL CODE (fix the issues above, keep working parts):
{result.content}

Output the COMPLETE fixed code:"""

            try:
                role = _CODER_ROTATION[(structural_attempt - 1) % len(_CODER_ROTATION)]
                print(f"    Structural revision {structural_attempt}/{self.max_file_revisions} using {role.value}")

                rev_step = f"PLAN_FILE_{self._file_counter}_STRUCT_REV_{structural_attempt}"
                self._log_prompt(rev_step, {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "file_name": file_spec.name,
                    "revision_attempt": structural_attempt,
                    "compliance_issues": structural_issues,
                })

                response = self.executor.execute_agent(
                    role=role,
                    system_prompt=system_prompt,
                    user_message=user_message,
                )

                self._log_prompt(f"{rev_step}_RESULT", {
                    "file_name": file_spec.name,
                    "result": response[:8000] if len(response) > 8000 else response,
                })

                content = self._clean_code_output(response, file_spec.name)
                actual_exports = self._extract_exports(content)

                revised = FileResult(
                    name=file_spec.name,
                    content=content,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED,
                    revision_count=total_revisions,
                )

                compliance = self._verify_plan_compliance(file_spec, revised, context)
                revised.plan_compliance = compliance

                if compliance["passed"]:
                    self.logger.info(f"Structural revision passed for {file_spec.name}")
                    return revised

                result = revised
                # Re-categorize for next loop
                categorized = self._categorize_issues(compliance.get("issues", []))
                structural_issues = categorized["structural"]

            except Exception as e:
                self.logger.error(f"Structural revision failed: {e}")

        # =================================================================
        # Phase 2: Implementation issues (surgical per-method)
        # =================================================================
        categorized = self._categorize_issues(result.plan_compliance.get("issues", []))
        impl_issues = categorized["implementation"]

        if impl_issues:
            self.logger.info(
                f"Surgical phase for {file_spec.name}: {len(impl_issues)} method issues"
            )
            method_pattern = re.compile(r'\[(\w+)\.(\w+)\]')
            current_content = result.content

            for issue in impl_issues:
                m = method_pattern.search(issue)
                if not m:
                    continue
                cls_name, meth_name = m.group(1), m.group(2)
                last_response = None
                same_count = 0

                for retry in range(self.max_issue_retries):
                    method_src = self._extract_method_source(current_content, meth_name, cls_name)
                    if not method_src:
                        self.logger.warning(
                            f"Could not extract {cls_name}.{meth_name} — skipping"
                        )
                        break

                    sys_prompt, usr_msg = self._build_surgical_prompt(
                        file_spec, meth_name, method_src, issue, context, cls_name,
                    )

                    try:
                        role = _CODER_ROTATION[retry % len(_CODER_ROTATION)]
                        print(f"    Surgical fix {cls_name}.{meth_name} attempt {retry + 1}/{self.max_issue_retries} using {role.value}")

                        surg_step = (
                            f"PLAN_FILE_{self._file_counter}_SURG_"
                            f"{cls_name}_{meth_name}_{retry + 1}"
                        )
                        self._log_prompt(surg_step, {
                            "system_prompt": sys_prompt,
                            "user_message": usr_msg,
                            "file_name": file_spec.name,
                            "revision_attempt": retry + 1,
                            "compliance_issues": [issue],
                        })

                        response = self.executor.execute_agent(
                            role=role,
                            system_prompt=sys_prompt,
                            user_message=usr_msg,
                        )

                        self._log_prompt(f"{surg_step}_RESULT", {
                            "file_name": file_spec.name,
                            "result": response[:4000] if len(response) > 4000 else response,
                        })

                        fixed_method = self._clean_code_output(response, file_spec.name)

                        # Early exit: if the model keeps returning the same thing,
                        # the code is probably correct and our check is a false positive
                        if fixed_method == last_response:
                            same_count += 1
                            if same_count >= 2:
                                self.logger.info(
                                    f"Surgical fix for {cls_name}.{meth_name}: "
                                    f"model returned same result 3 times — accepting as-is"
                                )
                                current_content = self._replace_method_source(
                                    current_content, meth_name, fixed_method, cls_name,
                                )
                                break
                        else:
                            same_count = 0
                        last_response = fixed_method

                        # Validate the method parses on its own
                        try:
                            ast.parse(fixed_method)
                        except SyntaxError:
                            self.logger.warning(
                                f"Surgical fix for {cls_name}.{meth_name} has syntax error, retrying"
                            )
                            continue

                        # Splice it in
                        patched = self._replace_method_source(
                            current_content, meth_name, fixed_method, cls_name,
                        )

                        # Validate the full file still parses
                        try:
                            ast.parse(patched)
                        except SyntaxError:
                            self.logger.warning(
                                f"Full file broke after splicing {cls_name}.{meth_name}, retrying"
                            )
                            continue

                        # Check if this specific issue is resolved
                        new_placeholder_issues = self._detect_placeholders(patched)
                        still_flagged = any(
                            f"[{cls_name}.{meth_name}]" in pi for pi in new_placeholder_issues
                        )
                        if not still_flagged:
                            self.logger.info(
                                f"Surgical fix succeeded: {cls_name}.{meth_name} "
                                f"(attempt {retry + 1})"
                            )
                            current_content = patched
                            break
                        else:
                            self.logger.info(
                                f"Surgical fix for {cls_name}.{meth_name} "
                                f"still flagged, retrying ({retry + 1}/{self.max_issue_retries})"
                            )
                            current_content = patched  # use improved version for next retry

                    except Exception as e:
                        self.logger.error(
                            f"Surgical fix error for {cls_name}.{meth_name}: {e}"
                        )

            # Update result with surgically patched content
            actual_exports = self._extract_exports(current_content)
            result = FileResult(
                name=file_spec.name,
                content=current_content,
                actual_exports=actual_exports,
                status=FileStatus.COMPLETED,
                revision_count=total_revisions,
            )

        # =================================================================
        # Phase 3: Final verification
        # =================================================================
        compliance = self._verify_plan_compliance(file_spec, result, context)
        result.plan_compliance = compliance

        if compliance["passed"]:
            self.logger.info(f"Revision successful for {file_spec.name}")
            return result

        # All revisions failed
        result.status = FileStatus.FAILED
        result.notes = (
            f"Failed after {structural_attempt} structural + surgical revisions. "
            f"Remaining issues: {compliance.get('issues', [])}"
        )
        return result
    
    # =========================================================================
    # INIT FILE GENERATION
    # =========================================================================
    
    def _generate_init_files(self) -> Dict[str, str]:
        """
        Generate __init__.py files for all folders containing Python files.
        """
        init_files = {}
        
        # Find all folders
        folders = set()
        for file_spec in self.plan.files:
            if '/' in file_spec.name:
                folder = file_spec.name.rsplit('/', 1)[0]
                folders.add(folder)
                # Also add parent folders
                while '/' in folder:
                    folder = folder.rsplit('/', 1)[0]
                    folders.add(folder)
        
        # Generate __init__.py for each folder
        for folder in folders:
            init_path = f"{folder}/__init__.py"
            
            # Collect exports from files in this folder
            folder_exports = []
            for file_spec in self.plan.files:
                if file_spec.name.startswith(folder + '/'):
                    # Direct child of this folder
                    if '/' not in file_spec.name[len(folder)+1:]:
                        module_name = file_spec.name.rsplit('/', 1)[-1].replace('.py', '')
                        for exp in file_spec.exports:
                            folder_exports.append((module_name, exp.name))
            
            # Generate content
            lines = ['"""Auto-generated __init__.py"""', '']
            
            if folder_exports:
                # Group by module
                by_module = defaultdict(list)
                for module, name in folder_exports:
                    by_module[module].append(name)
                
                for module, names in by_module.items():
                    lines.append(f"from .{module} import {', '.join(names)}")
                
                lines.append('')
                all_names = [name for _, name in folder_exports]
                lines.append(f"__all__ = {all_names!r}")
            
            init_files[init_path] = '\n'.join(lines)
        
        return init_files
    
    # =========================================================================
    # REVIEWER COMPLIANCE (per-file, inline)
    # =========================================================================

    _REVIEWER_SYSTEM_PROMPT = (
        "You are a code reviewer performing a FOCUSED COMPLIANCE REVIEW on a SINGLE FILE.\n\n"
        "YOUR JOB: Check for stub/placeholder implementations and missing required exports. That's it.\n\n"
        "ONLY FAIL a file if:\n"
        "1. A required export from the plan spec is MISSING (class/function not defined)\n"
        "2. A method has a stub body: pass-only, ellipsis-only, NotImplementedError, or returns a hardcoded constant while ignoring its parameters\n"
        "3. A method's docstring describes complex behavior but the body is 1-3 trivial lines that don't do real work\n\n"
        "DO NOT FAIL a file for:\n"
        "- Missing error handling, shutdown logic, or cleanup code (unless explicitly in plan requirements)\n"
        "- Code style preferences or architectural opinions\n"
        "- Features you think SHOULD exist but aren't in the plan spec\n"
        "- Minor issues that don't affect functionality\n\n"
        "OUTPUT FORMAT (follow exactly):\n"
        "FILE: <filename>\n"
        "STATUS: PASS | FAIL\n\n"
        "STUB AUDIT:\n"
        "- [For each stub: method name, what it should do, what it actually does]\n"
        "- If no stubs: \"No stubs detected\"\n\n"
        "EXPORT CHECK:\n"
        "- [Each required export from plan spec and whether it exists]\n\n"
        "ISSUES REQUIRING REVISION:\n"
        "- [Only list issues that match the FAIL criteria above, or \"None\" if PASS]"
    )

    def _reviewer_compliance_check(self, file_spec, content: str) -> Tuple[bool, List[str]]:
        """Run LLM reviewer compliance check on a single file.

        Returns (passed, issues_list).  Issues list is empty on PASS.
        """
        from swarm_coordinator_v2 import AgentRole

        # Build plan spec string (same format the coordinator uses)
        spec_parts = [f"File: {file_spec.name}", f"Purpose: {file_spec.purpose}"]
        if file_spec.exports:
            spec_parts.append("Required exports:")
            for exp in file_spec.exports:
                methods_str = ""
                if hasattr(exp, "methods") and exp.methods:
                    methods_str = f" (methods: {', '.join(exp.methods.keys())})"
                spec_parts.append(f"  - {exp.name} ({exp.type}){methods_str}")
        if file_spec.requirements:
            spec_parts.append("Requirements:")
            for req in file_spec.requirements:
                spec_parts.append(f"  - {req}")
        if file_spec.imports_from:
            spec_parts.append("Imports from:")
            for src, names in file_spec.imports_from.items():
                spec_parts.append(f"  - {src}: {', '.join(names)}")
        plan_spec_str = "\n".join(spec_parts)

        # Build dependency code (direct deps only, 4K cap each)
        dep_parts = []
        for dep_name in file_spec.dependencies:
            dep_result = self.completed_files.get(dep_name)
            if dep_result and dep_result.content:
                dep_code = dep_result.content
                if len(dep_code) > 4000:
                    dep_code = dep_code[:4000] + "\n# ... (truncated)"
                dep_parts.append(f"### {dep_name} ###\n{dep_code}")
        dep_section = "\n\n".join(dep_parts)

        user_message = f"FILE TO REVIEW: {file_spec.name}\n\nFILE CODE:\n{content}\n\nPLAN SPEC FOR THIS FILE:\n{plan_spec_str}"
        if dep_section:
            user_message += f"\n\nDEPENDENCY CODE:\n{dep_section}"
        user_message += "\n\nPerform a focused compliance review on this single file. Check every method for stubs."

        step_name = f"PLAN_FILE_{self._file_counter}_REVIEWER"
        self._log_prompt(step_name, {
            "system_prompt": "FILE_COMPLIANCE_PROMPT",
            "user_message": f"file={file_spec.name} ({len(content)} chars)",
            "file_name": file_spec.name,
        })

        try:
            response = self.executor.execute_agent(
                role=AgentRole.REVIEWER,
                system_prompt=self._REVIEWER_SYSTEM_PROMPT,
                user_message=user_message,
            )
        except Exception as e:
            self.logger.warning(f"Reviewer call failed for {file_spec.name}: {e}")
            return True, []  # don't block on reviewer failure

        self._log_prompt(f"{step_name}_RESULT", {
            "file_name": file_spec.name,
            "result": response[:5000] if len(response) > 5000 else response,
        })

        # Parse STATUS: PASS|FAIL
        upper = response.upper()
        if "STATUS: FAIL" not in upper and "STATUS:FAIL" not in upper:
            return True, []

        # Extract issues
        issues = []
        import re as _re
        issues_match = _re.search(
            r'ISSUES REQUIRING REVISION[:\s]*\n(.*?)(?:\n\n|\nFILE:|\nOVERALL:|\Z)',
            response, _re.DOTALL | _re.IGNORECASE,
        )
        if issues_match:
            for line in issues_match.group(1).strip().split("\n"):
                line = line.strip().lstrip("-").lstrip("0123456789.").strip()
                if line and line.lower() != "none":
                    issues.append(line)
        stub_match = _re.search(
            r'STUB AUDIT[:\s]*\n(.*?)(?:\n\n|\nEXPORT|\nREQUIREMENTS|\nISSUES|\Z)',
            response, _re.DOTALL | _re.IGNORECASE,
        )
        if stub_match:
            for line in stub_match.group(1).strip().split("\n"):
                line = line.strip().lstrip("-").strip()
                if not line:
                    continue
                low = line.lower()
                # Skip lines that say there is NO stub
                if any(phrase in low for phrase in (
                    "no stub", "none", "not a stub", "does not have a stub",
                    "not have a stub", "is not stub", "no stubs detected",
                    "not a placeholder", "fully implemented",
                )):
                    continue
                issues.append(f"STUB: {line}")

        if not issues:
            # Reviewer said FAIL but we found no parseable issues — check
            # if it actually listed zero problems (reviewer LLM confusion)
            issues_section = _re.search(
                r'ISSUES REQUIRING REVISION[:\s]*\n(.*?)(?:\n\n|\Z)',
                response, _re.DOTALL | _re.IGNORECASE,
            )
            if issues_section:
                content = issues_section.group(1).strip().lower()
                if content in ("none", "- none", "n/a", "no issues", ""):
                    # Reviewer said FAIL but listed no issues — treat as PASS
                    self.logger.warning(
                        f"Reviewer said FAIL but ISSUES section is '{content}' — overriding to PASS"
                    )
                    return True, []
            issues = [response[:2000]]

        return False, issues

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    def execute(self, plan_yaml: str, user_request: str, job_scope: str = "") -> Dict[str, Any]:
        """
        Execute the full plan with strict enforcement.
        """
        self.start_time = time.time()
        
        # Parse plan
        self.plan = self.parse_plan(plan_yaml)
        self.job_scope = job_scope or user_request
        
        self.logger.info("=" * 70)
        self.logger.info(f"PLAN EXECUTOR V2: {self.plan.name}")
        self.logger.info("=" * 70)
        self.logger.info(f"Description: {self.plan.description}")
        self.logger.info(f"Files: {len(self.plan.files)}")
        self.logger.info(f"Order: {' -> '.join(self.plan.execution_order)}")
        
        print(f"\n{'=' * 70}")
        print(f"PLAN EXECUTOR V2: {self.plan.name}")
        print(f"{'=' * 70}")
        print(f"Files to generate: {len(self.plan.files)}")
        print(f"Order: {' → '.join(self.plan.execution_order)}")
        print(f"{'=' * 70}\n")
        
        # Execute each file — wrapped in try/except so partial results survive crashes
        for filename in self.plan.execution_order:
            try:
                file_spec = self._get_file_spec(filename)
                if not file_spec:
                    self.logger.warning(f"Unknown file in execution order: {filename}")
                    continue

                print(f"\n▶ Generating: {filename}")
                print(f"  Purpose: {file_spec.purpose}")
                print(f"  Exports: {[e.name for e in file_spec.exports]}")

                self.status[filename] = FileStatus.IN_PROGRESS

                # Build context
                context = self._build_context_for_file(file_spec)

                # Generate file
                result = self._generate_file(file_spec, user_request, context)

                # Handle plan mismatch with revision
                if result.status == FileStatus.PLAN_MISMATCH:
                    print(f"  ⚠ Plan compliance failed, attempting revision...")
                    result = self._handle_revision(file_spec, result, context, user_request)

                # Reviewer compliance check (LLM-based, per-file)
                if result.status == FileStatus.COMPLETED and result.content:
                    print(f"  🔍 Reviewer compliance check...")
                    rev_passed, rev_issues = self._reviewer_compliance_check(file_spec, result.content)
                    if rev_passed:
                        print(f"  ✓ Reviewer: PASS")
                    else:
                        print(f"  ✗ Reviewer: FAIL ({len(rev_issues)} issues)")
                        for iss in rev_issues[:5]:
                            print(f"    - {iss[:120]}")
                        # Feed reviewer issues back into revision
                        result = FileResult(
                            name=filename,
                            content=result.content,
                            actual_exports=result.actual_exports,
                            status=FileStatus.PLAN_MISMATCH,
                            plan_compliance={"passed": False, "issues": rev_issues, "warnings": []},
                        )
                        result = self._handle_revision(file_spec, result, context, user_request)

                # Store result
                self.completed_files[filename] = result
                self.status[filename] = result.status

                # Report
                if result.status == FileStatus.COMPLETED:
                    print(f"  ✓ Generated successfully")
                    if result.plan_compliance.get('warnings'):
                        for warning in result.plan_compliance['warnings']:
                            print(f"    ⚠ {warning}")
                else:
                    print(f"  ✗ Failed: {result.notes}")
                    self.errors.append({
                        'filename': filename,
                        'error': result.notes,
                        'compliance': result.plan_compliance
                    })
            except Exception as e:
                print(f"  ✗ CRASH generating {filename}: {e}")
                self.logger.error(f"Unhandled error generating {filename}: {e}")
                self.errors.append({
                    'filename': filename,
                    'error': f"Unhandled crash: {e}",
                    'compliance': {}
                })
                self.status[filename] = FileStatus.FAILED
                # Continue to next file — don't lose already-completed files

        # Generate __init__.py files
        init_files = self._generate_init_files()
        for init_path, content in init_files.items():
            self.completed_files[init_path] = FileResult(
                name=init_path,
                content=content,
                actual_exports=[],
                status=FileStatus.COMPLETED,
                notes="Auto-generated"
            )
            print(f"  ✓ Generated: {init_path} (auto)")
        
        # Generate combined output
        combined_output = self._generate_combined_output()
        
        elapsed = time.time() - self.start_time
        
        print(f"\n{'=' * 70}")
        print(f"EXECUTION COMPLETE ({elapsed:.1f}s)")
        print(f"{'=' * 70}")
        print(f"Successful: {sum(1 for s in self.status.values() if s == FileStatus.COMPLETED)}")
        print(f"Failed: {sum(1 for s in self.status.values() if s != FileStatus.COMPLETED)}")
        
        return {
            'success': len(self.errors) == 0,
            'plan': self.plan,
            'completed_files': self.completed_files,
            'status': {k: v.value for k, v in self.status.items()},
            'errors': self.errors,
            'combined_output': combined_output,
            'init_files': init_files
        }
    
    def _generate_combined_output(self) -> str:
        """Generate combined output in ### FILE: ### format."""
        parts = []
        
        # Regular files first
        for filename in self.plan.execution_order:
            if filename in self.completed_files:
                result = self.completed_files[filename]
                if result.content:
                    parts.append(f"### FILE: {filename} ###")
                    parts.append(result.content)
                    parts.append("")
        
        # Then __init__.py files
        for filename, result in self.completed_files.items():
            if filename.endswith('__init__.py'):
                parts.append(f"### FILE: {filename} ###")
                parts.append(result.content)
                parts.append("")
        
        return '\n'.join(parts)


# =============================================================================
# ARCHITECT PROMPT (Enhanced for better plans)
# =============================================================================

ARCHITECT_PLAN_SYSTEM_PROMPT = """You are an expert software architect creating a YAML execution plan.

Your plan will be STRICTLY ENFORCED - the coder MUST export exactly what you specify.
Be precise about exports and imports.

OUTPUT: Valid YAML with this structure:

```yaml
program:
  name: "project_name"
  description: "What it does"
  type: "cli|library|service"

architecture:
  pattern: "simple|modular"
  entry_point: "main.py"

files:
  - name: "config.py"
    purpose: "Configuration management"
    dependencies: []
    exports:
      - name: "Settings"
        type: "class"
        methods:
          __init__:
            args: ["self", "config_path: str = None"]
          get:
            args: ["self", "key: str"]
            returns: "Any"
    imports_from: {}
    requirements:
      - "Load config from file or environment"
      - "Provide get() method for accessing values"

  - name: "core.py"
    purpose: "Core business logic"
    dependencies: ["config.py"]
    exports:
      - name: "Processor"
        type: "class"
        methods:
          __init__:
            args: ["self", "settings: Settings"]
          process:
            args: ["self", "data: dict"]
            returns: "dict"
    imports_from:
      config.py: ["Settings"]
    requirements:
      - "Main processing logic"

  - name: "main.py"
    purpose: "Entry point"
    dependencies: ["config.py", "core.py"]
    exports: []
    imports_from:
      config.py: ["Settings"]
      core.py: ["Processor"]
    requirements:
      - "Parse CLI arguments"
      - "Initialize and run Processor"

execution_order: ["config.py", "core.py", "main.py"]
```

CRITICAL RULES:
1. exports MUST list what each file will define (classes, functions, constants)
2. imports_from MUST only reference names in the source file's exports
3. dependencies MUST include every file used in imports_from
4. execution_order MUST have dependencies before dependents
5. Include method signatures for classes when they're part of the public API

Keep it SIMPLE. Output ONLY the YAML."""


def get_architect_plan_prompt(user_request: str, job_scope: str = "") -> str:
    """Build user message for architect."""
    if job_scope and job_scope != "Requirements are clear and complete.":
        return f"Create a YAML execution plan for:\n\n{job_scope}\n\nOutput the YAML:"
    return f"Create a YAML execution plan for:\n\n{user_request}\n\nOutput the YAML:"


def extract_yaml_from_response(response: str) -> str:
    """Extract YAML from architect response."""
    # Try markdown block
    match = re.search(r'```ya?ml\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try raw YAML
    match = re.search(r'(program:\s*\n.*)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return response.strip()


# =============================================================================
# INTEGRATION WITH SWARMCOORDINATOR
# =============================================================================

def create_planned_workflow(coordinator, user_request: str):
    """Create planned workflow tasks using collaborative front-end + PlanExecutor build."""
    from swarm_coordinator_v2 import AgentRole, Task, TaskStatus

    # T001: Structured clarification (same as workflow 8)
    coordinator.add_task(Task(
        task_id="T001_clarify",
        task_type="clarification",
        description="Clarify requirements and produce structured job spec",
        assigned_role=AgentRole.CLARIFIER,
        status=TaskStatus.PENDING,
        priority=10,
        metadata={"user_request": user_request, "output_format": "structured"}
    ))

    # T002: Architect drafts plan (same as workflow 8)
    coordinator.add_task(Task(
        task_id="T002_draft_plan",
        task_type="draft_plan",
        description="Create architecture plan",
        assigned_role=AgentRole.ARCHITECT,
        status=TaskStatus.PENDING,
        priority=9,
        dependencies=["T001_clarify"],
        metadata={"user_request": user_request, "output_format": "yaml_plan"}
    ))

    # T003: Coder reviews plan (same as workflow 8)
    coordinator.add_task(Task(
        task_id="T003_plan_review",
        task_type="plan_review",
        description="Review the plan",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=8,
        dependencies=["T002_draft_plan"],
        metadata={"user_request": user_request}
    ))

    # T004: Architect finalizes plan with coder feedback (same as workflow 8)
    coordinator.add_task(Task(
        task_id="T004_finalize_plan",
        task_type="finalize_plan",
        description="Finalize the plan",
        assigned_role=AgentRole.ARCHITECT,
        status=TaskStatus.PENDING,
        priority=7,
        dependencies=["T003_plan_review"],
        metadata={"user_request": user_request}
    ))

    # T005: PlanExecutor builds files one-by-one (keeps existing behavior)
    coordinator.add_task(Task(
        task_id="T005_plan_execution",
        task_type="plan_execution",
        description="Execute the plan file-by-file",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=6,
        dependencies=["T004_finalize_plan"],
        metadata={"user_request": user_request, "use_plan_executor": True}
    ))

    # T006: Generate and validate tests
    coordinator.add_task(Task(
        task_id="T006_test",
        task_type="test_generation",
        description="Generate tests",
        assigned_role=AgentRole.TESTER,
        status=TaskStatus.PENDING,
        priority=5,
        dependencies=["T005_plan_execution"],
        metadata={"user_request": user_request}
    ))

    # T007: Compliance review against plan + job spec
    coordinator.add_task(Task(
        task_id="T007_compliance",
        task_type="compliance_review",
        description="Compliance review",
        assigned_role=AgentRole.REVIEWER,
        status=TaskStatus.PENDING,
        priority=4,
        dependencies=["T005_plan_execution"],
        metadata={"user_request": user_request}
    ))

    # T008: Documentation
    coordinator.add_task(Task(
        task_id="T008_document",
        task_type="documentation",
        description="Generate documentation",
        assigned_role=AgentRole.DOCUMENTER,
        status=TaskStatus.PENDING,
        priority=3,
        dependencies=["T007_compliance"],
        metadata={"user_request": user_request}
    ))

    # T009: Final verification
    coordinator.add_task(Task(
        task_id="T009_verify",
        task_type="verification",
        description="Final verification",
        assigned_role=AgentRole.VERIFIER,
        status=TaskStatus.PENDING,
        priority=2,
        dependencies=["T007_compliance", "T006_test"]
    ))


def execute_plan_task(coordinator, task, plan_yaml: str, user_request: str, job_scope: str = "") -> str:
    """Execute plan_execution task."""
    project_dir = coordinator.state.get("project_info", {}).get("project_dir", "")
    executor = PlanExecutor(
        executor=coordinator.executor,
        config=coordinator.config,
        project_dir=project_dir
    )
    
    result = executor.execute(plan_yaml, user_request, job_scope)
    
    if result['success']:
        return result['combined_output']
    else:
        print("\n⚠ Plan execution had errors:")
        for error in result['errors']:
            print(f"  - {error['filename']}: {error['error']}")
        return result['combined_output']


if __name__ == "__main__":
    print("Plan Executor V2 - Run via SwarmCoordinator")
