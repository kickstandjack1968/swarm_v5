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
    
    def __init__(self, executor, config: Dict, log_dir: str = "logs"):
        self.executor = executor
        self.config = config
        self.plan: Optional[ProgramPlan] = None
        self.completed_files: Dict[str, FileResult] = {}
        self.status: Dict[str, FileStatus] = {}
        self.errors: List[Dict[str, Any]] = []
        self.max_file_revisions = 3
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir)
        self.job_scope = ""
        self.start_time: Optional[float] = None
    
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
            
            # Determine import style based on folder structure
            source_folder = source_file.rsplit('/', 1)[0] if '/' in source_file else ''
            source_module = source_file.replace('.py', '')
            
            if source_folder == current_folder and current_folder:
                # Same folder -> relative import
                source_base = source_file.rsplit('/', 1)[-1].replace('.py', '')
                import_stmt = f"from .{source_base} import {', '.join(import_names)}"
            elif '/' in source_file:
                # Different folder -> absolute with dots
                import_path = source_module.replace('/', '.')
                import_stmt = f"from {import_path} import {', '.join(import_names)}"
            else:
                # Root level -> relative import
                import_stmt = f"from .{source_module} import {', '.join(import_names)}"
            
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
        
        # Check bracket balance
        if content.count('(') != content.count(')'):
            return False, f"Mismatched parentheses"
        if content.count('[') != content.count(']'):
            return False, f"Mismatched brackets"
        if content.count('{') != content.count('}'):
            return False, f"Mismatched braces"
        
        # AST parse
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
        
        # 3. Import compliance
        import_issues = self._validate_imports_against_plan(result.content, file_spec)
        if import_issues:
            # Import issues are warnings, not failures (coder might have valid reasons)
            compliance["warnings"].extend(import_issues)
        
        # 4. Placeholder detection
        placeholder_issues = self._detect_placeholders(result.content)
        if placeholder_issues:
            compliance["passed"] = False
            compliance["issues"].extend(placeholder_issues)
        
        # 5. Integration check (call signatures)
        if file_spec.dependencies:
            integration_issues = self._check_integration(file_spec, result, context)
            if integration_issues:
                compliance["warnings"].extend(integration_issues)
        
        return compliance
    
    def _detect_placeholders(self, content: str) -> List[str]:
        """Detect lazy placeholder implementations."""
        issues = []
        
        placeholder_patterns = [
            (r'raise\s+NotImplementedError', "Contains NotImplementedError"),
            (r'pass\s*$', "Contains empty pass statement"),
            (r'\.\.\.', "Contains ellipsis placeholder"),
            (r'#\s*TODO', "Contains TODO comment"),
            (r'#\s*FIXME', "Contains FIXME comment"),
            (r'return\s+None\s*#', "Suspicious return None with comment"),
        ]
        
        for pattern, message in placeholder_patterns:
            if re.search(pattern, content, re.MULTILINE):
                # Check context - some are legitimate
                if pattern == r'pass\s*$':
                    # pass is OK in except blocks, abstract methods, etc.
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.match(r'^\s+pass\s*$', line):
                            # Check if previous line suggests it's a stub
                            if i > 0:
                                prev = lines[i-1].strip()
                                if prev.startswith('def ') and ':' in prev:
                                    # Function with just pass - likely stub
                                    issues.append(f"{message} in function definition")
                                    break
                elif pattern == r'\.\.\.':
                    # Ellipsis is OK in type hints, not in function bodies
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                                if node.value.value is ...:
                                    issues.append(message)
                                    break
                    except:
                        pass
                else:
                    issues.append(message)
        
        return issues
    
    def _check_integration(
        self,
        file_spec: FileSpec,
        result: FileResult,
        context: Dict[str, Any]
    ) -> List[str]:
        """Check that calls to dependencies use correct signatures."""
        issues = []
        
        try:
            tree = ast.parse(result.content)
        except SyntaxError:
            return issues
        
        # Build signature map from dependencies
        dep_signatures = {}
        for dep_name, dep_info in context.get('dependencies', {}).items():
            dep_content = dep_info.get('content', '')
            if dep_content:
                try:
                    from agent_base import extract_function_signatures, extract_class_info
                    
                    # Get function signatures
                    sigs = extract_function_signatures(dep_content)
                    dep_signatures.update(sigs)
                    
                    # Get class method signatures
                    classes = extract_class_info(dep_content)
                    for class_name, class_info in classes.items():
                        for method_name, method_info in class_info.get('methods', {}).items():
                            dep_signatures[f"{class_name}.{method_name}"] = method_info
                except ImportError:
                    pass
        
        # Check calls in current file
        class CallChecker(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_Call(self, node):
                # Get function name
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                else:
                    self.generic_visit(node)
                    return
                
                # Check against known signatures
                if func_name in dep_signatures:
                    sig = dep_signatures[func_name]
                    required = sig.get('required_args', 0)
                    provided = len(node.args) + len(node.keywords)
                    
                    if provided < required:
                        self.issues.append(
                            f"Call to {func_name}() provides {provided} args but requires {required}"
                        )
                
                self.generic_visit(node)
        
        checker = CallChecker()
        checker.visit(tree)
        issues.extend(checker.issues)
        
        return issues
    
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
4. NO placeholder implementations (pass, ..., NotImplementedError)
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
                
                # Methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        try:
                            args_str = ast.unparse(item.args)
                        except:
                            args_str = "..."
                        lines.append(f"    def {item.name}({args_str}): ...")
            
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    try:
                        args_str = ast.unparse(node.args)
                    except:
                        args_str = "..."
                    lines.append(f"\ndef {node.name}({args_str}): ...")
        
        return '\n'.join(lines)
    
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
        """Handle revision when plan compliance fails."""
        
        from swarm_coordinator_v2 import AgentRole
        
        for attempt in range(self.max_file_revisions):
            self.logger.info(f"Revision attempt {attempt + 1}/{self.max_file_revisions} for {file_spec.name}")
            
            # Build revision prompt with plan context
            system_prompt = """You are an expert programmer REVISING code to match a plan.

The previous code was REJECTED because it doesn't match the required plan.
You MUST fix the issues listed below.

OUTPUT RULES:
1. Output ONLY valid, executable Python code
2. NO markdown code blocks
3. Fix ALL issues mentioned
4. Keep working parts of the original code"""
            
            # Include plan requirements in revision prompt
            mandatory_exports = self._generate_mandatory_exports(file_spec)
            import_block = self._generate_import_block(file_spec)
            
            user_message = f"""REVISE THIS FILE: {file_spec.name}

COMPLIANCE FAILURES (MUST FIX):
{chr(10).join('- ' + i for i in result.plan_compliance.get('issues', []))}

{mandatory_exports}

REQUIRED IMPORTS (copy exactly):
{import_block if import_block else "(no imports required)"}

ORIGINAL CODE (with errors):
{result.content}

Output the COMPLETE fixed code:"""
            
            try:
                # Use fallback coder on later attempts
                role = AgentRole.CODER if attempt == 0 else AgentRole.FALLBACK_CODER
                
                response = self.executor.execute_agent(
                    role=role,
                    system_prompt=system_prompt,
                    user_message=user_message
                )
                
                content = self._clean_code_output(response, file_spec.name)
                actual_exports = self._extract_exports(content)
                
                revised_result = FileResult(
                    name=file_spec.name,
                    content=content,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED,
                    revision_count=attempt + 1
                )
                
                # Re-verify
                compliance = self._verify_plan_compliance(file_spec, revised_result, context)
                revised_result.plan_compliance = compliance
                
                if compliance['passed']:
                    self.logger.info(f"Revision successful for {file_spec.name}")
                    return revised_result
                
                result = revised_result
                
            except Exception as e:
                self.logger.error(f"Revision failed: {e}")
        
        # All revisions failed
        result.status = FileStatus.FAILED
        result.notes = f"Failed after {self.max_file_revisions} revision attempts"
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
        
        # Execute each file
        for filename in self.plan.execution_order:
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
    """Create planned workflow tasks."""
    from swarm_coordinator_v2 import AgentRole, Task, TaskStatus
    
    # Clarification
    coordinator.add_task(Task(
        task_id="T001_clarify",
        task_type="clarification",
        description="Clarify requirements",
        assigned_role=AgentRole.CLARIFIER,
        status=TaskStatus.PENDING,
        priority=10,
        metadata={"user_request": user_request}
    ))
    
    # Architecture plan
    coordinator.add_task(Task(
        task_id="T002_plan",
        task_type="architecture_plan",
        description="Create YAML execution plan",
        assigned_role=AgentRole.ARCHITECT,
        status=TaskStatus.PENDING,
        priority=9,
        dependencies=["T001_clarify"],
        metadata={"user_request": user_request, "output_format": "yaml_plan"}
    ))
    
    # Plan execution
    coordinator.add_task(Task(
        task_id="T003_execute_plan",
        task_type="plan_execution",
        description="Execute plan file-by-file",
        assigned_role=AgentRole.CODER,
        status=TaskStatus.PENDING,
        priority=8,
        dependencies=["T002_plan"],
        metadata={"user_request": user_request, "use_plan_executor": True}
    ))
    
    # Test generation
    coordinator.add_task(Task(
        task_id="T004_tests",
        task_type="test_generation",
        description="Generate tests",
        assigned_role=AgentRole.TESTER,
        status=TaskStatus.PENDING,
        priority=7,
        dependencies=["T003_execute_plan"],
        metadata={"user_request": user_request}
    ))
    
    # Documentation
    coordinator.add_task(Task(
        task_id="T005_document",
        task_type="documentation",
        description="Generate documentation",
        assigned_role=AgentRole.DOCUMENTER,
        status=TaskStatus.PENDING,
        priority=6,
        dependencies=["T003_execute_plan", "T004_tests"],
        metadata={"user_request": user_request}
    ))
    
    # Verification
    coordinator.add_task(Task(
        task_id="T006_verify",
        task_type="verification",
        description="Final verification",
        assigned_role=AgentRole.VERIFIER,
        status=TaskStatus.PENDING,
        priority=5,
        dependencies=["T005_document"]
    ))


def execute_plan_task(coordinator, task, plan_yaml: str, user_request: str, job_scope: str = "") -> str:
    """Execute plan_execution task."""
    executor = PlanExecutor(
        executor=coordinator.executor,
        config=coordinator.config
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
