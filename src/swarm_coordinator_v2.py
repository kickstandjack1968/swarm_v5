#!/usr/bin/env python3
"""
Advanced Multi-Agent Swarm Coordinator v2 - FIXED
- Parallel agent execution
- Dynamic routing and task distribution
- Enhanced state management
- Tool integration for agents
- Metrics and observability

FIXES APPLIED:
- Added VERIFIER to fallback map
- Fixed clarification flow to properly feed architect
- Added custom workflow support
- Fixed context propagation for all tasks
- Added revision loop when reviewer rejects
- Fixed security audit task handling
- Added proper empty result handling
- Fixed metrics lock issue
- Improved documenter to receive actual code
- Added user_request to all downstream contexts
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from enum import Enum
import requests
from threading import Lock
import re
import subprocess
import sys


# Plan Executor integration
try:
    from .plan_executor_v2 import (
        PlanExecutor,
        create_planned_workflow,
        execute_plan_task,
        ARCHITECT_PLAN_SYSTEM_PROMPT,
        get_architect_plan_prompt,
        extract_yaml_from_response,
        FileSpec,
        FileResult,
        FileStatus
    )
    PLAN_EXECUTOR_AVAILABLE = True
except ImportError:
    # Fallback to absolute import for direct script execution
    try:
        from plan_executor_v2 import (
            PlanExecutor,
            create_planned_workflow,
            execute_plan_task,
            ARCHITECT_PLAN_SYSTEM_PROMPT,
            get_architect_plan_prompt,
            extract_yaml_from_response,
            FileSpec,
            FileResult,
            FileStatus
        )
        PLAN_EXECUTOR_AVAILABLE = True
    except ImportError as e:
        print(f"Import failed: {e}")
        PLAN_EXECUTOR_AVAILABLE = False


class DockerSandbox:
    """
    Manages a persistent Docker container for running hot tests during swarm execution.
    Container is started once and reused for all tests, then cleaned up at the end.
    """
    
    CONTAINER_NAME = "swarm_sandbox"
    IMAGE = "python:3.12-slim"
    BASE_PACKAGES = ["pytest", "pyyaml", "requests", "chardet", "python-dateutil"]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.container_running = False
        self.installed_requirements: set = set()
        self._check_docker_available()
    
    def _check_docker_available(self):
        """Check if Docker is available on the system"""
        if not self.enabled:
            return
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                print("⚠ Docker not available, falling back to local execution")
                self.enabled = False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("⚠ Docker not found, falling back to local execution")
            self.enabled = False
    
    def start(self) -> bool:
        """Start the sandbox container with base packages installed"""
        if not self.enabled:
            return False
        
        # Clean up any existing container with same name
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True
        )
        
        # Start container with tail -f /dev/null to keep it alive
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.CONTAINER_NAME,
                "-w", "/workspace",
                self.IMAGE,
                "tail", "-f", "/dev/null"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"⚠ Failed to start Docker sandbox: {result.stderr}")
            self.enabled = False
            return False
        
        # Install base packages
        print("🐳 Starting Docker sandbox...")
        install_cmd = ["pip", "install", "--quiet"] + self.BASE_PACKAGES
        result = self._exec(install_cmd, timeout=120)
        
        if result.returncode != 0:
            print(f"⚠ Failed to install base packages: {result.stderr}")
            self.stop()
            self.enabled = False
            return False
        
        # Create workspace directories
        self._exec(["mkdir", "-p", "/workspace/src", "/workspace/tests"])
        
        self.container_running = True
        print(f"   ✓ Sandbox ready (base packages: {', '.join(self.BASE_PACKAGES)})")
        return True
    
    def stop(self):
        """Stop and remove the sandbox container"""
        if not self.enabled:
            return
        
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True
        )
        self.container_running = False
        self.installed_requirements.clear()
        print("🐳 Docker sandbox stopped")
    
    def _exec(self, cmd: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        """Execute a command inside the container"""
        full_cmd = ["docker", "exec", self.CONTAINER_NAME] + cmd
        return subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def install_requirements(self, requirements_content: str) -> bool:
        """Install requirements from requirements.txt content (only new ones)"""
        if not self.enabled or not self.container_running:
            return False
        
        # Parse requirements
        new_reqs = set()
        for line in requirements_content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (without version specifier)
                pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()
                if pkg and pkg not in self.installed_requirements:
                    new_reqs.add(pkg)
        
        if not new_reqs:
            return True
        
        # Install new requirements
        install_cmd = ["pip", "install", "--quiet"] + list(new_reqs)
        result = self._exec(install_cmd, timeout=120)
        
        if result.returncode == 0:
            self.installed_requirements.update(new_reqs)
            return True
        else:
            print(f"⚠ Failed to install some requirements: {result.stderr[:200]}")
            return False
    
    def copy_to_container(self, local_path: str, container_path: str) -> bool:
        """Copy a file or directory to the container"""
        if not self.enabled or not self.container_running:
            return False
        
        result = subprocess.run(
            ["docker", "cp", local_path, f"{self.CONTAINER_NAME}:{container_path}"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    def write_file(self, container_path: str, content: str) -> bool:
        """Write content to a file inside the container"""
        if not self.enabled or not self.container_running:
            return False
        
        # Use docker exec with echo/cat to write file
        # Escape content for shell
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        
        result = self._exec([
            "sh", "-c",
            f"echo '{encoded}' | base64 -d > {container_path}"
        ])
        return result.returncode == 0
    
    def run_pytest(self, test_file: str, timeout: int = 60) -> Tuple[bool, str]:
        """Run pytest on a test file inside the container"""
        if not self.enabled or not self.container_running:
            return False, "Docker sandbox not available"
        
        result = self._exec(
            ["python", "-m", "pytest", test_file, "-v"],
            timeout=timeout
        )
        
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return result.returncode == 0, output
    
    def cleanup_workspace(self):
        """Clean the workspace directories for next test"""
        if not self.enabled or not self.container_running:
            return
        
        self._exec(["sh", "-c", "rm -rf /workspace/src/* /workspace/tests/*"])
    
    def run_hot_test(
        self,
        completed_files: Dict[str, Any],
        test_content: str,
        file_name: str,
        requirements_content: str = ""
    ) -> Tuple[bool, str]:
        """
        Run a hot test in the Docker sandbox.
        
        Args:
            completed_files: Dict of filename -> FileResult with .content attribute
            test_content: The test file content
            file_name: Name of the file being tested (for naming the test)
            requirements_content: Optional requirements.txt content
            
        Returns:
            (success, output) tuple
        """
        if not self.enabled or not self.container_running:
            return self._run_local_fallback(completed_files, test_content, file_name)
        
        try:
            # Clean workspace
            self.cleanup_workspace()
            
            # Install requirements if provided
            if requirements_content:
                self.install_requirements(requirements_content)
            
            # Write all source files
            for fname, result in completed_files.items():
                content = result.content if hasattr(result, 'content') else str(result)
                container_path = f"/workspace/src/{fname}"
                
                # Ensure directory exists
                dir_path = os.path.dirname(container_path)
                if dir_path != "/workspace/src":
                    self._exec(["mkdir", "-p", dir_path])
                
                if not self.write_file(container_path, content):
                    return False, f"Failed to write {fname} to container"
            
            # Ensure __init__.py exists
            self.write_file("/workspace/src/__init__.py", "")
            
            # Write test file
            test_name = f"test_{file_name}"
            if not self.write_file(f"/workspace/tests/{test_name}", test_content):
                return False, "Failed to write test file to container"
            
            # Run pytest
            return self.run_pytest(f"/workspace/tests/{test_name}", timeout=60)
            
        except subprocess.TimeoutExpired:
            return False, "Test execution timed out"
        except Exception as e:
            return False, f"Docker execution error: {str(e)}"
    
    def _run_local_fallback(
        self,
        completed_files: Dict[str, Any],
        test_content: str,
        file_name: str
    ) -> Tuple[bool, str]:
        """Fallback to local execution if Docker is not available"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = os.path.join(tmp_dir, "src")
            tests_dir = os.path.join(tmp_dir, "tests")
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(tests_dir, exist_ok=True)
            
            # Write source files
            for fname, result in completed_files.items():
                content = result.content if hasattr(result, 'content') else str(result)
                full_path = os.path.join(src_dir, fname)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
            
            # Ensure __init__.py
            init_path = os.path.join(src_dir, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write("")
            
            # Write test file
            test_name = f"test_{file_name}"
            with open(os.path.join(tests_dir, test_name), 'w') as f:
                f.write(test_content)
            
            # Run pytest
            env = os.environ.copy()
            env["PYTHONPATH"] = tmp_dir
            
            try:
                cmd = [sys.executable, "-m", "pytest", os.path.join(tests_dir, test_name)]
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if proc.returncode == 0:
                    return True, proc.stdout
                else:
                    return False, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                    
            except subprocess.TimeoutExpired:
                return False, "Execution timed out (infinite loop?)"
            except Exception as e:
                return False, f"Local execution error: {str(e)}"


# Global sandbox instance (initialized when SwarmCoordinator starts)
_docker_sandbox: Optional[DockerSandbox] = None


def get_docker_sandbox() -> Optional[DockerSandbox]:
    """Get the global Docker sandbox instance"""
    global _docker_sandbox
    # Lazily ensure the sandbox is started when requested by other modules.
    if _docker_sandbox:
        try:
            # If sandbox is enabled but not yet running, attempt to start it.
            if getattr(_docker_sandbox, 'enabled', False) and not getattr(_docker_sandbox, 'container_running', False):
                _docker_sandbox.start()
        except Exception:
            # Do not raise here; callers will fall back to local execution if needed.
            pass
    return _docker_sandbox


def init_docker_sandbox(enabled: bool = True) -> DockerSandbox:
    """Initialize the global Docker sandbox"""
    global _docker_sandbox
    _docker_sandbox = DockerSandbox(enabled=enabled)
    return _docker_sandbox


def shutdown_docker_sandbox():
    """Shutdown the global Docker sandbox"""
    global _docker_sandbox
    if _docker_sandbox:
        _docker_sandbox.stop()
        _docker_sandbox = None

class AgentRole(Enum):
    """Defined agent roles in the swarm"""
    ARCHITECT = "architect"
    CLARIFIER = "clarifier"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    OPTIMIZER = "optimizer"
    DOCUMENTER = "documenter"
    DEBUGGER = "debugger"
    SECURITY = "security"
    VERIFIER = "verifier"
    FALLBACK_CODER= "fallback_coder"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_REVISION = "needs_revision"


@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking"""
    agent_name: str
    role: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    avg_response_time: float = 0.0
    last_call_time: Optional[float] = None
    
    def update(self, success: bool, response_time: float, tokens: int = 0):
        """Update metrics after an agent call"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        self.total_tokens += tokens
        
        # Update average response time
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time * (self.total_calls - 1) + response_time) / self.total_calls
        
        self.last_call_time = response_time


@dataclass
class Task:
    """Represents a task in the workflow"""
    task_id: str
    task_type: str
    description: str
    assigned_role: AgentRole
    status: TaskStatus
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    revision_count: int = 0
    max_revisions: int = 3


class AgentExecutor:
    """Handles execution of individual agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics: Dict[str, AgentMetrics] = {}
        self.metrics_lock = Lock()
        
    def _get_agent_config(self, role) -> Dict:
        """Get configuration for specific agent role"""
        role_str = role.value if hasattr(role, 'value') else str(role)
        mode = self.config['model_config']['mode']
        
        if mode == 'multi' and role_str in self.config['model_config']['multi_model']:
            return self.config['model_config']['multi_model'][role_str]
        elif mode == 'single':
            return self.config['model_config']['single_model']
        else:
            # FIXED: Added VERIFIER to fallback map
            fallback_map = {
                AgentRole.ARCHITECT: 'architect',
                AgentRole.CLARIFIER: 'clarifier',
                AgentRole.CODER: 'coder',
                AgentRole.FALLBACK_CODER: 'fallback_coder',
                AgentRole.REVIEWER: 'reviewer',
                AgentRole.TESTER: 'tester',
                AgentRole.OPTIMIZER: 'optimizer',
                AgentRole.DOCUMENTER: 'documenter',
                AgentRole.DEBUGGER: 'debugger',
                AgentRole.SECURITY: 'security',
                AgentRole.VERIFIER: 'verifier'
            }
            fallback_role = fallback_map.get(role, 'coder')
            if fallback_role in self.config['model_config']['multi_model']:
                return self.config['model_config']['multi_model'][fallback_role]
        
        return self.config['model_config']['single_model']
    
    def _call_api(self, url: str, api_type: str, model: str, system_prompt: str, 
                  user_message: str, params: Dict, timeout: int) -> tuple[str, int]:
        """Make API call to LLM (returns response and approximate token count)"""
        
        try:
            if api_type == 'ollama':
                ollama_url = url.replace('/v1', '').rstrip('/')
                response = requests.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": params.get('temperature', 0.7),
                            "num_predict": params.get('max_tokens', 4000),
                            "top_p": params.get('top_p', 0.9)
                        }
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                content = result.get('message', {}).get('content', '')
                tokens = result.get('eval_count', 0) + result.get('prompt_eval_count', 0)
                return content, tokens
                
            else:  # OpenAI-compatible
                response = requests.post(
                    f"{url}/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": params.get('temperature', 0.7),
                        "max_tokens": params.get('max_tokens', 4000),
                        "top_p": params.get('top_p', 0.9)
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                tokens = result.get('usage', {}).get('total_tokens', 0)
                return content, tokens
                
        except requests.exceptions.Timeout:
            raise Exception(f"API call timed out after {timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid API response format: {str(e)}")
    
    def execute_agent(self, role, system_prompt: str, user_message: str,
                      agent_params: Optional[Dict] = None) -> str:
        """Execute a single agent with the given prompts"""

        # Normalize role to string for dict lookups
        role_str = role.value if hasattr(role, 'value') else str(role)

        # Get agent configuration
        agent_config = self._get_agent_config(role)
        url = agent_config['url']
        model = agent_config.get('model', 'local-model')
        api_type = agent_config.get('api_type', 'openai')
        timeout = agent_config.get('timeout', 7200)

        # Merge parameters
        params = self.config.get('agent_parameters', {}).get(role_str, {}).copy()
        if agent_params:
            params.update(agent_params)

        # Initialize metrics if needed
        agent_key = role_str
        with self.metrics_lock:
            if agent_key not in self.metrics:
                self.metrics[agent_key] = AgentMetrics(agent_name=agent_key, role=role_str)
        
        # Execute the call
        start_time = time.time()
        success = False
        tokens = 0
        response = ""
        
        try:
            response, tokens = self._call_api(url, api_type, model, system_prompt, 
                                             user_message, params, timeout)
            # FIXED: Check for empty response
            if not response or not response.strip():
                raise Exception("Agent returned empty response")
            success = True
            return response
            
        except Exception as e:
            raise Exception(f"Agent {role_str} failed: {str(e)}")
            
        finally:
            elapsed = time.time() - start_time
            with self.metrics_lock:
                self.metrics[agent_key].update(success, elapsed, tokens)


class SwarmCoordinator:
    """Advanced coordinator for multi-agent swarm execution"""

    def _run_semantic_audit(self) -> dict:
        """Local audit for obvious placeholders/stubs in generated code."""
        project_dir = self.state.get("project_info", {}).get("project_dir")
        if not project_dir:
            return {"status": "fail", "reason": "missing project_dir", "findings": []}

        src_root = os.path.join(project_dir, "src")
        if not os.path.isdir(src_root):
            return {"status": "fail", "reason": "missing src/", "findings": []}

        findings = []
        markers = ("TODO", "FIXME", "HACK", "PLACEHOLDER", "STUB")

        for root, _, files in os.walk(src_root):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                if fn == "__init__.py":
                    continue

                path = os.path.join(root, fn)
                rel = os.path.relpath(path, project_dir)

                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception as e:
                    findings.append({"file": rel, "line": None, "kind": "read_error", "detail": str(e)})
                    continue

                for i, line in enumerate(content.splitlines(), start=1):
                    s = line.strip()

                    if s in ("pass", "..."):
                        findings.append({"file": rel, "line": i, "kind": "stub", "detail": s})
                        continue
                    if "NotImplementedError" in line:
                        findings.append({"file": rel, "line": i, "kind": "not_implemented", "detail": "NotImplementedError"})
                        continue

                    up = line.upper()
                    if any(m in up for m in markers):
                        findings.append({"file": rel, "line": i, "kind": "marker", "detail": s[:200]})
                        continue

        return {"status": "pass" if not findings else "fail", "count": len(findings), "findings": findings}


    def __init__(self, config_file: str = "config_v2.json", use_docker: bool = True):
        self.config = self._load_config(config_file)
        self.executor = AgentExecutor(self.config)
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.state = {
            "workflow_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "phase": "initial",
            "iteration": 0,
            "max_iterations": self.config.get('workflow', {}).get('max_iterations', 3),
            "context": {
                "user_request": "",
            },
            "history": [],
            "project_info": {
                "project_number": None,
                "project_name": None,
                "version": 1,
                "project_dir": None
            }
        }
        self.max_parallel = self.config.get('workflow', {}).get('max_parallel_agents', 4)
        self.projects_root = "projects"
        self._ensure_projects_dir()
        
        # Initialize Docker sandbox for hot tests
        self.use_docker = use_docker
        self.sandbox = None
        if use_docker:
            self.sandbox = init_docker_sandbox(enabled=True)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        # Try multiple paths
        paths_to_try = [
            config_file,
            os.path.join("config", config_file),
            os.path.join(os.path.dirname(__file__), config_file),
            os.path.join(os.path.dirname(__file__), "config", config_file)
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {path}: {e}")
        
        print(f"Warning: Config file not found, using defaults")
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            "model_config": {
                "mode": "single",
                "single_model": {
                    "url": "http://localhost:1233/v1",
                    "model": "local-model",
                    "api_type": "openai",
                    "timeout": 7200
                }
            },
            "agent_parameters": {
                "architect": {"temperature": 0.6, "max_tokens": 3000},
                "clarifier": {"temperature": 0.7, "max_tokens": 2000},
                "coder": {"temperature": 0.2, "max_tokens": 6000},
                "reviewer": {"temperature": 0.8, "max_tokens": 3000},
                "tester": {"temperature": 0.7, "max_tokens": 4000},
                "optimizer": {"temperature": 0.6, "max_tokens": 4000},
                "documenter": {"temperature": 0.7, "max_tokens": 3000},
                "debugger": {"temperature": 0.6, "max_tokens": 4000},
                "security": {"temperature": 0.8, "max_tokens": 3000},
                "verifier": {"temperature": 0.3, "max_tokens": 3000}
            },
            "workflow": {
                "max_iterations": 3,
                "max_parallel_agents": 4,
                "enable_parallel": True
            }
        }

    @staticmethod
    def _fix_missing_typing_imports(code: str) -> str:
        """Auto-fix missing typing imports in Python code.

        Scans for common typing names (Any, Dict, List, etc.) used in type hints
        and ensures they're imported from typing. This catches one of the most
        common LLM errors: using type annotations without importing them.
        """
        import re
        # All common typing module names
        typing_names = {
            'Any', 'Dict', 'List', 'Tuple', 'Set', 'FrozenSet',
            'Optional', 'Union', 'Callable', 'Iterator', 'Generator',
            'Sequence', 'Mapping', 'MutableMapping', 'Iterable',
            'Type', 'ClassVar', 'Final', 'Literal', 'TypeVar',
            'Protocol', 'NamedTuple', 'TypedDict', 'Awaitable',
            'Coroutine', 'AsyncIterator', 'AsyncGenerator',
        }

        lines = code.split('\n')

        # Find which typing names are used in the code (outside strings/comments)
        used_names = set()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            # Look for typing name usage patterns: annotations, generics
            for name in typing_names:
                # Match as standalone word (not part of a longer name)
                if re.search(r'\b' + name + r'\b', line):
                    used_names.add(name)

        if not used_names:
            return code

        # Find existing typing imports
        already_imported = set()
        typing_import_line_idx = None
        for i, line in enumerate(lines):
            m = re.match(r'^from\s+typing\s+import\s+(.+)', line)
            if m:
                typing_import_line_idx = i
                # Parse imported names (handle multi-line with parentheses later)
                import_str = m.group(1).strip()
                if import_str.startswith('('):
                    import_str = import_str[1:]
                import_str = import_str.rstrip(')')
                for name in import_str.split(','):
                    name = name.strip()
                    if name:
                        already_imported.add(name)

        missing = used_names - already_imported
        if not missing:
            return code

        if typing_import_line_idx is not None:
            # Add missing names to existing import line
            all_names = sorted(already_imported | missing)
            new_import = f"from typing import {', '.join(all_names)}"
            lines[typing_import_line_idx] = new_import
        else:
            # Add new typing import at the top (after any docstring and __future__)
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(('"""', "'''")):
                    # Skip past docstring
                    if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                        for j in range(i + 1, len(lines)):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                insert_idx = j + 1
                                break
                    else:
                        insert_idx = i + 1
                elif stripped.startswith('from __future__'):
                    insert_idx = i + 1
                elif stripped.startswith(('import ', 'from ')):
                    insert_idx = i  # Insert before first import
                    break
                elif stripped and not stripped.startswith('#'):
                    insert_idx = i
                    break

            new_import = f"from typing import {', '.join(sorted(missing))}"
            lines.insert(insert_idx, new_import)

        return '\n'.join(lines)

    @staticmethod
    def _fix_missing_stdlib_imports(code: str) -> str:
        """Auto-fix missing stdlib module imports.

        Detects patterns like `json.loads(...)` or `os.path.join(...)` where
        the module isn't imported and adds the import.
        """
        import re

        # Map: module name -> usage patterns to detect
        stdlib_modules = {
            'json': r'\bjson\.',
            'os': r'\bos\.',
            'sys': r'\bsys\.',
            'time': r'\btime\.(time|sleep|strftime|monotonic|perf_counter)\b',
            'datetime': r'\bdatetime\.(datetime|date|time|timedelta)\b',
            'logging': r'\blogging\.',
            'hashlib': r'\bhashlib\.',
            'base64': r'\bbase64\.',
            'pathlib': r'\bpathlib\.',
            'subprocess': r'\bsubprocess\.',
            'threading': r'\bthreading\.',
            'asyncio': r'\basyncio\.',
            'sqlite3': r'\bsqlite3\.',
            'math': r'\bmath\.',
            're': r'\bre\.(match|search|findall|sub|compile|split|IGNORECASE|MULTILINE|DOTALL)\b',
            'uuid': r'\buuid\.',
            'tempfile': r'\btempfile\.',
            'shutil': r'\bshutil\.',
            'collections': r'\bcollections\.',
            'functools': r'\bfunctools\.',
            'itertools': r'\bitertools\.',
            'copy': r'\bcopy\.(copy|deepcopy)\b',
            'struct': r'\bstruct\.',
            'io': r'\bio\.(BytesIO|StringIO|BufferedReader)\b',
            'csv': r'\bcsv\.',
            'secrets': r'\bsecrets\.',
        }

        lines = code.split('\n')

        # Find already imported modules
        imported = set()
        for line in lines:
            stripped = line.strip()
            m = re.match(r'^import\s+(\w+)', stripped)
            if m:
                imported.add(m.group(1))
            m = re.match(r'^from\s+(\w+)', stripped)
            if m:
                imported.add(m.group(1))

        # Detect usage of unimported modules
        missing = []
        code_no_strings = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')', '', code)
        # Also remove comments
        code_no_strings = re.sub(r'#.*', '', code_no_strings)

        for module, pattern in stdlib_modules.items():
            if module not in imported and re.search(pattern, code_no_strings):
                missing.append(module)

        if not missing:
            return code

        # Find insertion point (after existing imports)
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                insert_idx = i + 1

        # Insert missing imports
        for module in sorted(missing, reverse=True):
            lines.insert(insert_idx, f'import {module}')

        return '\n'.join(lines)

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
        
        # Generate empty __init__.py for each
        # NOTE: PlanExecutor already generates proper __init__.py with specific imports.
        # We only create empty ones here to ensure packages are importable.
        # Using `from .module import *` causes circular imports (e.g. from .main import *).
        for folder in folders_with_py:
            init_path = os.path.join(folder, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write('"""Auto-generated __init__.py"""\n')

                print(f"   ✓ Generated: {os.path.relpath(init_path, project_dir)}")


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

    def _ensure_projects_dir(self):
        """Ensure projects directory exists"""
        if not os.path.exists(self.projects_root):
            os.makedirs(self.projects_root)
    
    def _get_next_project_number(self) -> int:
        """Get next project number by scanning existing projects"""
        if not os.path.exists(self.projects_root):
            return 1
        
        existing = os.listdir(self.projects_root)
        numbers = []
        for dirname in existing:
            if dirname[0].isdigit():
                try:
                    num = int(dirname.split('_')[0])
                    numbers.append(num)
                except (ValueError, IndexError):
                    continue
        
        return max(numbers) + 1 if numbers else 1
    
    def _create_project_name(self, user_request: str) -> str:
        """Generate project name from user request"""
        words = user_request.lower().split()
        
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                     'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'should', 'could', 'may', 'might', 'must', 'can', 'create', 'make',
                     'build', 'write', 'generate', 'i', 'need', 'want', 'please', 'help',
                     'that', 'this', 'which', 'what', 'where', 'when', 'how', 'why'}
        
        keywords = [w for w in words[:15] if w not in stop_words and len(w) > 2]
        name_parts = keywords[:3] if keywords else ['project']
        project_name = '_'.join(name_parts)
        project_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in project_name)
        project_name = re.sub(r'_+', '_', project_name).strip('_')
        
        return project_name or 'project'
    
    def _check_existing_project(self, project_name: str) -> Optional[int]:
        """Check if project exists and return latest version number"""
        if not os.path.exists(self.projects_root):
            return None
        
        existing = os.listdir(self.projects_root)
        versions = []
        
        for dirname in existing:
            if project_name in dirname and '_v' in dirname:
                try:
                    version = int(dirname.split('_v')[-1])
                    versions.append(version)
                except (ValueError, IndexError):
                    continue
        
        return max(versions) if versions else None
    
    def _setup_project_directory(self, project_name: str, user_request: str) -> str:
        """Create project directory structure"""
        project_num = self._get_next_project_number()
        version = 1
        
        existing_version = self._check_existing_project(project_name)
        if existing_version is not None:
            version = existing_version + 1
        
        dir_name = f"{project_num:03d}_{project_name}_v{version}"
        project_dir = os.path.join(self.projects_root, dir_name)
        
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "tests"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "docs"), exist_ok=True)
        
        self.state["project_info"] = {
            "project_number": project_num,
            "project_name": project_name,
            "version": version,
            "project_dir": project_dir
        }
        
        self._create_project_info(user_request)
        
        return project_dir
    
    def _create_project_info(self, user_request: str):
        """Create PROJECT_INFO.txt in project directory"""
        project_dir = self.state["project_info"]["project_dir"]
        project_num = self.state["project_info"]["project_number"]
        version = self.state["project_info"]["version"]
        project_name = self.state["project_info"]["project_name"]
        
        info_content = f"""PROJECT INFORMATION
{'=' * 80}

Project Number: {project_num:03d}
Project Name: {project_name}
Version: {version}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Workflow ID: {self.state['workflow_id']}

{'=' * 80}
ORIGINAL REQUEST
{'=' * 80}

{user_request}

{'=' * 80}
PROJECT STRUCTURE
{'=' * 80}

{os.path.basename(project_dir)}/
├── src/              - Source code
├── tests/            - Test files
├── docs/             - Documentation
├── PROJECT_INFO.txt  - This file
└── requirements.txt  - Dependencies (if applicable)

{'=' * 80}
"""
        
        info_path = os.path.join(project_dir, "PROJECT_INFO.txt")
        with open(info_path, 'w') as f:
            f.write(info_content)

    def _set_project_type_from_plan(self, plan_yaml: str):
        """Extract project type and entry point from architect's YAML plan.

        Supports:
          project: {type: ..., entry_point: ...}
        and planned-workflow format:
          program: {type: ...}
          architecture: {entry_point: ...}
        """
        import yaml
        try:
            plan = yaml.safe_load(plan_yaml)
            if not plan or not isinstance(plan, dict):
                return

            project_info = plan.get("project", {}) or {}
            program_info = plan.get("program", {}) or {}
            arch_info = plan.get("architecture", {}) or {}

            project_type = (project_info.get("type")
                            or program_info.get("type")
                            or project_info.get("project_type")
                            or program_info.get("project_type")
                            or "cli")
            if project_type in ("cli", "subprocess_tool", "library", "service"):
                self.state["project_info"]["project_type"] = project_type

            entry_point = (project_info.get("entry_point")
                           or project_info.get("entrypoint")
                           or arch_info.get("entry_point")
                           or arch_info.get("entrypoint"))
            if entry_point:
                self.state["project_info"]["entry_point"] = entry_point
            elif project_type == "library":
                self.state["project_info"]["entry_point"] = None

            print(f"   Project type: {self.state['project_info'].get('project_type')}")
            print(f"   Entry point: {self.state['project_info'].get('entry_point')}")
        except Exception as e:
            print(f"   Warning: Could not parse project type from plan: {e}")

    def _save_project_outputs(self):
        """Save all outputs to project directory with SMART ROUTING"""
        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir:
            return
        
        # Save code - handle multi-file output
        code_task = next((t for t in reversed(self.completed_tasks) if t.task_type in ("coding", "plan_execution", "revision", "build_from_plan")), None)
        if code_task and code_task.result:
            files_dict = self._parse_multi_file_output(code_task.result)
            
            if files_dict:
                # Multi-file output detected
                for filename, content in files_dict.items():
                    # Normalize path separators
                    filename = filename.replace('\\', '/')
                    
                    # INTELLIGENT ROUTING
                    if filename.startswith("src/"):
                        # Explicitly marked for src -> Strip prefix, go to src/
                        rel_path = filename[len("src/"):]
                        base_dir = os.path.join(project_dir, "src")
                    elif filename.startswith("tests/"):
                        # Explicitly marked for tests -> Go to root/tests
                        rel_path = filename 
                        base_dir = project_dir 
                    elif filename.startswith("docs/"):
                        # Explicitly marked for docs -> Go to root/docs
                        rel_path = filename
                        base_dir = project_dir
                    else:
                        # Implicit source file -> Go to src/ (SAFE DEFAULT)
                        rel_path = filename
                        base_dir = os.path.join(project_dir, "src")

                    code_file = os.path.join(base_dir, rel_path)

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(code_file), exist_ok=True)

                    # Validate Python syntax before saving
                    if code_file.endswith('.py'):
                        content = self._validate_and_fix_python_syntax(content)

                    with open(code_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                print(f"   ✓ Created {len(files_dict)} files (smart routed)")

            else:
                # Single file output (legacy behavior) - assume source
                project_name = self.state["project_info"]["project_name"]
                code_file = os.path.join(project_dir, "src", f"{project_name}.py")
                
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code_task.result)
        self._generate_init_files(project_dir)


        # Save tests (from Tester agent specifically)
        test_task = next((t for t in self.completed_tasks if t.task_type == "test_generation"), None)
        if test_task and test_task.result:
            project_name = self.state["project_info"]["project_name"]
            # If tester outputted multiple files, use parsing logic
            if "### FILE:" in test_task.result:
                test_files = self._parse_multi_file_output(test_task.result)
                for fname, content in test_files.items():
                    # Force into tests dir if not already
                    if not fname.startswith("tests/"):
                        fname = f"tests/{fname}"
                    # Validate Python test file syntax
                    if fname.endswith('.py'):
                        content = self._validate_and_fix_python_syntax(content)
                    path = os.path.join(project_dir, fname)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
            else:
                # Single test file fallback
                test_file = os.path.join(project_dir, "tests", f"test_{project_name}.py")
                test_content = self._clean_test_output(test_task.result)
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)


        # Save documentation
        doc_task = next((t for t in self.completed_tasks if t.task_type == "documentation"), None)
        if doc_task and doc_task.result:
            doc_file = os.path.join(project_dir, "docs", "README.md")
            
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(doc_task.result)
            
            # Also copy to project root
            root_readme = os.path.join(project_dir, "README.md")
            with open(root_readme, 'w', encoding='utf-8') as f:
                f.write(doc_task.result)
        
        # Save review results
        review_tasks = [t for t in self.completed_tasks if "review" in t.task_type and t.result]
        if review_tasks:
            review_file = os.path.join(project_dir, "REVIEW_RESULTS.txt")
            
            with open(review_file, 'w', encoding='utf-8') as f:
                f.write("CODE REVIEW RESULTS\n")
                f.write("=" * 80 + "\n\n")
                for i, task in enumerate(review_tasks, 1):
                    f.write(f"Review #{i} ({task.assigned_role.value}):\n")
                    f.write("-" * 80 + "\n")
                    f.write(task.result + "\n\n")
        
        # Save requirements.txt
        self._create_requirements_txt(project_dir)
        
        # Save session state
        self.save_state()


    def _check_entry_point_integration(self, code_content: str) -> str:
        """
        Check that the generated project has a usable entry point based on project type.
        Returns a short human-readable report for the verifier prompt (no exceptions).
        """
        try:
            if not code_content or not code_content.strip():
                return "ENTRYPOINT CHECK: SKIP\n- No code content provided."
            files = self._parse_multi_file_output(code_content)
            if not files:
                return "ENTRYPOINT CHECK: SKIP\n- Could not parse multi-file output."
            project_type = self.state["project_info"].get("project_type", "cli")
            entry_point = self.state["project_info"].get("entry_point", "src/main.py")
           
            # Libraries don't need entry point checks
            if project_type == "library":
                return "ENTRYPOINT CHECK: SKIP (library project)"
            # Normalize entry point path
            entry_point_normalized = entry_point.replace("\\", "/")
           
            # Find the entry point file
            entry_code = None
            for p in files.keys():
                pp = p.replace("\\", "/")
                if pp == entry_point_normalized or pp.endswith("/" + entry_point_normalized.split("/")[-1]):
                    entry_code = files[p]
                    entry_point_normalized = pp
                    break
            issues = []
           
            if entry_code is None:
                issues.append(f"Missing entry point: {entry_point}")
            else:
                # Check based on project type
                if project_type == "cli":
                    # Check for __name__ == "__main__" guard (the essential part)
                    has_name_guard = "__name__" in entry_code and "__main__" in entry_code
                    
                    # Check for any callable function (doesn't have to be named "main")
                    has_callable = "def " in entry_code or "class " in entry_code
                    
                    # Check if something is called under the name guard
                    # Look for pattern: if __name__ followed by function call
                    has_execution = has_name_guard and ("()" in entry_code)
                    
                    if not has_name_guard:
                        issues.append(f"{entry_point_normalized}: missing __name__ == '__main__' guard.")
                    
                    if not has_callable and not has_execution:
                        issues.append(f"{entry_point_normalized}: no executable code found.")
                    
                    # Only warn about argparse if the program seems to need arguments
                    # (has sys.argv, mentions "args", "arguments", etc.)
                    needs_args = any(x in entry_code.lower() for x in ["sys.argv", "argument", "args", "option", "flag", "--"])
                    if needs_args and "argparse" not in entry_code and "click" not in entry_code and "typer" not in entry_code:
                        issues.append(f"{entry_point_normalized}: program may need argument parsing but no argparse/click/typer found.")
                       
                elif project_type == "subprocess_tool":
                    if "def main" not in entry_code:
                        issues.append(f"{entry_point_normalized}: missing main() function.")
                    if "__name__" not in entry_code:
                        issues.append(f"{entry_point_normalized}: missing __name__ guard.")
                    # Subprocess tools typically read stdin/write stdout
                    if "stdin" not in entry_code and "sys.stdin" not in entry_code:
                        issues.append(f"{entry_point_normalized}: no stdin reading detected (expected for subprocess tool).")
                       
                elif project_type == "service":
                    # Services might use various patterns, just check it's importable
                    pass
                # Common check: warn about problematic imports
                if "from src." in entry_code or "import src." in entry_code:
                    issues.append(
                        f"{entry_point_normalized}: imports using 'src.' may fail depending on how it's run. "
                        f"Consider relative imports or running as module."
                    )
            if issues:
                return "ENTRYPOINT CHECK: WARN\n- " + "\n- ".join(issues)
            return "ENTRYPOINT CHECK: PASS"
        except Exception as e:
            return f"ENTRYPOINT CHECK: SKIP\n- Exception during checks: {e}"
    
    def _parse_multi_file_output(self, code_output: str) -> Dict[str, str]:
        """Parse coder output that contains multiple files."""
        files = {}
        
        # Pattern 1: ### FILE: filename.py ### (trailing ### optional)
        file_pattern = r'###\s*FILE:\s*(\S+?)(?:\s*###)?\s*$'
        matches = list(re.finditer(file_pattern, code_output, re.MULTILINE))

        if matches:
            for i, match in enumerate(matches):
                filename = match.group(1).strip()
                start_pos = match.end()
                
                # Find end position (start of next file or end of string)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(code_output)
                
                content = code_output[start_pos:end_pos].strip()
                content = self._clean_file_content(content)
                
                if content:
                    files[filename] = content
            return files
    
        # Pattern 2: #### File: `filename.py` ... ```python ... ```
        md_pattern = r'#{1,4}\s*[Ff]ile:\s*`([^`]+)`.*?\n\s*```(?:python|py)?\s*\n(.*?)```'
        md_matches = list(re.finditer(md_pattern, code_output, re.DOTALL))
        if md_matches:
            for match in md_matches:
                filename = match.group(1).strip()
                content = match.group(2).strip()
                if content:
                    files[filename] = content
            if files:
                return files
        
        # Pattern 3: **filename.py** or ==filename.py== (bold/underline markers)
        bold_pattern = r'\*{2}(\w+\.py)\*{2}|==(\w+\.py)=='
        bold_matches = list(re.finditer(bold_pattern, code_output))
        if bold_matches:
            for match in bold_matches:
                filename = match.group(1) or match.group(2)
                start_pos = match.end()
                # Find next bold or end
                next_match = re.search(r'\*{2}\w+\.py\*{2}|==\w+\.py==', code_output[start_pos:])
                if next_match:
                    end_pos = start_pos + next_match.start()
                else:
                    end_pos = len(code_output)
                content = code_output[start_pos:end_pos].strip()
                content = self._clean_file_content(content)
                if content:
                    files[filename] = content
            if files:
                return files
        
        # Pattern 4: filename.py: (followed by code)
        colon_pattern = r'^(\w+\.py):\s*\n'
        for match in re.finditer(colon_pattern, code_output, re.MULTILINE):
            filename = match.group(1)
            start_pos = match.end()
            # Find next filename.py: or end
            next_match = re.search(r'^(\w+\.py):\s*\n', code_output[start_pos:], re.MULTILINE)
            if next_match:
                end_pos = start_pos + next_match.start()
            else:
                end_pos = len(code_output)
            content = code_output[start_pos:end_pos].strip()
            content = self._clean_file_content(content)
            if content:
                files[filename] = content
        if files:
            return files
        
        # Pattern 5: --- filename.py --- or === filename.py ===
        dash_pattern = r'^[-=]{3}\s*(\w+\.py)\s*[-=]{3}'
        for match in re.finditer(dash_pattern, code_output, re.MULTILINE):
            filename = match.group(1)
            start_pos = match.end()
            next_match = re.search(r'^[-=]{3}\s*\w+\.py\s*[-=]{3}', code_output[start_pos:], re.MULTILINE)
            if next_match:
                end_pos = start_pos + next_match.start()
            else:
                end_pos = len(code_output)
            content = code_output[start_pos:end_pos].strip()
            content = self._clean_file_content(content)
            if content:
                files[filename] = content
        if files:
            return files
        
        # Pattern 6: File: filename.py (no backticks, just plain text)
        plain_pattern = r'^File:\s*(\w+\.py)\s*$'
        for match in re.finditer(plain_pattern, code_output, re.MULTILINE):
            filename = match.group(1)
            start_pos = match.end()
            # Skip to next non-empty line that starts code
            next_line_match = re.search(r'^(\w|\"|\'|import|from|class|def|#)', code_output[start_pos:], re.MULTILINE)
            if next_line_match:
                code_start = start_pos + next_line_match.start()
            else:
                code_start = start_pos
            next_file_match = re.search(r'^File:\s*\w+\.py\s*$', code_output[code_start:], re.MULTILINE)
            if next_file_match:
                end_pos = code_start + next_file_match.start()
            else:
                end_pos = len(code_output)
            content = code_output[code_start:end_pos].strip()
            content = self._clean_file_content(content)
            if content:
                files[filename] = content
        
        return files


    def _extract_exports_from_code(self, code: str) -> List[str]:
        """Extract public function and class names from Python code"""
        exports = []
        for line in code.split('\n'):
            stripped = line.strip()
            # Top-level function definitions (not indented)
            if line.startswith('def ') and not line.startswith('    '):
                match = re.match(r'def\s+(\w+)', stripped)
                if match and not match.group(1).startswith('_'):
                    exports.append(match.group(1))
            # Top-level class definitions
            elif line.startswith('class ') and not line.startswith('    '):
                match = re.match(r'class\s+(\w+)', stripped)
                if match and not match.group(1).startswith('_'):
                    exports.append(match.group(1))
            # Module-level constants (ALL_CAPS)
            elif re.match(r'^[A-Z][A-Z_0-9]*\s*=', stripped) and not line.startswith(' '):
                var_name = stripped.split('=')[0].strip()
                exports.append(var_name)
        return exports

    def _extract_signatures(self, code: str) -> str:
        """Extract function/class signatures and docstrings for documentation context"""
        lines = code.split('\n')
        signatures = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Capture class definitions with docstrings
            if line.startswith('class ') or (stripped.startswith('class ') and not line.startswith('    ')):
                signatures.append(line)
                # Capture docstring if present
                i += 1
                while i < len(lines) and (lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")):
                    signatures.append(lines[i])
                    if lines[i].strip().endswith('"""') or lines[i].strip().endswith("'''"):
                        if len(lines[i].strip()) > 3:  # Single line docstring
                            i += 1
                            break
                    i += 1
                    # Multi-line docstring
                    while i < len(lines):
                        signatures.append(lines[i])
                        if '"""' in lines[i] or "'''" in lines[i]:
                            i += 1
                            break
                        i += 1
                continue
            
            # Capture function definitions with docstrings
            if line.startswith('def ') or (stripped.startswith('def ') and not line.startswith('        ')):
                signatures.append(line)
                # Capture docstring if present
                i += 1
                if i < len(lines):
                    next_stripped = lines[i].strip()
                    if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                        signatures.append(lines[i])
                        if not (next_stripped.endswith('"""') or next_stripped.endswith("'''")):
                            # Multi-line docstring
                            i += 1
                            while i < len(lines):
                                signatures.append(lines[i])
                                if '"""' in lines[i] or "'''" in lines[i]:
                                    i += 1
                                    break
                                i += 1
                signatures.append("")  # Add blank line after function
                continue
            
            # Capture module-level constants
            if re.match(r'^[A-Z][A-Z_0-9]*\s*=', stripped) and not line.startswith(' '):
                signatures.append(line)
            
            i += 1
        
        return '\n'.join(signatures)

    def _validate_tester_output_format(self, output: str) -> tuple:
        """
        Validate tester output format compliance.
        Returns (is_valid: bool, error_message: str)
        """
        if not output or not output.strip():
            return False, "Output is empty"

        stripped = output.strip()

        # Check 1: Must start with import/from
        if not stripped.startswith(('import ', 'from ')):
            first_line = stripped.split('\n')[0][:60]
            return False, f"Output must start with 'import' or 'from', not: '{first_line}'"

        # Check 2: No markdown code blocks
        if '```' in output:
            return False, "Output contains forbidden markdown code blocks (```)"

        # Check 3: No preamble text or commentary headers
        forbidden_phrases = [
            'here is', "here's", 'test plan', 'rationale',
            'design', '### file:', 'how to', 'expected output'
        ]
        first_100 = stripped[:100].lower()
        for phrase in forbidden_phrases:
            if phrase in first_100:
                return False, f"Output contains forbidden preamble/commentary: '{phrase}'"

        return True, ""  # Valid!

    def _clean_file_content(self, content: str) -> str:
        """Clean individual file content from markdown artifacts

        Extracts code from markdown code blocks if present, stopping at commentary.
        If no code blocks, returns content up to first documentation section.
        """
        lines = content.split('\n')

        # Check if content contains markdown code blocks
        code_block_start = None
        code_block_end = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('```'):
                if code_block_start is None:
                    # Found start of first code block
                    code_block_start = i
                else:
                    # Found end of first code block
                    code_block_end = i
                    break

        # If we found a complete code block, extract only that
        if code_block_start is not None and code_block_end is not None:
            # Extract lines between the markers
            code_lines = lines[code_block_start + 1:code_block_end]
            return '\n'.join(code_lines).strip()

        # If we found a start but no end, extract from start to end of content
        if code_block_start is not None and code_block_end is None:
            code_lines = lines[code_block_start + 1:]
            return '\n'.join(code_lines).strip()

        # No code blocks found - check for markdown headers that indicate commentary
        # and stop before them
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Stop at markdown headers (except our FILE markers which shouldn't appear here)
            if stripped.startswith('###') and 'FILE:' not in stripped:
                break
            # Stop at common documentation section headers
            if stripped.startswith('## ') or (stripped.startswith('### ') and any(
                keyword in stripped.upper() for keyword in [
                    'RATIONALE', 'HOW TO', 'EXPECTED', 'DESIGN',
                    'NOTES', 'EXAMPLE', 'USAGE', 'TEST', 'REQUIREMENTS'
                ]
            )):
                break
            cleaned.append(line)

        return '\n'.join(cleaned).strip()
    
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

    
    def _extract_expected_files(self, architect_result: str) -> set:
        """Extract expected file names from architect's design"""
        files = set()
        
        # Common patterns architects use to specify files
        patterns = [
            r'(?:^|\n)\s*[-*]\s*`?(\w+\.py)`?',  # - file.py or * file.py
            r'(?:^|\n)\s*(\w+\.py)\s*[-:]',       # file.py - description
            r'File:\s*`?(\w+\.py)`?',              # File: name.py
            r'Module:\s*`?(\w+\.py)`?',            # Module: name.py
            r'(\w+\.py)\s*(?:module|file)',        # name.py module/file
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, architect_result, re.IGNORECASE)
            files.update(m.lower() for m in matches)
        
        # Also look for explicit file listings
        # Pattern: something.py (with description)
        file_list_pattern = r'\b(\w+\.py)\b'
        all_py_files = re.findall(file_list_pattern, architect_result)
        
        # Only include if they appear to be deliverables (mentioned multiple times or in specific contexts)
        file_counts = {}
        for f in all_py_files:
            f_lower = f.lower()
            file_counts[f_lower] = file_counts.get(f_lower, 0) + 1
        
        # Add files mentioned multiple times or in deliverables section
        deliverables_section = re.search(r'deliverables?[:\s]+(.*?)(?:\n\n|\Z)', 
                                         architect_result, re.IGNORECASE | re.DOTALL)
        if deliverables_section:
            section_files = re.findall(r'\b(\w+\.py)\b', deliverables_section.group(1))
            files.update(f.lower() for f in section_files)
        
        return files

    def _create_requirements_txt(self, project_dir: str):
        """Generate requirements.txt with PINNED GOLDEN STACK versions."""
        code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")), None)
        if not code_task or not code_task.result:
            return

        code = code_task.result
        imports = set()

        # Scan imports
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Skip relative imports (from . or from .foo)
                if line.startswith('from .'):
                    continue
                parts = line.split()
                if len(parts) > 1:
                    mod = parts[1].split('.')[0].split(',')[0]
                    imports.add(mod)

        # Build set of project-local module names from files in src/
        project_modules = {'src', 'tests'}  # always exclude these package names
        src_dir = os.path.join(project_dir, "src")
        if os.path.isdir(src_dir):
            for fname in os.listdir(src_dir):
                if fname.endswith('.py') and fname != '__init__.py':
                    project_modules.add(fname[:-3])  # strip .py
        # Also extract module names from ### FILE: markers in the code output
        import re as _re
        for m in _re.findall(r'###\s*FILE:\s*(?:\S+/)?(\w+)\.py', code):
            project_modules.add(m)
        # Also check plan for file names
        final_plan = self.state.get("context", {}).get("final_plan", "")
        if final_plan:
            for m in _re.findall(r'\b(\w+)\.py\b', final_plan):
                project_modules.add(m)

        # Complete Python stdlib set (3.10+)
        stdlib = {
            '__future__', '_thread', 'abc', 'aifc', 'argparse', 'array', 'ast',
            'asynchat', 'asyncio', 'asyncore', 'atexit', 'audioop', 'base64',
            'bdb', 'binascii', 'binhex', 'bisect', 'builtins', 'bz2',
            'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code',
            'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
            'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy',
            'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes', 'curses',
            'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis',
            'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno',
            'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch',
            'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass',
            'gettext', 'glob', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac',
            'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp',
            'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
            'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
            'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
            'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
            'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev',
            'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil',
            'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
            'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr',
            'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
            'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
            'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
            'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
            'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep',
            'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig',
            'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile',
            'termios', 'test', 'textwrap', 'threading', 'time', 'timeit',
            'tkinter', 'token', 'tokenize', 'tomllib', 'trace', 'traceback',
            'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing',
            'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
            'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
            'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile',
            'zipimport', 'zlib',
        }

        # GOLDEN STACK: Pinned versions that are guaranteed to work together
        pinned_versions = {
            "fastapi": "fastapi>=0.109.0",
            "uvicorn": "uvicorn>=0.27.0",
            "sqlalchemy": "sqlalchemy>=2.0.25",
            "pydantic": "pydantic>=2.6.0",
            "pydantic_settings": "pydantic-settings>=2.1.0",
            "alembic": "alembic>=1.13.1",
            "psycopg2": "psycopg2-binary>=2.9.9",  # FORCE BINARY to avoid C-compilation errors
            "python_jose": "python-jose[cryptography]>=3.3.0",
            "passlib": "passlib[bcrypt]>=1.7.4",
            "bcrypt": "bcrypt==4.0.1",  # CRITICAL: 4.1.0+ is incompatible with passlib
            "python_multipart": "python-multipart>=0.0.7",
            "python_dotenv": "python-dotenv>=1.0.0",
            "pytest": "pytest>=8.0.0",
            "numpy": "numpy>=1.26.0",
            "pandas": "pandas>=2.2.0",
            "scikit_learn": "scikit-learn>=1.4.0",
            "cryptography": "cryptography>=42.0.0",
            "requests": "requests>=2.31.0",
            "flask": "flask>=3.0.0",
            "jinja2": "jinja2>=3.1.3",
            "click": "click>=8.1.7",
            "aiohttp": "aiohttp>=3.9.0",
            "redis": "redis>=5.0.0",
            "celery": "celery>=5.3.0",
            "httpx": "httpx>=0.27.0",
            "websockets": "websockets>=12.0",
            "pyyaml": "pyyaml>=6.0.1",
            "pillow": "Pillow>=10.2.0",
            "matplotlib": "matplotlib>=3.8.0",
            "scipy": "scipy>=1.12.0",
            "networkx": "networkx>=3.2.0",
            "psutil": "psutil>=5.9.0",
            "beautifulsoup4": "beautifulsoup4>=4.12.0",
            "lxml": "lxml>=5.1.0",
            "paramiko": "paramiko>=3.4.0",
            "boto3": "boto3>=1.34.0",
            "opencv_python": "opencv-python>=4.9.0",
            "torch": "torch>=2.2.0",
            "transformers": "transformers>=4.38.0",
            "sentence_transformers": "sentence-transformers>=2.3.0",
        }

        # Map import names to pinned_versions keys
        package_map = {
            "dotenv": "python_dotenv",
            "multipart": "python_multipart",
            "jose": "python_jose",
            "jwt": "python_jose",
            "psycopg2": "psycopg2",
            "yaml": "pyyaml",
            "sklearn": "scikit_learn",
            "cv2": "opencv_python",
            "PIL": "pillow",
            "bs4": "beautifulsoup4",
            "sentence_transformers": "sentence_transformers",
        }

        final_reqs = set()
        for imp in imports:
            if imp in stdlib:
                continue
            if imp in project_modules:
                continue
            pkg_key = package_map.get(imp, imp)

            # If Pydantic is detected, ensure pydantic-settings is added for V2 support
            if imp == "pydantic" and "BaseSettings" in code:
                final_reqs.add(pinned_versions["pydantic_settings"])

            # Map the import to our pinned version
            if pkg_key in pinned_versions:
                final_reqs.add(pinned_versions[pkg_key])
            else:
                final_reqs.add(pkg_key)

        # Safety override: Always force compatible bcrypt if passlib is involved
        if any("passlib" in r for r in final_reqs):
             final_reqs.add(pinned_versions["bcrypt"])

        if final_reqs:
            req_file = os.path.join(project_dir, "requirements.txt")
            with open(req_file, 'w') as f:
                for req in sorted(final_reqs):
                    f.write(f"{req}\n")
    
    def add_task(self, task: Task):
        """Add a task to the queue"""
        self.task_queue.append(task)
        self.state["history"].append({
            "action": "task_added",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "role": task.assigned_role.value,
            "timestamp": time.time()
        })
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met and not failed)"""
        ready = []
        completed_ids = {t.task_id for t in self.completed_tasks}
        failed_ids = {t.task_id for t in self.completed_tasks if t.status == TaskStatus.FAILED}
        
        for task in self.task_queue:
            if task.status == TaskStatus.PENDING:
                # Check if any dependency failed
                deps_failed = any(dep_id in failed_ids for dep_id in task.dependencies)
                
                if deps_failed:
                    # Mark task as blocked/failed due to dependency failure
                    failed_deps = [dep_id for dep_id in task.dependencies if dep_id in failed_ids]
                    print(f"  ⚠ Skipping {task.task_id}: dependency failed ({', '.join(failed_deps)})")
                    task.status = TaskStatus.FAILED
                    task.error = f"Dependency failed: {', '.join(failed_deps)}"
                    task.completed_at = time.time()
                    self.completed_tasks.append(task)
                    if task in self.task_queue:
                        self.task_queue.remove(task)
                    continue
                
                # All dependencies completed successfully
                if all(dep_id in completed_ids for dep_id in task.dependencies):
                    ready.append(task)
        
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready
    
    # -------------------------------------------------------------------------
    # NEW GENERIC RUNNER
    # -------------------------------------------------------------------------
    def _log_prompt(self, step: str, agent_name: str, payload: Dict, mode: str = ""):
        """Save full prompt payload to projects/<project>/prompt_logs/ as JSON."""
        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir:
            return

        log_dir = os.path.join(project_dir, "prompt_logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        # Sanitize step name: replace path separators to avoid creating subdirectories
        safe_step = step.replace("/", "_").replace("\\", "_")
        filename = f"{timestamp}_{agent_name}_{safe_step}.json"

        log_entry = {
            "step": step,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "system_prompt": payload.get("system_prompt", ""),
            "user_message": payload.get("user_message", ""),
        }
        # Include any extra keys from payload (job_spec, plan_yaml, etc.)
        for key in ("job_spec", "plan_yaml", "code", "draft_plan", "coder_feedback", "revision_feedback", "file_code", "plan_spec"):
            if key in payload:
                log_entry[key] = payload[key]

        log_path = os.path.join(log_dir, filename)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)
            print(f"  [LOG] {filename}")
        except Exception as e:
            print(f"  [LOG] Warning: could not save prompt log: {e}")

    def _build_file_plan_spec_string(self, file_spec) -> str:
        """Convert a FileSpec into a readable plan specification string for the reviewer."""
        parts = [f"File: {file_spec.name}"]
        parts.append(f"Purpose: {file_spec.purpose}")

        if file_spec.requirements:
            parts.append("\nRequirements:")
            for i, req in enumerate(file_spec.requirements, 1):
                parts.append(f"  {i}. {req}")

        if file_spec.exports:
            parts.append("\nRequired Exports:")
            for exp in file_spec.exports:
                exp_type = exp.type if hasattr(exp, 'type') else 'unknown'
                exp_name = exp.name if hasattr(exp, 'name') else str(exp)
                parts.append(f"  - {exp_name} ({exp_type})")
                # Include method signatures so reviewer can verify them
                methods = exp.methods if hasattr(exp, 'methods') else {}
                if methods:
                    for method_name, method_info in methods.items():
                        args = method_info.get('args', [])
                        returns = method_info.get('returns', '')
                        ret_hint = f" -> {returns}" if returns else ""
                        parts.append(f"    signature: {method_name}({', '.join(args)}){ret_hint}")

        if file_spec.imports_from:
            parts.append("\nImports From:")
            for source, names in file_spec.imports_from.items():
                parts.append(f"  - {source}: {', '.join(names)}")

        if file_spec.dependencies:
            parts.append(f"\nDependencies: {', '.join(file_spec.dependencies)}")

        return "\n".join(parts)

    def _build_dependency_code_for_review(self, file_spec, all_files_dict: Dict[str, str]) -> str:
        """Build dependency code context for per-file compliance review.

        Direct dependencies get full code (up to 8K each).
        """
        if not file_spec.dependencies:
            return ""

        parts = []
        for dep_name in file_spec.dependencies:
            if dep_name in all_files_dict:
                dep_code = all_files_dict[dep_name]
                if len(dep_code) <= 8000:
                    parts.append(f"--- {dep_name} (full code) ---\n{dep_code}")
                else:
                    # Use signature extraction for large files
                    try:
                        pe = PlanExecutor.__new__(PlanExecutor)
                        sig = pe._extract_signatures(dep_code)
                        parts.append(f"--- {dep_name} (signatures, {len(dep_code)} chars total) ---\n{sig}")
                    except Exception:
                        parts.append(f"--- {dep_name} (truncated to 4K) ---\n{dep_code[:4000]}")

        return "\n\n".join(parts) if parts else ""

    def _cap_reviewer_issues(self, result_text: str, max_issues: int = 10) -> str:
        """Cap the number of issues in reviewer output to prevent revision prompt flooding.

        Finds numbered issues (lines matching '^\d+\.') in the ISSUES: section
        and truncates to max_issues, appending a note about omitted issues.
        """
        if not result_text:
            return result_text

        # Find ISSUES: section
        issues_match = re.search(r'(ISSUES:\s*\n)(.*?)(\n\n|\Z)', result_text, re.DOTALL | re.IGNORECASE)
        if not issues_match:
            return result_text

        issues_section = issues_match.group(2)
        issue_lines = issues_section.strip().split('\n')

        # Count numbered issues (e.g. "1.", "2.", etc.)
        numbered_indices = []
        for i, line in enumerate(issue_lines):
            if re.match(r'^\s*\d+\.', line):
                numbered_indices.append(i)

        if len(numbered_indices) <= max_issues:
            return result_text

        # Find where the (max_issues+1)th issue starts
        cutoff_idx = numbered_indices[max_issues]
        kept_lines = issue_lines[:cutoff_idx]
        omitted = len(numbered_indices) - max_issues
        kept_lines.append(f"\n[... {omitted} additional issues omitted — fix the above first]")

        # Reconstruct result_text
        new_issues_section = '\n'.join(kept_lines)
        new_result = result_text[:issues_match.start(2)] + new_issues_section + (issues_match.group(3) or '')
        # Append the rest of result_text after the ISSUES section
        rest_start = issues_match.end()
        if rest_start < len(result_text):
            new_result += result_text[rest_start:]

        return new_result

    def _run_external_agent(self, role_name: str, payload: Dict) -> Dict:
        """
        Generic handler to run any external agent script.
        Expects script at: src/{role_name}/{role_name}_agent.py
        """
        import subprocess

        # 1. Resolve Script Path
        # Try relative to the current file first
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(base_dir, "src", role_name, f"{role_name}_agent.py")
        
        # Fallback to current working directory if not found (flat structure)
        if not os.path.exists(script_path):
            script_path = os.path.join("src", role_name, f"{role_name}_agent.py")
            
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Agent script not found: {script_path}")

        # 2. Determine Timeout
        # Use config timeout or default to 600s, plus 30s buffer for Python overhead
        agent_config_timeout = payload.get("config", {}).get("timeout", 600)
        proc_timeout = agent_config_timeout + 30

        # 3. Execute Subprocess
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=proc_timeout
            )
        except subprocess.TimeoutExpired:
            raise Exception(f"Agent {role_name} timed out after {proc_timeout}s")

        # 4. Handle Subprocess Failure
        if result.returncode != 0:
            # Try to read error message from stdout (if our agent printed JSON error) or stderr
            try:
                output = json.loads(result.stdout)
                error_msg = output.get("error", result.stderr)
            except:
                error_msg = result.stderr or f"Exit code {result.returncode}"
            raise Exception(f"Agent {role_name} failed: {error_msg}")

        # 5. Parse Output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON from agent {role_name}: {result.stdout}")


    def _run_smoke_tests(self) -> Dict:
        """Run smoke tests (Pytest + CLI) in Docker if available, otherwise Local."""
        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir: return {}

        results = {}

        # ---------------------------------------------------------------------
        # PATH A: DOCKER EXECUTION (If Sandbox is ready)
        # ---------------------------------------------------------------------
        if self.sandbox and self.sandbox.enabled and self.sandbox.container_running:
            print("   🐳 Running verification in Docker sandbox...")
            
            # 1. Sync files to container
            # We copy the whole project dir to /workspace
            # Note: simplistic copy; for huge projects rsync is better, but this works for agents.
            src_local = os.path.join(project_dir, "src")
            tests_local = os.path.join(project_dir, "tests")
            
            # Create remote dirs
            self.sandbox._exec(["mkdir", "-p", "/workspace/src", "/workspace/tests"])
            
            # Copy src files
            if os.path.exists(src_local):
                for f in os.listdir(src_local):
                    if f.endswith(".py"):
                        self.sandbox.copy_to_container(os.path.join(src_local, f), f"/workspace/src/{f}")
            
            # Copy test files
            if os.path.exists(tests_local):
                for f in os.listdir(tests_local):
                    if f.endswith(".py"):
                        self.sandbox.copy_to_container(os.path.join(tests_local, f), f"/workspace/tests/{f}")

            # Install project requirements if requirements.txt exists
            req_file = os.path.join(project_dir, "requirements.txt")
            if os.path.exists(req_file):
                self.sandbox.copy_to_container(req_file, "/workspace/requirements.txt")
                print("   📦 Installing project dependencies in sandbox...")
                install_result = self.sandbox._exec(
                    ["pip", "install", "--quiet", "-r", "/workspace/requirements.txt"],
                    timeout=180
                )
                if install_result.returncode != 0:
                    print(f"   ⚠ Some dependencies failed to install: {install_result.stderr[:200]}")

            # 2. Run Pytest in Docker
            # We use the sandbox's internal run_pytest or raw _exec
            code, output = self.sandbox.run_pytest("/workspace/tests")
            # Parse the output into a result dict format similar to local
            results["pytest"] = {
                "returncode": 0 if code else 1,
                "stdout": output,
                "stderr": "", 
                "cmd": "pytest (docker)"
            }

            # 3. Run CLI Help in Docker (skip for interactive programs without argparse/click/typer)
            entry_point = self.state.get("project_info", {}).get("entry_point", "src/main.py")
            entry_path = os.path.join(project_dir, entry_point)
            has_cli_framework = False
            try:
                with open(entry_path, "r") as f:
                    entry_code = f.read()
                has_cli_framework = any(fw in entry_code for fw in ("argparse", "click", "typer"))
                # Server frameworks start long-running processes — not suitable for --help
                is_server = any(fw in entry_code for fw in ("uvicorn", "gunicorn", "flask.run", "Django", "FastAPI"))
                if is_server:
                    has_cli_framework = False
            except Exception:
                has_cli_framework = False  # If we can't read, skip rather than hang

            if has_cli_framework:
                # Convert src/main.py -> src.main
                module_path = entry_point.replace("/", ".").replace("\\", ".").rstrip(".py")
                if module_path.endswith("."): module_path = module_path[:-1]
                if not module_path.startswith("src."): module_path = f"src.{module_path}"

                cli_res = self.sandbox._exec(
                    ["python", "-m", module_path, "--help"],
                    timeout=30
                )
                results["cli_help"] = {
                    "returncode": cli_res.returncode,
                    "stdout": cli_res.stdout,
                    "stderr": cli_res.stderr,
                    "cmd": f"python -m {module_path} --help (docker)"
                }
            else:
                print("   ℹ Skipping cli_help: entry point has no argparse/click/typer")
                results["cli_help"] = {
                    "returncode": 0,
                    "stdout": "SKIPPED: interactive program without argparse/click/typer",
                    "stderr": "",
                    "cmd": f"cli_help skipped (no CLI framework in {entry_point})"
                }

            return results

        # ---------------------------------------------------------------------
        # PATH B: LOCAL FALLBACK (Original Logic)
        # ---------------------------------------------------------------------
        print("   ⚠ Docker unavailable/disabled. Running verification locally.")
        
        # Setup Environment
        env = os.environ.copy()
        src_dir = os.path.join(project_dir, "src")
        env["PYTHONPATH"] = f"{project_dir}{os.pathsep}{src_dir}{os.pathsep}{env.get('PYTHONPATH','')}"
        
        # Run Pytest
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=project_dir, env=env, capture_output=True, text=True, timeout=30
            )
            results["pytest"] = {
                "returncode": res.returncode, 
                "stdout": res.stdout, 
                "stderr": res.stderr, 
                "cmd": "pytest -q"
            }
        except Exception as e:
            results["pytest"] = {"returncode": -1, "stderr": str(e), "cmd": "pytest -q"}

        # Run CLI Help (skip for interactive programs without argparse/click/typer)
        try:
            entry_point = self.state.get("project_info", {}).get("entry_point", "src/main.py")
            entry_path = os.path.join(project_dir, entry_point)
            has_cli_framework = False
            try:
                with open(entry_path, "r") as f:
                    entry_code = f.read()
                has_cli_framework = any(fw in entry_code for fw in ("argparse", "click", "typer"))
                # Server frameworks start long-running processes — not suitable for --help
                is_server = any(fw in entry_code for fw in ("uvicorn", "gunicorn", "flask.run", "Django", "FastAPI"))
                if is_server:
                    has_cli_framework = False
            except Exception:
                has_cli_framework = False  # If we can't read, skip rather than hang

            if has_cli_framework:
                module_path = entry_point.replace("/", ".").replace("\\", ".").rstrip(".py")
                if module_path.endswith("."): module_path = module_path[:-1]

                # Patch 3 Logic: Ensure src. prefix
                if os.path.exists(os.path.join(project_dir, "src")) and not module_path.startswith("src."):
                     module_path = f"src.{module_path}"

                res = subprocess.run(
                    [sys.executable, "-m", module_path, "--help"],
                    cwd=project_dir, env=env, capture_output=True, text=True, timeout=30
                )
                results["cli_help"] = {
                    "returncode": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "cmd": f"python -m {module_path} --help"
                }
            else:
                print("   ℹ Skipping cli_help: entry point has no argparse/click/typer")
                results["cli_help"] = {
                    "returncode": 0,
                    "stdout": "SKIPPED: interactive program without argparse/click/typer",
                    "stderr": "",
                    "cmd": f"cli_help skipped (no CLI framework in {entry_point})"
                }
        except Exception as e:
            results["cli_help"] = {"returncode": -1, "stderr": str(e), "cmd": "cli_help check"}
            
        return results
    # -------------------------------------------------------------------------
    # UPDATED EXECUTE_TASK
    # -------------------------------------------------------------------------
    def _diag(self, msg: str):
        """Append a diagnostic line to /tmp/swarm_diag.log"""
        try:
            with open("/tmp/swarm_diag.log", "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    def execute_task(self, task: Task) -> Task:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS

        self._diag(f"execute_task ENTER: task_id={task.task_id} task_type={task.task_type} role={task.assigned_role} role_type={type(task.assigned_role).__name__}")

        try:
# --- PRIORITY 0: PLAN EXECUTION ---
            if task.task_type == "plan_execution":
                plan_yaml = self.state["context"].get("plan_yaml", "")
                if not plan_yaml: raise Exception("No YAML plan found")
                task.result = execute_plan_task(self, task, plan_yaml, task.metadata.get("user_request",""), self.state["context"].get("job_scope", ""))
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self._update_context(task)
                return task
            
            # --- COLLABORATIVE WORKFLOW TASK HANDLERS ---

            # DRAFT_PLAN: Architect produces draft plan from job_spec (with retry on validation errors)
            if task.task_type == "draft_plan":
                job_spec = self.state["context"].get("job_spec", "")
                if not job_spec:
                    job_spec = self.state["context"].get("job_scope", "")
                architect_config = self.executor._get_agent_config(AgentRole.ARCHITECT)
                max_retries = self.state.get("max_iterations", 3)
                validation_error = None

                for attempt in range(max_retries):
                    payload = {
                        "mode": "draft",
                        "job_spec": job_spec,
                        "config": {
                            "model_url": architect_config.get("url", "http://localhost:1233/v1"),
                            "model_name": architect_config.get("model", "local-model"),
                            "api_type": architect_config.get("api_type", "openai"),
                            "temperature": 0.6,
                            "max_tokens": architect_config.get("max_tokens", 25000),
                            "timeout": architect_config.get("timeout", 600)
                        }
                    }
                    if validation_error:
                        payload["validation_error"] = validation_error

                    log_step = f"DRAFT_RETRY{attempt}" if attempt > 0 else "DRAFT"
                    self._log_prompt(log_step, "ARCHITECT", {"system_prompt": "DRAFT_PLAN_PROMPT", "user_message": job_spec, "job_spec": job_spec, "validation_error": validation_error}, mode="draft")

                    try:
                        output = self._run_external_agent("architect", payload)
                        if output.get("status") == "error":
                            error_msg = output.get("error", "")
                            # Check if this is a validation error we can retry
                            if "Validation error" in error_msg and attempt < max_retries - 1:
                                validation_error = error_msg
                                print(f"  ⚠ Draft plan validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                                continue
                            raise Exception(error_msg)

                        plan_yaml = output.get("plan_yaml") or output.get("result", "")
                        self.state["context"]["draft_plan"] = plan_yaml
                        task.result = plan_yaml
                        task.status = TaskStatus.COMPLETED
                        if attempt > 0:
                            print(f"  ✓ Draft plan produced on retry {attempt + 1} ({len(plan_yaml)} chars)")
                        else:
                            print(f"  ✓ Draft plan produced ({len(plan_yaml)} chars)")
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "Validation error" in error_msg and attempt < max_retries - 1:
                            validation_error = error_msg
                            print(f"  ⚠ Draft plan validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                            continue
                        task.status = TaskStatus.FAILED
                        task.error = error_msg
                        break

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # PLAN_REVIEW: Coder reviews draft plan (no code)
            if task.task_type == "plan_review":
                draft_plan = self.state["context"].get("draft_plan", "")
                job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))
                coder_config = self.executor._get_agent_config(AgentRole.CODER)

                payload = {
                    "mode": "plan_review",
                    "plan_yaml": draft_plan,
                    "job_spec": job_spec,
                    "config": {
                        "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                        "model_name": coder_config.get("model", "local-model"),
                        "api_type": coder_config.get("api_type", "openai"),
                        "temperature": 0.3,
                        "max_tokens": coder_config.get("max_tokens", 25000),
                        "timeout": coder_config.get("timeout", 1200)
                    }
                }

                self._log_prompt("PLAN_REVIEW", "CODER", {"system_prompt": "PLAN_REVIEW_PROMPT", "user_message": draft_plan, "plan_yaml": draft_plan, "job_spec": job_spec}, mode="plan_review")

                try:
                    output = self._run_external_agent("coder", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    review_text = output.get("review") or output.get("result", "")
                    self.state["context"]["plan_review"] = review_text
                    task.result = review_text
                    task.status = TaskStatus.COMPLETED
                    print(f"  ✓ Plan review complete")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # FINALIZE_PLAN: Architect incorporates coder feedback (with retry on validation errors)
            if task.task_type == "finalize_plan":
                draft_plan = self.state["context"].get("draft_plan", "")
                plan_review = self.state["context"].get("plan_review", "")
                job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))
                architect_config = self.executor._get_agent_config(AgentRole.ARCHITECT)
                max_retries = self.state.get("max_iterations", 3)
                validation_error = None

                for attempt in range(max_retries):
                    payload = {
                        "mode": "finalize",
                        "draft_plan": draft_plan,
                        "coder_feedback": plan_review,
                        "job_spec": job_spec,
                        "config": {
                            "model_url": architect_config.get("url", "http://localhost:1233/v1"),
                            "model_name": architect_config.get("model", "local-model"),
                            "api_type": architect_config.get("api_type", "openai"),
                            "temperature": 0.6,
                            "max_tokens": architect_config.get("max_tokens", 25000),
                            "timeout": architect_config.get("timeout", 600)
                        }
                    }
                    if validation_error:
                        payload["validation_error"] = validation_error

                    log_step = f"FINALIZE_RETRY{attempt}" if attempt > 0 else "FINALIZE"
                    self._log_prompt(log_step, "ARCHITECT", {"system_prompt": "FINALIZE_PROMPT", "user_message": plan_review, "draft_plan": draft_plan, "coder_feedback": plan_review, "validation_error": validation_error}, mode="finalize")

                    try:
                        output = self._run_external_agent("architect", payload)
                        if output.get("status") == "error":
                            error_msg = output.get("error", "")
                            if "Validation error" in error_msg and attempt < max_retries - 1:
                                validation_error = error_msg
                                print(f"  ⚠ Finalize plan validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                                continue
                            raise Exception(error_msg)

                        final_plan = output.get("plan_yaml") or output.get("result", "")
                        self.state["context"]["final_plan"] = final_plan
                        self.state["context"]["plan_yaml"] = final_plan  # For backward compat
                        task.result = final_plan
                        task.status = TaskStatus.COMPLETED
                        self._set_project_type_from_plan(final_plan)
                        if attempt > 0:
                            print(f"  ✓ Final plan produced on retry {attempt + 1} ({len(final_plan)} chars)")
                        else:
                            print(f"  ✓ Final plan produced ({len(final_plan)} chars)")
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "Validation error" in error_msg and attempt < max_retries - 1:
                            validation_error = error_msg
                            print(f"  ⚠ Finalize plan validation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                            continue
                        task.status = TaskStatus.FAILED
                        task.error = error_msg
                        break

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # BUILD_FROM_PLAN: File-by-file build via PlanExecutor
            if task.task_type == "build_from_plan":
                final_plan = self.state["context"].get("final_plan", "")
                job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))
                user_request = self.state["context"].get("user_request", "")

                if PLAN_EXECUTOR_AVAILABLE:
                    # Use PlanExecutor for file-by-file build with per-file compliance,
                    # revision, anti-stub detection, and proper __init__.py generation
                    try:
                        pe = PlanExecutor(self.executor, self.config)
                        # Wire up prompt logging
                        project_dir = self.state["project_info"].get("project_dir")
                        if project_dir:
                            pe.prompt_log_dir = os.path.join(project_dir, "prompt_logs")
                        result = pe.execute(final_plan, user_request=user_request, job_scope=job_spec)

                        combined_output = result.get("combined_output", "")
                        if not combined_output:
                            raise Exception("PlanExecutor produced no output")

                        self.state["context"]["latest_code"] = combined_output
                        task.result = combined_output
                        task.status = TaskStatus.COMPLETED

                        n_success = sum(1 for s in result.get("status", {}).values() if s == "completed")
                        n_total = len(result.get("status", {}))
                        print(f"  ✓ Code built file-by-file ({n_success}/{n_total} files, {len(combined_output)} chars)")

                        # Save files to disk immediately after build
                        try:
                            self._save_project_outputs()
                        except Exception as save_err:
                            print(f"  ⚠ Immediate save failed: {save_err}")

                    except Exception as e:
                        print(f"  ⚠ PlanExecutor failed ({e}), falling back to monolithic build")
                        # Fall through to monolithic fallback below
                        PLAN_EXECUTOR_AVAILABLE_FOR_BUILD = False
                    else:
                        PLAN_EXECUTOR_AVAILABLE_FOR_BUILD = True

                    if not PLAN_EXECUTOR_AVAILABLE_FOR_BUILD:
                        # Monolithic fallback
                        coder_config = self.executor._get_agent_config(AgentRole.CODER)
                        payload = {
                            "mode": "build",
                            "plan_yaml": final_plan,
                            "job_spec": job_spec,
                            "config": {
                                "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                                "model_name": coder_config.get("model", "local-model"),
                                "api_type": coder_config.get("api_type", "openai"),
                                "temperature": 0.2,
                                "max_tokens": coder_config.get("max_tokens", 25000),
                                "timeout": coder_config.get("timeout", 1200)
                            }
                        }
                        self._log_prompt("BUILD", "CODER", {"system_prompt": "BUILD_PROMPT", "user_message": final_plan, "plan_yaml": final_plan, "job_spec": job_spec}, mode="build")
                        try:
                            output = self._run_external_agent("coder", payload)
                            if output.get("status") == "error":
                                raise Exception(output.get("error"))
                            result_text = output.get("result", "")
                            result_text = self._clean_code_output(result_text)
                            self.state["context"]["latest_code"] = result_text
                            task.result = result_text
                            task.status = TaskStatus.COMPLETED
                            print(f"  ✓ Code built monolithic ({len(result_text)} chars)")
                        except Exception as e:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                else:
                    # No PlanExecutor available at all — monolithic build
                    coder_config = self.executor._get_agent_config(AgentRole.CODER)
                    payload = {
                        "mode": "build",
                        "plan_yaml": final_plan,
                        "job_spec": job_spec,
                        "config": {
                            "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                            "model_name": coder_config.get("model", "local-model"),
                            "api_type": coder_config.get("api_type", "openai"),
                            "temperature": 0.2,
                            "max_tokens": coder_config.get("max_tokens", 25000),
                            "timeout": coder_config.get("timeout", 1200)
                        }
                    }
                    self._log_prompt("BUILD", "CODER", {"system_prompt": "BUILD_PROMPT", "user_message": final_plan, "plan_yaml": final_plan, "job_spec": job_spec}, mode="build")
                    try:
                        output = self._run_external_agent("coder", payload)
                        if output.get("status") == "error":
                            raise Exception(output.get("error"))
                        result_text = output.get("result", "")
                        result_text = self._clean_code_output(result_text)
                        self.state["context"]["latest_code"] = result_text
                        task.result = result_text
                        task.status = TaskStatus.COMPLETED
                        print(f"  ✓ Code built monolithic ({len(result_text)} chars)")
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # COMPLIANCE_REVIEW: Reviewer checks code against plan (FILE-BY-FILE)
            if task.task_type == "compliance_review":
                final_plan = self.state["context"].get("final_plan", "")
                latest_code = self.state["context"].get("latest_code", "")
                reviewer_config = self.executor._get_agent_config(AgentRole.REVIEWER)
                job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))

                # Build reviewer list: primary + fallback (reviewer_2) if configured
                reviewer_configs = [("primary", reviewer_config)]
                try:
                    multi_model = self.config.get('model_config', {}).get('multi_model', {})
                    rc2 = multi_model.get('reviewer_2', {})
                    if rc2 and rc2.get("url"):
                        reviewer_configs.append(("fallback", rc2))
                        print(f"  → Dual reviewer: {reviewer_config.get('model', '?')} + {rc2.get('model', '?')}")
                except Exception:
                    pass

                # Try file-by-file compliance review
                use_file_by_file = False
                parsed_plan = None
                all_files = {}

                if PLAN_EXECUTOR_AVAILABLE and final_plan:
                    try:
                        pe = PlanExecutor.__new__(PlanExecutor)
                        pe.plan = None
                        pe.completed_files = {}
                        parsed_plan = pe.parse_plan(final_plan)
                        all_files = self._parse_multi_file_output(latest_code)
                        if parsed_plan and parsed_plan.files and all_files:
                            use_file_by_file = True
                            print(f"  → File-by-file compliance review ({len(parsed_plan.files)} files)")
                    except Exception as e:
                        print(f"  → Plan parse failed ({e}), using monolithic compliance")

                if use_file_by_file:
                    try:
                        per_file_results = {}
                        failed_files = []
                        all_results_text = []

                        for file_spec in parsed_plan.files:
                            fname = file_spec.name
                            file_code = all_files.get(fname, "")

                            if not file_code:
                                per_file_results[fname] = {"status": "FAIL", "result": "File missing from generated code"}
                                failed_files.append(fname)
                                all_results_text.append(f"- {fname}: FAIL — File missing from generated code")
                                print(f"    {fname}: FAIL (missing)")
                                continue

                            # Try each reviewer (primary, then fallback) per file.
                            # If primary crashes or errors, fallback gets a chance before marking FAIL.
                            plan_spec_str = self._build_file_plan_spec_string(file_spec)
                            dep_code = self._build_dependency_code_for_review(file_spec, all_files)

                            result_text = None
                            used_reviewer = "primary"

                            for reviewer_label, rc in reviewer_configs:
                                try:
                                    payload = {
                                        "mode": "compliance_file",
                                        "file_name": fname,
                                        "file_code": file_code,
                                        "plan_spec": plan_spec_str,
                                        "dependency_code": dep_code,
                                        "job_spec": job_spec,
                                        "config": {
                                            "model_url": rc.get("url", "http://localhost:1233/v1"),
                                            "model_name": rc.get("model", "local-model"),
                                            "api_type": rc.get("api_type", "openai"),
                                            "temperature": 0.3,
                                            "max_tokens": rc.get("max_tokens", 12000),
                                            "timeout": rc.get("timeout", 600)
                                        }
                                    }

                                    self._log_prompt(f"COMPLIANCE_FILE_{fname}_{reviewer_label}", "REVIEWER", {
                                        "system_prompt": "FILE_COMPLIANCE_PROMPT",
                                        "user_message": f"Review {fname}",
                                        "file_code": file_code[:2000],
                                        "plan_spec": plan_spec_str,
                                        "reviewer_model": rc.get("model", "?")
                                    }, mode="compliance_file")

                                    output = self._run_external_agent("reviewer", payload)

                                    if output.get("status") == "error":
                                        print(f"    {fname}: {reviewer_label} ({rc.get('model','?')}) error, trying next...")
                                        continue

                                    result_text = output.get("result", "")
                                    if not result_text or not result_text.strip():
                                        print(f"    {fname}: {reviewer_label} ({rc.get('model','?')}) empty response, trying next...")
                                        result_text = None
                                        continue

                                    used_reviewer = reviewer_label
                                    break  # Got a response — stop trying reviewers

                                except Exception as rev_err:
                                    print(f"    {fname}: {reviewer_label} ({rc.get('model','?')}) crashed ({rev_err}), trying next...")
                                    continue

                            # All reviewers failed for this file
                            if result_text is None:
                                per_file_results[fname] = {"status": "FAIL", "result": "All reviewers failed for this file"}
                                failed_files.append(fname)
                                all_results_text.append(f"- {fname}: FAIL (all reviewers failed)")
                                print(f"    {fname}: FAIL (all reviewers failed)")
                                continue

                            # Cap issues to 10 per file to prevent revision prompt flooding
                            result_text = self._cap_reviewer_issues(result_text, max_issues=10)

                            reviewer_tag = f" [{used_reviewer}]" if used_reviewer != "primary" else ""
                            self._log_prompt(f"COMPLIANCE_FILE_{fname}_RESULT", "REVIEWER", {
                                "system_prompt": "",
                                "user_message": result_text,
                                "reviewer_used": used_reviewer
                            }, mode="compliance_file_result")

                            # Parse STATUS from result — require explicit PASS, never default to it
                            result_upper = result_text.upper()
                            if "STATUS: FAIL" in result_upper or "STATUS: NEEDS_REVISION" in result_upper:
                                per_file_results[fname] = {"status": "FAIL", "result": result_text}
                                failed_files.append(fname)
                                all_results_text.append(f"- {fname}: FAIL{reviewer_tag}\n{result_text}")
                                print(f"    {fname}: FAIL{reviewer_tag}")
                            elif "STATUS: PASS" in result_upper:
                                # Coherence check: does the review actually mention the file?
                                base_name = fname.rsplit('/', 1)[-1].replace('.py', '')
                                export_names = [e.name for e in (file_spec.exports or [])]
                                coherence_terms = [fname, base_name] + export_names
                                mentions_file = any(term.lower() in result_text.lower() for term in coherence_terms if term)
                                if mentions_file:
                                    per_file_results[fname] = {"status": "PASS", "result": result_text}
                                    all_results_text.append(f"- {fname}: PASS{reviewer_tag}")
                                    print(f"    {fname}: PASS{reviewer_tag}")
                                else:
                                    per_file_results[fname] = {"status": "FAIL", "result": f"Review output did not reference the file or its exports — likely hallucinated. Raw output:\n{result_text[:500]}"}
                                    failed_files.append(fname)
                                    all_results_text.append(f"- {fname}: FAIL (hallucinated review){reviewer_tag}")
                                    print(f"    {fname}: FAIL (hallucinated review — no mention of {fname} or exports){reviewer_tag}")
                            else:
                                # No explicit STATUS found — treat as FAIL
                                per_file_results[fname] = {"status": "FAIL", "result": f"Reviewer did not produce a STATUS: PASS or STATUS: FAIL. Raw output:\n{result_text[:500]}"}
                                failed_files.append(fname)
                                all_results_text.append(f"- {fname}: FAIL (no STATUS found){reviewer_tag}")
                                print(f"    {fname}: FAIL (no STATUS in reviewer output){reviewer_tag}")

                        # Assemble overall result
                        task.result = "FILE-BY-FILE COMPLIANCE REVIEW:\n\n" + "\n\n".join(all_results_text)

                        if failed_files:
                            print(f"  ⚠ Compliance review: {len(failed_files)}/{len(parsed_plan.files)} files NEED REVISION")
                        else:
                            print(f"  ✓ Compliance review: ALL {len(parsed_plan.files)} files APPROVED")

                        # Always run AST integration check — catches cross-file bugs
                        # (wrong attribute names, constructor arg mismatches, phantom imports)
                        # regardless of whether LLM review passed or failed
                        try:
                            integration_issues = self._run_integration_check(all_files, parsed_plan)
                            if integration_issues:
                                total_issues = sum(len(v) for v in integration_issues.values())
                                print(f"  ⚠ Integration check found {total_issues} cross-file issues in {len(integration_issues)} files")
                                for ifname, iissues in integration_issues.items():
                                    for iss in iissues[:2]:
                                        print(f"    → {ifname}: {iss}")
                                for ifname, iissues in integration_issues.items():
                                    issue_text = "\n".join(f"- {iss}" for iss in iissues[:10])
                                    if ifname in per_file_results:
                                        # Merge with existing LLM feedback — don't overwrite it
                                        per_file_results[ifname]["status"] = "FAIL"
                                        per_file_results[ifname]["result"] += f"\n\nINTEGRATION ISSUES (cross-file):\n{issue_text}"
                                    else:
                                        per_file_results[ifname] = {
                                            "status": "FAIL",
                                            "result": f"STATUS: FAIL\nISSUES (cross-file integration):\n{issue_text}"
                                        }
                                    if ifname not in failed_files:
                                        failed_files.append(ifname)
                                task.result += f"\n\nINTEGRATION CHECK: {total_issues} cross-file issues in {len(integration_issues)} files"
                                print(f"  ⚠ Integration check: {len(failed_files)} total files will be revised")
                            else:
                                print(f"  ✓ Integration check: no cross-file issues found")
                        except Exception as ie:
                            print(f"  ⚠ Integration check failed ({ie}), skipping")

                        # Set revision metadata if any files failed (from LLM review or integration check)
                        if failed_files:
                            task.metadata["needs_revision"] = True
                            task.metadata["final_plan"] = final_plan
                            task.metadata["per_file_results"] = per_file_results
                            task.metadata["failed_files"] = failed_files

                        task.status = TaskStatus.COMPLETED

                    except Exception as e:
                        print(f"  ⚠ File-by-file compliance failed ({e}), falling back to monolithic")
                        use_file_by_file = False

                # Fallback: monolithic compliance review (original behavior)
                if not use_file_by_file:
                    self._log_prompt("COMPLIANCE", "REVIEWER", {"system_prompt": "COMPLIANCE_PROMPT", "user_message": "(plan+code)", "plan_yaml": final_plan, "code": latest_code[:2000]}, mode="compliance")

                    # Try each reviewer (primary, then fallback)
                    result_text = None
                    last_error = None
                    for reviewer_label, rc in reviewer_configs:
                        try:
                            payload = {
                                "mode": "compliance",
                                "plan_yaml": final_plan,
                                "code": latest_code,
                                "config": {
                                    "model_url": rc.get("url", "http://localhost:1233/v1"),
                                    "model_name": rc.get("model", "local-model"),
                                    "api_type": rc.get("api_type", "openai"),
                                    "temperature": 0.3,
                                    "max_tokens": rc.get("max_tokens", 12000),
                                    "timeout": rc.get("timeout", 600)
                                }
                            }
                            output = self._run_external_agent("reviewer", payload)
                            if output.get("status") == "error":
                                print(f"  ⚠ Monolithic {reviewer_label} ({rc.get('model','?')}) error, trying next...")
                                last_error = output.get("error", "Agent error")
                                continue

                            result_text = output.get("result", "")
                            if not result_text or not result_text.strip():
                                print(f"  ⚠ Monolithic {reviewer_label} ({rc.get('model','?')}) empty response, trying next...")
                                result_text = None
                                continue
                            break
                        except Exception as e:
                            print(f"  ⚠ Monolithic {reviewer_label} ({rc.get('model','?')}) crashed ({e}), trying next...")
                            last_error = str(e)
                            continue

                    if result_text:
                        task.result = result_text
                        if "NEEDS_REVISION" in result_text.upper() or "STATUS: NEEDS_REVISION" in result_text.upper():
                            task.metadata["needs_revision"] = True
                            task.metadata["final_plan"] = final_plan
                            print(f"  ⚠ Compliance review: NEEDS_REVISION")
                        else:
                            print(f"  ✓ Compliance review: APPROVED")
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = last_error or "All reviewers failed"

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # Debug
            if task.assigned_role == AgentRole.DEBUGGER:
                if task.task_type == "semantic_audit":
                    report = self._run_semantic_audit()
                    task.metadata["semantic_audit"] = report
                    task.result = json.dumps(report, indent=2)

                    if report.get("status") == "pass":
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = f"Semantic audit failed ({report.get('count', 0)} finding(s))"
                    return task
                
                # BUG DIAGNOSIS - analyze code and diagnose root cause
                elif task.task_type == "bug_diagnosis":
                    system_prompt = self._get_system_prompt(AgentRole.DEBUGGER, "bug_diagnosis")
                    user_request = task.metadata.get("user_request", "")
                    bug_description = task.metadata.get("bug_description", "")
                    
                    user_message = f"""Analyze this bug and diagnose the root cause.

BUG DESCRIPTION:
{bug_description}

PROJECT CONTEXT:
{user_request}

Provide:
1. Root cause analysis - what is causing the bug
2. Affected files/functions
3. Recommended fix approach
4. Any related issues that might also need attention

Be specific and reference actual code from the project."""
                    
                    try:
                        result = self.executor.execute_agent(
                            role=AgentRole.DEBUGGER,
                            system_prompt=system_prompt,
                            user_message=user_message
                        )
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        
                        # Store diagnosis in context for coder task
                        self.state["context"]["bug_diagnosis"] = result
                        
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                    
                    task.completed_at = time.time()
                    return task

            # fallback: if you have other debugger tasks that should use the LLM,
# --- VERIFIER TASK (External: Verifier) ---
            if task.task_type == "verification" and task.assigned_role == AgentRole.VERIFIER:
                try:
                    self._save_project_outputs()
                except Exception as e:
                    print(f"Warning: save failed before verify: {e}")
                
                # RUN SMOKE TESTS LOCALLY
                smoke_tests = self._run_smoke_tests()
                task.metadata["smoke_tests"] = smoke_tests
                
                # Smart smoke test failure check
                def _smoke_failed(results):
                    failing = []
                    project_type = self.state["project_info"].get("project_type", "cli")
                    for name, info in (results or {}).items():
                        rc = info.get("returncode", None)
                        # Skip CLI checks for non-CLI projects (libraries, services)
                        if name in ("cli_usage", "cli_help"):
                            if project_type in ("library", "service"):
                                continue
                            if rc in (0, 2):
                                continue
                        if rc != 0:
                            failing.append(f"{name} (rc={rc})")
                    return (len(failing) > 0), failing

                failed, failing_list = _smoke_failed(smoke_tests)

                if failed:
                    # Collect error details for revision feedback
                    error_details = []
                    for name, info in smoke_tests.items():
                        rc = info.get("returncode", None)
                        if rc != 0:
                            detail = f"Test '{name}' failed (rc={rc})"
                            detail += f"\nCommand: {info.get('cmd')}"
                            stdout = (info.get("stdout") or "").strip()
                            stderr = (info.get("stderr") or "").strip()
                            if stdout:
                                detail += f"\nstdout:\n{stdout[:3000]}"
                            if stderr:
                                detail += f"\nstderr:\n{stderr[:3000]}"
                            error_details.append(detail)

                    error_feedback = "\n\n".join(error_details)

                    print("\n" + "=" * 80)
                    print("❌ SMOKE TESTS FAILED")
                    print("=" * 80)
                    for item in failing_list:
                        print(f" - {item}")
                    print(error_feedback[:4000])
                    print("=" * 80)

                    # Find the code task to check revision count
                    code_task = next((t for t in reversed(self.completed_tasks)
                                      if t.task_type in ("coding", "plan_execution", "build_from_plan")), None)

                    if code_task and code_task.revision_count < code_task.max_revisions:
                        print(f"\n🔄 Triggering revision {code_task.revision_count + 1}/{code_task.max_revisions} based on smoke test errors")

                        revision_metadata = {
                            "revision_feedback": (
                                "SMOKE TESTS FAILED. Fix these errors:\n\n"
                                + error_feedback
                                + "\n\nIMPORTANT: Fix the EXACT errors shown above. "
                                "Do NOT change code that is working. Only fix what is broken."
                            ),
                            "original_code": code_task.result,
                            "user_request": self.state["context"].get("user_request", ""),
                            "is_revision": True,
                        }
                        if self.state["context"].get("final_plan"):
                            revision_metadata["final_plan"] = self.state["context"]["final_plan"]

                        revision_task = Task(
                            task_id=f"T_revision_smoke_{code_task.revision_count + 1}",
                            task_type="revision",
                            description="Fix code based on smoke test failures",
                            assigned_role=AgentRole.CODER,
                            status=TaskStatus.PENDING,
                            priority=10,
                            metadata=revision_metadata
                        )
                        revision_task.revision_count = code_task.revision_count + 1

                        self.execute_task(revision_task)

                        if revision_task.status == TaskStatus.COMPLETED and revision_task.result:
                            # Merge revised code back
                            original_files = self._parse_multi_file_output(code_task.result or "")
                            revised_files = self._parse_multi_file_output(revision_task.result)

                            if not revised_files and revision_task.result.strip():
                                # Revision output has no ### FILE: markers — try to
                                # match it to original files or treat as full replacement
                                raw = revision_task.result.strip()
                                # Clean markdown fences if present
                                raw = re.sub(r'^```(?:python)?\s*\n', '', raw)
                                raw = re.sub(r'\n```\s*$', '', raw)

                                if original_files:
                                    # If original had multiple files, wrap raw output
                                    # using original filenames as markers and re-parse
                                    # Check if raw output contains any original filenames as hints
                                    matched_any = False
                                    for fname in original_files:
                                        if fname in raw or fname.replace('.py', '') in raw:
                                            matched_any = True
                                            break

                                    if not matched_any and len(original_files) == 1:
                                        # Single-file project: use the one filename
                                        only_name = list(original_files.keys())[0]
                                        revised_files = {only_name: raw}
                                    else:
                                        # Multi-file: replace entire code_task result
                                        # and let _save_project_outputs re-parse
                                        code_task.result = raw
                                        revised_files = None  # signal to skip merge
                                else:
                                    # No original files parsed either — store raw
                                    code_task.result = raw
                                    revised_files = None

                            if revised_files:
                                merged = {**original_files, **revised_files}
                                # Rebuild combined output
                                merged_parts = []
                                for fname, content in merged.items():
                                    merged_parts.append(f"### FILE: {fname}\n```python\n{content}\n```")
                                code_task.result = "\n\n".join(merged_parts)

                            # Update state and re-save
                            code_task.revision_count = revision_task.revision_count
                            self._update_context(code_task)
                            self._save_project_outputs()

                            # Re-run smoke tests
                            print("\n🔁 Re-running smoke tests after revision...")
                            smoke_tests = self._run_smoke_tests()
                            task.metadata["smoke_tests"] = smoke_tests
                            failed, failing_list = _smoke_failed(smoke_tests)

                            if not failed:
                                print("✓ Smoke tests pass after revision!")
                                # Fall through to external agent verification below
                            else:
                                print(f"❌ Smoke tests still failing after revision: {', '.join(failing_list)}")
                                task.status = TaskStatus.FAILED
                                task.error = "Smoke tests failed after revision: " + ", ".join(failing_list)
                                task.completed_at = time.time()
                                return task
                        else:
                            print("  ⚠ Revision task failed")
                            task.status = TaskStatus.FAILED
                            task.error = "Smoke tests failed, revision failed"
                            task.completed_at = time.time()
                            return task
                    else:
                        if code_task:
                            print(f"\n⚠ Max revisions ({code_task.max_revisions}) reached, no more retries")
                        task.status = TaskStatus.FAILED
                        task.error = "Smoke tests failed: " + ", ".join(failing_list)
                        task.completed_at = time.time()
                        return task

                # Continue to External Agent (smoke tests passed)
                system_prompt, user_message = self._get_verifier_prompt(task)
                ver_config = self.executor._get_agent_config(AgentRole.VERIFIER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": ver_config.get("url", "http://localhost:1233/v1"),
                        "model_name": ver_config.get("model", "local-model"),
                        "api_type": ver_config.get("api_type", "openai"),
                        "temperature": 0.3,
                        "max_tokens": 3000,
                        "timeout": ver_config.get("timeout", 600)
                    }
                }

                try:
                    output = self._run_external_agent("verifier", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    result = output.get("result", "")
                    task.result = result

                    # Parse verification result
                    resp_upper = (result or "").upper()
                    
                    # Check for explicit FAIL first
                    explicit_fail = "VERIFICATION: FAIL" in resp_upper
                    
                    # Check for pass conditions (PASS, WARN with APPROVE, or just APPROVE)
                    explicit_pass = "VERIFICATION: PASS" in resp_upper
                    warn_with_approve = ("VERIFICATION: WARN" in resp_upper and "RECOMMENDATION: APPROVE" in resp_upper)
                    recommend_approve = "RECOMMENDATION: APPROVE" in resp_upper and not explicit_fail
                    
                    passed = (explicit_pass or warn_with_approve or recommend_approve) and not explicit_fail

                    if not passed:
                        print("\n" + "=" * 80)
                        print("⚠️ VERIFICATION FAILED")
                        print("=" * 80)
                        print(result)
                        print("=" * 80)
                        task.status = TaskStatus.FAILED
                        task.error = "Verifier returned FAIL or did not approve"
                    else:
                        if warn_with_approve:
                            print("\n✓ Verification passed with warnings - project approved")
                        else:
                            print("\n✓ Verification passed - project ready for delivery")
                        task.status = TaskStatus.COMPLETED

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                return task

            # --- 2. CLARIFICATION TASK (External: Clarifier) ---
            if task.task_type == "clarification":
                user_request = task.metadata.get('user_request', '')
                clarifier_config = self.executor._get_agent_config(AgentRole.CLARIFIER)
                
                payload = {
                    "user_request": task.metadata.get("user_request", ""),
                    "config": {
                        "model_url": clarifier_config.get("url", "http://localhost:1233/v1"),
                        "model_name": clarifier_config.get("model", "local-model"),
                        "api_type": clarifier_config.get("api_type", "openai"),
                        "temperature": 0.7,
                        "max_tokens": clarifier_config.get("max_tokens", 25000),
                        "timeout": clarifier_config.get("timeout", 300)
                    }
                }

                try:
                    # Generic Call
                    output = self._run_external_agent("clarifier", payload)
                    
                    # Clarifier specific response handling
                    if output.get("status") == "clear":
                        task.result = "STATUS: CLEAR - All requirements are well-defined."
                    else:
                        questions = output.get("questions", [])
                        task.result = "CLARIFYING QUESTIONS:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                    
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                return task

            # --- 3. ARCHITECT TASK ---
            if task.assigned_role == AgentRole.ARCHITECT:
                if task.task_type == "architecture_plan":
                    if not PLAN_EXECUTOR_AVAILABLE:
                        raise Exception("Plan executor not available - ensure plan_executor.py is in the same directory")
                system_prompt = ARCHITECT_PLAN_SYSTEM_PROMPT
                base_user_message = get_architect_plan_prompt(
                    task.metadata.get('user_request', ''),
                    self.state["context"].get('clarification', '')
                    )
                architect_config = self.executor._get_agent_config(AgentRole.ARCHITECT)

                max_architect_retries = 3
                last_error = None
                for attempt in range(max_architect_retries):
                    user_message = base_user_message
                    if last_error:
                        user_message += (
                            f"\n\n{'='*60}\n"
                            f"YOUR PREVIOUS PLAN FAILED VALIDATION (attempt {attempt + 1}/{max_architect_retries}):\n"
                            f"{last_error}\n"
                            f"{'='*60}\n"
                            f"Fix this error. Every file's imports_from must only reference "
                            f"names that appear in the source file's exports list.\n"
                        )
                        print(f"  🔄 Architect retry {attempt + 1}/{max_architect_retries}: {last_error[:120]}")

                    payload = {
                            "system_prompt": system_prompt,
                            "user_message": user_message,
                            "config": {
                                "model_url": architect_config.get("url", "http://localhost:1233/v1"),
                                "model_name": architect_config.get("model", "local-model"),
                                "api_type": architect_config.get("api_type", "openai"),
                                "temperature": 0.6,
                                "max_tokens": 3000,
                                "timeout": architect_config.get("timeout", 600)
                            }
                        }

                    output = self._run_external_agent("architect", payload)
                    if output.get("status") == "error":
                        error_msg = output.get("error", "")
                        if "Validation error" in error_msg and attempt < max_architect_retries - 1:
                            last_error = error_msg
                            continue
                        raise Exception(error_msg)

                    result_text = output.get("result", "")
                    plan_yaml = extract_yaml_from_response(result_text)
                    self.state["context"]["plan_yaml"] = plan_yaml
                    task.result = plan_yaml
                    self._set_project_type_from_plan(plan_yaml)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    self._update_context(task)
                    return task

                # All retries exhausted
                raise Exception(f"Architect failed after {max_architect_retries} attempts: {last_error}")

            # --- 4. CODER TASK (External: Coder) ---
            if task.assigned_role == AgentRole.CODER:
                system_prompt = self._get_system_prompt(AgentRole.CODER, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                coder_config = self.executor._get_agent_config(AgentRole.CODER)

                # Check context size
                if len(user_message) // 4 > 20000:
                    print(f"  ⚠ Large context warning for {task.task_id}")

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                        "model_name": coder_config.get("model", "local-model"),
                        "api_type": coder_config.get("api_type", "openai"),
                        "temperature": 0.5,
                        "max_tokens": coder_config.get("max_tokens", 25000),
                        "timeout": coder_config.get("timeout", 1200)
                    }
                }

                payload = self._enhance_coder_payload(payload, task)

                try:
                    # Generic Call
                    output = self._run_external_agent("coder", payload)

                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    result_text = output.get("result", "")
                    
                    # Post-processing
                    result_text = self._clean_code_output(result_text)
                    
                    # Fix relative imports for multi-file output
                    files_dict = self._parse_multi_file_output(result_text)
                    if files_dict:
                        fixed_files = {}
                        for fname, fcontent in files_dict.items():
                            fixed_files[fname] = self._fix_relative_imports(fcontent, files_dict)
                        # Rebuild combined output
                        parts = []
                        for fname, fcontent in fixed_files.items():
                            parts.append(f"### FILE: {fname} ###")
                            parts.append(fcontent)
                            parts.append("")
                        result_text = '\n'.join(parts)
                    
                    # Verify files if this is the main coding task
                    if task.task_type == "coding":
                        architect_task = next((t for t in self.completed_tasks 
                                               if t.assigned_role == AgentRole.ARCHITECT), None)
                        if architect_task and architect_task.result:
                            passed, missing = self._verify_coder_files_v2(architect_task.result, result_text)
                            if not passed and missing:
                                print(f"\n⚠️ FILE VERIFICATION WARNING: Missing files: {', '.join(missing)}")
                                task.metadata["missing_files"] = missing
                                task.metadata["file_verification_failed"] = True
                    
                    task.result = result_text
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                self._update_context(task)
                return task

            # ... (after Coder block) ...

            # --- 5. REVIEWER TASK (External: Reviewer) ---
            if task.assigned_role == AgentRole.REVIEWER:
                system_prompt = self._get_system_prompt(AgentRole.REVIEWER, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                reviewer_config = self.executor._get_agent_config(AgentRole.REVIEWER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": reviewer_config.get("url", "http://localhost:1233/v1"),
                        "model_name": reviewer_config.get("model", "local-model"),
                        "api_type": reviewer_config.get("api_type", "openai"),
                        "temperature": 0.3,
                        "max_tokens": 3000,
                        "timeout": reviewer_config.get("timeout", 600)
                    }
                }

                try:
                    # Generic Call
                    output = self._run_external_agent("reviewer", payload)

                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    result_text = output.get("result", "")
                    task.result = result_text
                    
                    # Reviewer specific logic: Check for rejection
                    if "NEEDS_REVISION" in result_text.upper() or "STATUS: REJECT" in result_text.upper():
                        task.metadata["needs_revision"] = True
                        print(f"  ⚠ Reviewer requested revision for task {task.task_id}")

                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                self._update_context(task)
                return task

            # ... (before Plan Execution block) ...

            # ... (after Reviewer block) ...

            # --- 6. TESTER TASK (External: Tester) ---
            if task.assigned_role == AgentRole.TESTER:
                self._diag(f"TESTER HANDLER MATCHED for {task.task_id}")
                system_prompt = self._get_system_prompt(AgentRole.TESTER, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                tester_config = self.executor._get_agent_config(AgentRole.TESTER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": tester_config.get("url", "http://localhost:1233/v1"),
                        "model_name": tester_config.get("model", "local-model"),
                        "api_type": tester_config.get("api_type", "openai"),
                        "temperature": tester_config.get("temperature", 0.7),
                        "max_tokens": tester_config.get("max_tokens", 4000),
                        "timeout": tester_config.get("timeout", 600)
                    }
                }

                try:
                    self._diag(f"TESTER calling _run_external_agent")
                    output = self._run_external_agent("tester", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    raw_result = output.get("result", "")
                    self._diag(f"TESTER got raw result: {len(raw_result)} chars, starts_with={repr(raw_result[:60])}")
                    result_text = self._clean_test_output(raw_result)
                    self._diag(f"TESTER cleaned result: {len(result_text)} chars, starts_with={repr(result_text[:60])}")
                    task.result = result_text
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    self._diag(f"TESTER EXCEPTION: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                task.completed_at = time.time()
                self._update_context(task)
                self._diag(f"TESTER DONE: status={task.status.value} result_len={len(task.result) if task.result else 0}")
                return task

            # --- 7. DOCUMENTER TASK (External: Documenter) ---
            if task.assigned_role == AgentRole.DOCUMENTER:
                system_prompt = self._get_system_prompt(AgentRole.DOCUMENTER, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                doc_config = self.executor._get_agent_config(AgentRole.DOCUMENTER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": doc_config.get("url", "http://localhost:1233/v1"),
                        "model_name": doc_config.get("model", "local-model"),
                        "api_type": doc_config.get("api_type", "openai"),
                        "temperature": 0.7,
                        "max_tokens": 3000,
                        "timeout": doc_config.get("timeout", 600)
                    }
                }

                try:
                    output = self._run_external_agent("documenter", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    task.result = output.get("result", "")
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                self._update_context(task)
                return task

            # ... (after Documenter block) ...

            # --- 8. SECURITY TASK (External) ---
# ... (Place this after the Documenter block) ...

            # --- 8. SECURITY TASK (External: Security) ---
            if task.assigned_role == AgentRole.SECURITY:
                system_prompt = self._get_system_prompt(AgentRole.SECURITY, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                sec_config = self.executor._get_agent_config(AgentRole.SECURITY)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": sec_config.get("url", "http://localhost:1233/v1"),
                        "model_name": sec_config.get("model", "local-model"),
                        "api_type": sec_config.get("api_type", "openai"),
                        "temperature": 0.8,
                        "max_tokens": 3000,
                        "timeout": sec_config.get("timeout", 600)
                    }
                }

                try:
                    output = self._run_external_agent("security", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    task.result = output.get("result", "")
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                self._update_context(task)
                return task

            # --- 9. OPTIMIZER TASK (External: Optimizer) ---
            if task.assigned_role == AgentRole.OPTIMIZER:
                system_prompt = self._get_system_prompt(AgentRole.OPTIMIZER, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                opt_config = self.executor._get_agent_config(AgentRole.OPTIMIZER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": opt_config.get("url", "http://localhost:1233/v1"),
                        "model_name": opt_config.get("model", "local-model"),
                        "api_type": opt_config.get("api_type", "openai"),
                        "temperature": 0.6,
                        "max_tokens": opt_config.get("max_tokens", 4000),
                        "timeout": opt_config.get("timeout", 600)
                    }
                }

                try:
                    output = self._run_external_agent("optimizer", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    task.result = output.get("result", "")
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                self._update_context(task)
                return task

            # --- 10. VERIFIER TASK (External: Verifier) ---
            # NOTE: Verifier requires smoke tests to run LOCALLY first
            if task.task_type == "verification" and task.assigned_role == AgentRole.VERIFIER:
                # 1. Save outputs & Run Smoke Tests (Internal Coordinator Logic)
                try:
                    self._save_project_outputs()
                except Exception as e:
                    print(f"   ! Warning: failed to save before verification: {e}")
                
                # Run the smoke tests using the internal method
                smoke_tests = self._run_smoke_tests()
                task.metadata["smoke_tests"] = smoke_tests
                
                # Check for hard failures (Fast Fail)
                def _smoke_failed(results):
                    failing = []
                    project_type = self.state["project_info"].get("project_type", "cli")
                    for name, info in (results or {}).items():
                        rc = info.get("returncode", None)
                        # Skip CLI checks for non-CLI projects (libraries, services)
                        if name in ("cli_usage", "cli_help"):
                            if project_type in ("library", "service"):
                                continue
                            if rc in (0, 2):
                                continue
                        if rc != 0:
                            failing.append(f"{name} (rc={rc})")
                    return (len(failing) > 0), failing


                failed, failing_list = _smoke_failed(smoke_tests)
                if failed:
                    print("SMOKE TEST DETAILS:")
                    print(json.dumps(smoke_tests, indent=2))
                    task.status = TaskStatus.FAILED
                    task.error = "Smoke tests failed: " + ", ".join(failing_list)
                    task.completed_at = time.time()
                    return task


                # 2. Prepare Prompt for External Agent
                system_prompt, user_message = self._get_verifier_prompt(task)
                ver_config = self.executor._get_agent_config(AgentRole.VERIFIER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": ver_config.get("url", "http://localhost:1233/v1"),
                        "model_name": ver_config.get("model", "local-model"),
                        "api_type": ver_config.get("api_type", "openai"),
                        "temperature": 0.3,
                        "max_tokens": 3000,
                        "timeout": ver_config.get("timeout", 600)
                    }
                }

                try:
                    # 3. Call External Agent
                    output = self._run_external_agent("verifier", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    result = output.get("result", "")
                    task.result = result

                    # 4. Parse Result
                    resp_upper = (result or "").upper()
                    
                    # Check for explicit FAIL first
                    explicit_fail = "VERIFICATION: FAIL" in resp_upper
                    
                    # Check for pass conditions (PASS, WARN with APPROVE, or just APPROVE)
                    explicit_pass = "VERIFICATION: PASS" in resp_upper
                    warn_with_approve = ("VERIFICATION: WARN" in resp_upper and "RECOMMENDATION: APPROVE" in resp_upper)
                    recommend_approve = "RECOMMENDATION: APPROVE" in resp_upper and not explicit_fail
                    
                    passed = (explicit_pass or warn_with_approve or recommend_approve) and not explicit_fail
                    
                    if passed:
                        task.status = TaskStatus.COMPLETED
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = "Verifier returned FAIL or did not approve"

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                return task

            # --- END OF TASK HANDLERS ---
            self._diag(f"FALLTHROUGH: no handler matched task_id={task.task_id} task_type={task.task_type} role={task.assigned_role}")

        except Exception as e:
            # Global exception handler
            self._diag(f"OUTER EXCEPT: task_id={task.task_id} error={e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            print(f"  ✗ Task {task.task_id} failed: {e}")

        return task
    
    def _clean_code_output(self, code: str) -> str:
        """Clean markdown formatting and explanatory text from code output"""
        
        # Check if this is multi-file output with our markers - if so, preserve structure
        if '### FILE:' in code:
            # Just clean markdown code blocks but preserve file structure
            lines = code.split('\n')
            cleaned = []
            for line in lines:
                # Skip markdown code block markers
                if line.strip().startswith('```'):
                    continue
                cleaned.append(line)
            return '\n'.join(cleaned).strip()
        
        # Single file output - apply normal cleaning
        # First, try to extract from markdown code blocks
        code_block_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_block_pattern, code, re.DOTALL)
        if matches:
            code = '\n\n'.join(matches)
        
        lines = code.split('\n')
        cleaned_lines = []
        found_code_start = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines before code starts
            if not found_code_start and not stripped:
                continue
            
            # Skip markdown headers (but not our file markers)
            if re.match(r'^#{1,6}\s', line) and '### FILE:' not in line:
                continue
            
            # Skip numbered lists at start
            if not found_code_start and re.match(r'^\d+\.\s', stripped):
                continue
            
            # Skip bold markdown headers
            if stripped.startswith('**') and stripped.endswith('**'):
                continue
            
            # Detect actual Python code or file markers
            code_indicators = (
                stripped.startswith('import '),
                stripped.startswith('from '),
                stripped.startswith('class '),
                stripped.startswith('def '),
                stripped.startswith('"""'),
                stripped.startswith("'''"),
                stripped.startswith('@'),  # decorators
                stripped.startswith('#!'),  # shebang
                (stripped.startswith('#') and not stripped.startswith('###')),
                '### FILE:' in stripped  # Our file markers
            )
            
            if any(code_indicators):
                found_code_start = True
            
            if found_code_start:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _fix_relative_imports(self, content: str, files_dict: Dict[str, str]) -> str:
        """Fix absolute imports to relative imports for sibling modules."""
        # Get module names from the files dict
        sibling_modules = [f.replace('.py', '') for f in files_dict.keys() if f.endswith('.py')]
        
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            for module in sibling_modules:
                # Match "from module import" but not "from .module import"
                import_pattern = f"from {module} import "
                relative_pattern = f"from .{module} import "
                
                if import_pattern in fixed_line and relative_pattern not in fixed_line:
                    fixed_line = fixed_line.replace(import_pattern, relative_pattern)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _clean_test_output(self, test_code: str) -> str:
        """Clean markdown formatting and prose from test output"""
        # Strategy 1: If there's a code block, extract from it
        code_block_match = re.search(r'```(?:python)?\s*\n(.*?)```', test_code, re.DOTALL)
        if code_block_match:
            test_code = code_block_match.group(1)
        else:
            # Remove stray backticks
            test_code = re.sub(r'^```python\s*\n?', '', test_code)
            test_code = re.sub(r'^```\s*\n?', '', test_code, flags=re.MULTILINE)
            test_code = re.sub(r'\n```\s*$', '', test_code)
        
        test_code = test_code.strip()
        
        # Strategy 2: Strip leading prose (anything before first valid Python line)
        lines = test_code.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Valid Python starts: imports, comments, docstrings, decorators, or code
            if stripped.startswith(('import ', 'from ', '#', '"""', "'''", '@', 'def ', 'class ', 'async ')):
                start_idx = i
                break
            # Also accept pytest fixtures and constants
            if re.match(r'^[A-Z_][A-Z0-9_]*\s*=', stripped) or stripped.startswith('pytest'):
                start_idx = i
                break
        
        # Strategy 3: Strip trailing prose (anything after code ends)
        # Look for explanatory text patterns
        end_idx = len(lines)
        code_ended = False
        for i in range(len(lines) - 1, start_idx, -1):
            stripped = lines[i].strip()
            # Skip empty lines at the end
            if not stripped:
                continue
            # Prose indicators: sentences that explain rather than code
            prose_patterns = [
                r'^(This |These |The |Note:|Here |I |We |To |In this|This test)',
                r'^(Make sure|Remember|You can|For example)',
                r'^\d+\.\s+\w',  # Numbered lists
                r'^\*\s+\w',     # Bullet points
            ]
            is_prose = any(re.match(p, stripped, re.IGNORECASE) for p in prose_patterns)
            if is_prose and not code_ended:
                end_idx = i
            elif stripped and not is_prose:
                code_ended = True
                break
        
        cleaned_lines = lines[start_idx:end_idx if end_idx != len(lines) else None]
        cleaned = '\n'.join(cleaned_lines).rstrip()

        # Syntax validation: try to compile, attempt fixes if broken
        cleaned = self._validate_and_fix_python_syntax(cleaned)
        return cleaned

    def _validate_and_fix_python_syntax(self, code: str) -> str:
        """Validate Python syntax and fix common LLM errors."""
        import ast

        # Fix missing typing imports (causes NameError at runtime even if syntax is valid)
        code = self._fix_missing_typing_imports(code)
        # Fix missing stdlib imports (e.g. json.loads used without import json)
        code = self._fix_missing_stdlib_imports(code)

        # First check: is it already valid?
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass

        # Fix attempt 1: Mismatched brackets across the whole file
        # Common LLM error: [{'key': val]}  should be  [{'key': val}]
        fixed = self._fix_brackets_whole_file(code)

        try:
            ast.parse(fixed)
            return fixed
        except SyntaxError:
            pass

        # Fix attempt 2: Remove lines that cause syntax errors one at a time
        # (last resort - try to salvage what we can)
        lines = fixed.split('\n')
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                ast.parse('\n'.join(lines))
                return '\n'.join(lines)
            except SyntaxError as e:
                if e.lineno and 1 <= e.lineno <= len(lines):
                    # Comment out the broken line
                    bad_idx = e.lineno - 1
                    lines[bad_idx] = '# SYNTAX_FIX: ' + lines[bad_idx]
                else:
                    break

        # If all fixes fail, return original (will fail at smoke test, trigger revision)
        return code

    @staticmethod
    def _fix_brackets_whole_file(code: str) -> str:
        """Fix mismatched brackets in LLM-generated Python code.

        Uses targeted fixes at SyntaxError locations:
        1. Swap adjacent mismatched bracket pairs (e.g. ]} -> }])
        2. Remove extra closers one at a time
        3. Re-parse after each attempt
        """
        import ast

        max_fix_rounds = 15
        current = code

        for _ in range(max_fix_rounds):
            try:
                ast.parse(current)
                return current
            except SyntaxError as e:
                if not e.lineno:
                    return current

                lines = current.split('\n')
                err_line_idx = e.lineno - 1
                if err_line_idx < 0 or err_line_idx >= len(lines):
                    return current

                err_line = lines[err_line_idx]
                fixed_line = None

                # Strategy 1: Swap adjacent mismatched bracket pairs
                # e.g. ]} -> }]  or )} -> })  or ]) -> )]
                bracket_chars = set('()[]{}')
                for j in range(len(err_line) - 1):
                    a, b = err_line[j], err_line[j + 1]
                    if a in bracket_chars and b in bracket_chars and a != b:
                        # Try swapping
                        candidate = err_line[:j] + b + a + err_line[j + 2:]
                        lines[err_line_idx] = candidate
                        test_code = '\n'.join(lines)
                        try:
                            ast.parse(test_code)
                            return test_code
                        except SyntaxError:
                            pass
                        lines[err_line_idx] = err_line  # restore

                # Strategy 2: Replace each closer with each other closer type
                # e.g. ) -> ] or } -> ) when the bracket type is wrong
                closer_types = [')', ']', '}']
                for j in range(len(err_line)):
                    if err_line[j] in closer_types:
                        for replacement in closer_types:
                            if replacement != err_line[j]:
                                candidate = err_line[:j] + replacement + err_line[j + 1:]
                                lines[err_line_idx] = candidate
                                test_code = '\n'.join(lines)
                                try:
                                    ast.parse(test_code)
                                    return test_code
                                except SyntaxError:
                                    pass
                                lines[err_line_idx] = err_line

                # Strategy 3: Try removing each closer on the error line one at a time
                closer_positions = [j for j, c in enumerate(err_line) if c in ')]}']
                for j in closer_positions:
                    candidate = err_line[:j] + err_line[j + 1:]
                    lines[err_line_idx] = candidate
                    test_code = '\n'.join(lines)
                    try:
                        ast.parse(test_code)
                        return test_code
                    except SyntaxError:
                        pass
                    lines[err_line_idx] = err_line  # restore

                # Strategy 4: Also check previous line (error often reported on next line)
                if err_line_idx > 0:
                    prev_line = lines[err_line_idx - 1]
                    prev_closer_positions = [j for j, c in enumerate(prev_line) if c in ')]}']
                    for j in reversed(prev_closer_positions):
                        candidate = prev_line[:j] + prev_line[j + 1:]
                        lines[err_line_idx - 1] = candidate
                        test_code = '\n'.join(lines)
                        try:
                            ast.parse(test_code)
                            return test_code
                        except SyntaxError:
                            pass
                        lines[err_line_idx - 1] = prev_line  # restore

                # Strategy 5: Swap on previous line too
                if err_line_idx > 0:
                    prev_line = lines[err_line_idx - 1]
                    for j in range(len(prev_line) - 1):
                        a, b = prev_line[j], prev_line[j + 1]
                        if a in bracket_chars and b in bracket_chars and a != b:
                            candidate = prev_line[:j] + b + a + prev_line[j + 2:]
                            lines[err_line_idx - 1] = candidate
                            test_code = '\n'.join(lines)
                            try:
                                ast.parse(test_code)
                                return test_code
                            except SyntaxError:
                                pass
                            lines[err_line_idx - 1] = prev_line

                # No fix found for this error, give up
                return current

        return current

    def execute_tasks_parallel(self, tasks: List[Task]) -> List[Task]:
        """Execute multiple tasks in parallel"""
        if not self.config.get('workflow', {}).get('enable_parallel', True) or len(tasks) <= 1:
            return [self.execute_task(task) for task in tasks]
        
        completed = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_parallel)) as executor:
            future_to_task = {executor.submit(self.execute_task, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    completed_task = future.result()
                    completed.append(completed_task)
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
                    completed.append(task)
        
        return completed
    
    def _handle_clarification_interactive(self, task: Task):
        """Handle clarification task by showing questions to user and getting answers"""
        if not task.result:
            return
        user_request = task.metadata.get("user_request", "")
        
        result = (task.result).strip()
        
        # Check if requirements are already clear
        if "STATUS: CLEAR" in result.upper():
            print("\n✓ Requirements are clear, proceeding...")
            self.state["context"]['clarification'] = "Requirements are clear and complete."
            return
        
        # Extract and display questions
        print("\n" + "=" * 80)
        print("CLARIFICATION NEEDED")
        print("=" * 80)
        print("\nThe clarifier has identified some questions:\n")
        print(result)
        print("\n" + "=" * 80)
        
        # Get user input
        print("\nPlease provide answers to these questions:")
        print("(Enter your answers, then type 'DONE' on a new line when finished)\n")
        
        user_answers = []
        while True:
            try:
                line = input()
                if line.strip().upper() == 'DONE':
                    break
                user_answers.append(line)
            except EOFError:
                break
        
        answers_text = '\n'.join(user_answers).rstrip()

        # If user typed DONE immediately (no answers), proceed with defaults but record it
        assumed_defaults = (answers_text.strip() == "")
        if assumed_defaults:
            self.state["context"]["clarification_assumed_defaults"] = True
            answers_text = "(no answers provided; proceeding with defaults)"
        else:
            self.state["context"]["clarification_assumed_defaults"] = False

        
        # FIXED: Store comprehensive clarified requirements
        # Synthesize job scope from Q&A
        print("\n⏳ Synthesizing job scope from requirements...")
        
        clarifier_config = self.executor._get_agent_config(AgentRole.CLARIFIER)
        
        # Check if structured output is requested (collaborative workflow)
        output_format = task.metadata.get("output_format", "")

        synthesis_payload = {
            "mode": "synthesize",
            "user_request": task.metadata.get("user_request", ""),
            "questions": result,
            "answers": answers_text,
            "config": {
                "model_url": clarifier_config.get("url", "http://localhost:1233/v1"),
                "model_name": clarifier_config.get("model", "local-model"),
                "api_type": clarifier_config.get("api_type", "openai"),
                "temperature": 0.7,
                "max_tokens": clarifier_config.get("max_tokens", 25000),
                "timeout": clarifier_config.get("timeout", 300)
            }
        }

        # Pass output_format for structured job spec
        if output_format:
            synthesis_payload["output_format"] = output_format

        # Log the prompt
        self._log_prompt("SYNTHESIZE", "CLARIFIER", {
            "system_prompt": "SYNTHESIZE_STRUCTURED_PROMPT" if output_format == "structured" else "SYNTHESIZE_PROMPT",
            "user_message": f"Request: {user_request[:200]}... Q&A: {result[:200]}..."
        }, mode=f"synthesize_{output_format}" if output_format else "synthesize")

        try:
            synthesis_output = self._run_external_agent("clarifier", synthesis_payload)
            job_scope = synthesis_output.get("job_scope", "")

            if not job_scope:
                # Fallback to raw Q&A if synthesis failed
                print("⚠ Synthesis failed, using raw Q&A")
                job_scope = f"ORIGINAL REQUEST:\n{user_request}\n\nCLARIFICATION Q&A:\n{result}\n\nANSWERS:\n{answers_text}"
            else:
                print("✓ Job scope synthesized")

            # Store job_spec for collaborative workflow
            if output_format == "structured":
                job_spec = synthesis_output.get("job_spec", job_scope)
                self.state["context"]["job_spec"] = job_spec

        except Exception as e:
            print(f"⚠ Synthesis error: {e}, using raw Q&A")
            job_scope = f"ORIGINAL REQUEST:\n{user_request}\n\nCLARIFICATION Q&A:\n{result}\n\nANSWERS:\n{answers_text}"
            if output_format == "structured":
                self.state["context"]["job_spec"] = job_scope

        if self.state["context"].get("clarification_assumed_defaults"):
            job_scope += "\n\nASSUMPTIONS:\n- No clarifier answers were provided; agent should use simplest reasonable defaults."

        self.state["context"]['job_scope'] = job_scope
        self.state["context"]['clarification'] = job_scope  # Keep for backward compatibility
        self.state["context"]['user_answers'] = answers_text

        print("\n✓ Clarification complete, proceeding with workflow...")
    
    def _handle_revision_cycle(self, review_tasks: List[Task]) -> bool:
        """
        Check if any review requires revision and handle the cycle.
        Returns True if revision was triggered.
        """
        needs_revision = any(
            t.metadata.get("needs_revision", False) 
            for t in review_tasks 
            if t.status == TaskStatus.COMPLETED
        )
        
        if not needs_revision:
            return False

        # Check if any review task has per_file_results → route to file-by-file revision
        for rt in review_tasks:
            if rt.metadata.get("per_file_results") and rt.metadata.get("failed_files"):
                print(f"\n🔄 File-by-file revision for {len(rt.metadata['failed_files'])} failed files")
                try:
                    success = self._handle_file_by_file_revision(rt)
                    if success:
                        return True
                    print(f"  ⚠ File-by-file revision failed, falling back to generic revision")
                except Exception as e:
                    print(f"  ⚠ File-by-file revision error ({e}), falling back to generic revision")
                break

        # Find the coding task (also check build_from_plan for collaborative workflow)
        code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution", "build_from_plan")), None)
        if not code_task:
            return False

        # Check revision count
        if code_task.revision_count >= code_task.max_revisions:
            print(f"\n⚠ Max revisions ({code_task.max_revisions}) reached, proceeding anyway")
            return False

        print(f"\n🔄 Revision cycle {code_task.revision_count + 1}/{code_task.max_revisions}")

        # Collect review feedback
        feedback = []
        for rt in review_tasks:
            if rt.result and rt.metadata.get("needs_revision"):
                feedback.append(f"[{rt.task_id}]: {rt.result}")

        # Build revision metadata
        revision_metadata = {
            "revision_feedback": "\n\n".join(feedback),
            "original_code": code_task.result,
            "user_request": self.state["context"].get("user_request", ""),
            "is_revision": True
        }

        # Include final_plan when revision is triggered from compliance review
        for rt in review_tasks:
            if rt.task_type == "compliance_review" and rt.metadata.get("final_plan"):
                revision_metadata["final_plan"] = rt.metadata["final_plan"]
                break
        if "final_plan" not in revision_metadata and self.state["context"].get("final_plan"):
            revision_metadata["final_plan"] = self.state["context"]["final_plan"]

        # Create revision task
        revision_task = Task(
            task_id=f"T_revision_{code_task.revision_count + 1}",
            task_type="revision",
            description="Revise code based on review feedback",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata=revision_metadata
        )
        revision_task.revision_count = code_task.revision_count + 1
        
        # Execute revision
        self.execute_task(revision_task)
        
        if revision_task.status == TaskStatus.COMPLETED:
            # MERGE revised files back into the full code
            # Revision only outputs changed files — we must preserve unchanged files
            original_code = code_task.result or ""
            revised_code = revision_task.result or ""

            original_files = self._parse_multi_file_output(original_code)
            revised_files = self._parse_multi_file_output(revised_code)

            if original_files and revised_files:
                # Merge: revised files overwrite originals, keep everything else
                merged_files = dict(original_files)
                merged_files.update(revised_files)

                # Reassemble into ### FILE: format
                merged_parts = []
                for fname, content in merged_files.items():
                    merged_parts.append(f"### FILE: {fname} ###\n{content}")
                merged_result = "\n\n".join(merged_parts)

                revision_task.result = merged_result
                print(f"   ✓ Merged {len(revised_files)} revised file(s) into {len(original_files)} total files")

            # Update the context with merged code
            self._update_context(revision_task)
            self.completed_tasks.append(revision_task)

            # Re-validate revised code with a fresh compliance review
            self._revalidate_after_revision()

            return True

        return False

    def _run_integration_check(self, all_files: Dict[str, str], parsed_plan=None) -> Dict[str, list]:
        """
        Pure-AST cross-file integration check. Validates:
        1. Constructor arg count: ClassName(a, b) matches __init__(self, a, b)
        2. Import resolution: from .module import X — module exists and exports X
        3. Attribute access: self.x.field — field exists in x's class

        Returns dict[filename -> list[str]] of issues per file.
        """
        import ast

        issues_by_file = {}

        # Step 1: Build maps from all files
        # class_init_args: {"ClassName": {"file": fname, "args": ["a", "b"], "min_args": N}}
        # file_exports: {"fname": set of top-level class/function names}
        # class_attributes: {"ClassName": set of attr names (fields + methods)}
        class_init_args = {}
        file_exports = {}
        class_attributes = {}

        for fname, code in all_files.items():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue

            exports = set()
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    exports.add(node.name)
                    attrs = set()

                    for item in ast.walk(node):
                        # Public methods
                        if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                            attrs.add(item.name)
                            # Parse __init__ for constructor signature
                            if item.name == '__init__':
                                args = item.args
                                all_arg_names = [a.arg for a in args.args if a.arg != 'self']
                                n_defaults = len(args.defaults)
                                min_args = len(all_arg_names) - n_defaults
                                class_init_args[node.name] = {
                                    "file": fname,
                                    "args": all_arg_names,
                                    "min_args": min_args,
                                    "max_args": len(all_arg_names)
                                }
                                # Also extract self.X assignments in __init__
                                for stmt in ast.walk(item):
                                    if isinstance(stmt, ast.Assign):
                                        for target in stmt.targets:
                                            if (isinstance(target, ast.Attribute) and
                                                isinstance(target.value, ast.Name) and
                                                target.value.id == 'self'):
                                                attrs.add(target.attr)

                        # Dataclass fields (annotated assignments)
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            attrs.add(item.target.id)

                        # Class-level assignments
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    attrs.add(target.id)

                    class_attributes[node.name] = attrs

                    # Dataclass auto-init detection
                    if node.name not in class_init_args:
                        is_dataclass = any(
                            (isinstance(d, ast.Name) and d.id == 'dataclass') or
                            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass') or
                            (isinstance(d, ast.Attribute) and d.attr == 'dataclass')
                            for d in node.decorator_list
                        )
                        if is_dataclass:
                            dc_fields = []
                            dc_required = 0
                            for item in node.body:
                                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                    dc_fields.append(item.target.id)
                                    if item.value is None:
                                        dc_required += 1
                            class_init_args[node.name] = {
                                "file": fname,
                                "args": dc_fields,
                                "min_args": dc_required,
                                "max_args": len(dc_fields)
                            }

                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    exports.add(node.name)

                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            exports.add(target.id)

            file_exports[fname] = exports

        # Build module name map: "module_name" -> fname (e.g. "config" -> "config.py")
        module_to_file = {}
        for fname in all_files:
            # Handle both "config.py" and "core/config.py"
            base = fname.rsplit('/', 1)[-1] if '/' in fname else fname
            module_name = base.replace('.py', '')
            module_to_file[module_name] = fname
            # Also map full path without .py
            module_to_file[fname.replace('.py', '').replace('/', '.')] = fname

        # Step 2: Check each file for integration issues
        for fname, code in all_files.items():
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue

            file_issues = []

            # Check 1: Constructor arg count
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in class_init_args:
                        cls_name = node.func.id
                        info = class_init_args[cls_name]
                        if info["file"] == fname:
                            continue  # Skip self-file constructors
                        n_provided = len(node.args) + len(node.keywords)
                        if n_provided < info["min_args"]:
                            file_issues.append(
                                f"Constructor {cls_name}() called with {n_provided} args, "
                                f"but __init__ requires at least {info['min_args']} "
                                f"(params: {', '.join(info['args'])})"
                            )
                        elif n_provided > info["max_args"] and not any(
                            True for a in [ast.parse("def f(**k): pass").body[0].args]
                        ):
                            # Don't flag if we can't detect **kwargs properly
                            pass

            # Check 2: Import resolution
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    # Handle relative imports: from .config import Settings
                    module_path = node.module.lstrip('.')
                    # Try to resolve module
                    resolved_file = module_to_file.get(module_path)
                    if not resolved_file:
                        # Try last component
                        last_part = module_path.rsplit('.', 1)[-1]
                        resolved_file = module_to_file.get(last_part)

                    if resolved_file and resolved_file in file_exports:
                        for alias in (node.names or []):
                            imported_name = alias.name
                            if imported_name != '*' and imported_name not in file_exports[resolved_file]:
                                file_issues.append(
                                    f"Imports '{imported_name}' from {module_path}, "
                                    f"but {resolved_file} does not export it "
                                    f"(available: {', '.join(sorted(file_exports[resolved_file])[:10])})"
                                )

            # Check 3: self.X.Y attribute access validation
            seen_attr_issues = set()
            for cls_node in ast.iter_child_nodes(tree):
                if not isinstance(cls_node, ast.ClassDef):
                    continue
                # Build self_type_map for this class
                self_type_map = {}
                for item in cls_node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        param_types = {}
                        for arg in item.args.args:
                            if arg.arg == 'self':
                                continue
                            if arg.annotation:
                                if isinstance(arg.annotation, ast.Name):
                                    param_types[arg.arg] = arg.annotation.id
                                elif isinstance(arg.annotation, ast.Attribute):
                                    param_types[arg.arg] = arg.annotation.attr

                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if (isinstance(target, ast.Attribute) and
                                            isinstance(target.value, ast.Name) and
                                            target.value.id == 'self'):
                                        attr_name = target.attr
                                        if isinstance(stmt.value, ast.Name) and stmt.value.id in param_types:
                                            self_type_map[attr_name] = param_types[stmt.value.id]
                                        elif (isinstance(stmt.value, ast.Call) and
                                              isinstance(stmt.value.func, ast.Name) and
                                              stmt.value.func.id in class_attributes):
                                            self_type_map[attr_name] = stmt.value.func.id

                # Validate self.X.Y
                for node in ast.walk(cls_node):
                    if (isinstance(node, ast.Attribute) and
                            isinstance(node.value, ast.Attribute) and
                            isinstance(node.value.value, ast.Name) and
                            node.value.value.id == 'self'):
                        x_attr = node.value.attr
                        y_attr = node.attr
                        if x_attr in self_type_map:
                            type_name = self_type_map[x_attr]
                            if type_name in class_attributes:
                                known_attrs = class_attributes[type_name]
                                if y_attr not in known_attrs:
                                    issue_key = (x_attr, y_attr, type_name)
                                    if issue_key not in seen_attr_issues:
                                        seen_attr_issues.add(issue_key)
                                        file_issues.append(
                                            f"self.{x_attr}.{y_attr} — '{y_attr}' not found in "
                                            f"{type_name} (available: {', '.join(sorted(known_attrs)[:15])})"
                                        )

            # Check 4: Constructor keyword arg validation
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    cls_name = node.func.id
                    if cls_name in class_init_args:
                        info = class_init_args[cls_name]
                        if info["file"] == fname:
                            continue
                        for kw in node.keywords:
                            if kw.arg and kw.arg not in info["args"]:
                                file_issues.append(
                                    f"Constructor {cls_name}(... {kw.arg}=...) — "
                                    f"'{kw.arg}' is not a parameter of __init__ "
                                    f"(params: {', '.join(info['args'])})"
                                )

            if file_issues:
                issues_by_file[fname] = file_issues

        return issues_by_file

    def _handle_file_by_file_revision(self, review_task) -> bool:
        """
        Handle revision using PlanExecutor's per-file revision infrastructure.
        Uses multi-coder fallback, AST validation, anti-stub detection per file.
        Returns True if revision succeeded.
        """
        if not PLAN_EXECUTOR_AVAILABLE:
            return False

        per_file_results = review_task.metadata.get("per_file_results", {})
        failed_files = review_task.metadata.get("failed_files", [])
        final_plan = review_task.metadata.get("final_plan", self.state["context"].get("final_plan", ""))

        if not failed_files or not final_plan:
            return False

        # Get current code
        latest_code = self.state["context"].get("latest_code", "")
        all_files = self._parse_multi_file_output(latest_code)
        if not all_files:
            return False

        # Instantiate PlanExecutor with the coordinator's executor and config
        pe = PlanExecutor(self.executor, self.config)
        # Wire up prompt logging
        project_dir = self.state["project_info"].get("project_dir")
        if project_dir:
            pe.prompt_log_dir = os.path.join(project_dir, "prompt_logs")
        try:
            pe.plan = pe.parse_plan(final_plan)
        except Exception as e:
            print(f"    ⚠ Could not parse plan for revision: {e}")
            return False

        pe.job_scope = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))
        user_request = self.state["context"].get("user_request", "")

        # Populate completed_files from current code so deps are available
        for fname, code in all_files.items():
            file_spec = pe._get_file_spec(fname)
            if file_spec:
                actual_exports = pe._extract_exports(code) if hasattr(pe, '_extract_exports') else []
                pe.completed_files[fname] = FileResult(
                    name=fname,
                    content=code,
                    actual_exports=actual_exports,
                    status=FileStatus.COMPLETED
                )

        revised_count = 0
        still_failing = []

        for fname in failed_files:
            file_spec = pe._get_file_spec(fname)
            if not file_spec:
                print(f"    ⚠ {fname}: not in plan, skipping")
                continue

            file_code = all_files.get(fname, "")
            review_result = per_file_results.get(fname, {})
            review_text = review_result.get("result", "")

            context = pe._build_context_for_file(file_spec)

            if not file_code:
                # Missing file — generate from scratch
                print(f"    → {fname}: generating (missing)")
                try:
                    result = pe._generate_file(file_spec, user_request, context)
                    if result.content:
                        all_files[fname] = result.content
                        pe.completed_files[fname] = result
                        revised_count += 1
                        print(f"    ✓ {fname}: generated ({len(result.content)} chars)")
                    else:
                        print(f"    ✗ {fname}: generation failed")
                except Exception as e:
                    print(f"    ✗ {fname}: generation error ({e})")
            else:
                # Failed file — build FileResult with reviewer issues, use _handle_revision
                print(f"    → {fname}: revising")

                # Parse issues from review text (capped at 10 to prevent prompt flooding)
                issues = []
                if review_text:
                    # Extract issues from ISSUES: section
                    issues_match = re.search(r'ISSUES:\s*\n(.*?)(?:\n\n|\Z)', review_text, re.DOTALL)
                    if issues_match:
                        for line in issues_match.group(1).strip().split('\n'):
                            line = line.strip()
                            if line and not line.lower().startswith('status'):
                                # Remove leading numbering
                                line = re.sub(r'^\d+\.\s*', '', line)
                                if line:
                                    issues.append(line)
                    if not issues:
                        issues.append(review_text[:500])
                    # Cap to 10 issues max
                    if len(issues) > 10:
                        omitted = len(issues) - 10
                        issues = issues[:10]
                        issues.append(f"[... {omitted} additional issues omitted — fix the above first]")

                actual_exports = pe._extract_exports(file_code) if hasattr(pe, '_extract_exports') else []
                file_result = FileResult(
                    name=fname,
                    content=file_code,
                    actual_exports=actual_exports,
                    status=FileStatus.PLAN_MISMATCH,
                    plan_compliance={"passed": False, "issues": issues}
                )

                try:
                    revised = pe._handle_revision(file_spec, file_result, context, user_request)
                    if revised.content and revised.status == FileStatus.COMPLETED:
                        all_files[fname] = revised.content
                        pe.completed_files[fname] = revised
                        revised_count += 1
                        print(f"    ✓ {fname}: revised ({len(revised.content)} chars)")
                    elif revised.content and revised.status == FileStatus.PLAN_MISMATCH:
                        # Partially improved but still failing compliance —
                        # update the code (may help dependent files) but don't
                        # count as success so coordinator can retry
                        all_files[fname] = revised.content
                        pe.completed_files[fname] = revised
                        still_failing.append(fname)
                        print(f"    ⚠ {fname}: revised but still failing compliance ({len(revised.content)} chars)")
                    else:
                        still_failing.append(fname)
                        print(f"    ✗ {fname}: revision produced no content")
                except Exception as e:
                    print(f"    ✗ {fname}: revision error ({e})")

            self._log_prompt(f"COMPLIANCE_REVISION_{fname}", "CODER", {
                "system_prompt": "(per-file revision)",
                "user_message": f"Revised {fname}",
                "file_code": all_files.get(fname, "")[:2000]
            }, mode="file_revision")

        if revised_count == 0 and not still_failing:
            return False

        # Post-revision cross-file integration validation
        try:
            integration_issues = self._run_integration_check(all_files, pe.plan if pe.plan else None)
            if integration_issues:
                total_issues = sum(len(v) for v in integration_issues.values())
                print(f"\n  ⚠ Integration check found {total_issues} cross-file issues in {len(integration_issues)} files")
                for ifname, iissues in integration_issues.items():
                    for iss in iissues[:3]:
                        print(f"    → {ifname}: {iss}")

                # Attempt one more revision pass on affected files
                integration_revised = 0
                for ifname, iissues in integration_issues.items():
                    file_spec = pe._get_file_spec(ifname)
                    if not file_spec or ifname not in all_files:
                        continue

                    context = pe._build_context_for_file(file_spec)
                    actual_exports = pe._extract_exports(all_files[ifname]) if hasattr(pe, '_extract_exports') else []
                    file_result = FileResult(
                        name=ifname,
                        content=all_files[ifname],
                        actual_exports=actual_exports,
                        status=FileStatus.PLAN_MISMATCH,
                        plan_compliance={"passed": False, "issues": iissues[:10]}
                    )

                    try:
                        revised = pe._handle_revision(file_spec, file_result, context, user_request)
                        if revised.content and revised.status == FileStatus.COMPLETED:
                            all_files[ifname] = revised.content
                            pe.completed_files[ifname] = revised
                            integration_revised += 1
                            print(f"    ✓ {ifname}: integration-revised ({len(revised.content)} chars)")
                        elif revised.content:
                            # Partially improved — update code but don't count as fixed
                            all_files[ifname] = revised.content
                            pe.completed_files[ifname] = revised
                            print(f"    ⚠ {ifname}: integration-revised but still failing")
                    except Exception as e:
                        print(f"    ✗ {ifname}: integration revision error ({e})")

                if integration_revised > 0:
                    print(f"  ✓ Integration revision: {integration_revised} files fixed")
                else:
                    print(f"  ⚠ Integration issues remain (could not auto-fix)")
        except Exception as e:
            print(f"  ⚠ Integration check error: {e}")

        # Reassemble all files into ### FILE: format
        merged_parts = []
        for fname, content in all_files.items():
            merged_parts.append(f"### FILE: {fname} ###\n{content}")
        merged_result = "\n\n".join(merged_parts)

        # Create synthetic revision task
        revision_task = Task(
            task_id=f"T_file_revision_{int(time.time())}",
            task_type="revision",
            description=f"File-by-file revision of {revised_count} files",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.COMPLETED,
            priority=10,
            metadata={"is_revision": True, "revised_files": list(all_files.keys())}
        )
        revision_task.result = merged_result
        revision_task.completed_at = time.time()

        # Update context and save
        self._update_context(revision_task)
        self.completed_tasks.append(revision_task)

        # Save to disk immediately
        try:
            self._save_project_outputs()
            print(f"  ✓ File-by-file revision complete: {revised_count}/{len(failed_files)} files revised, all {len(all_files)} files saved")
            if still_failing:
                print(f"  ⚠ Still failing compliance: {', '.join(still_failing)}")
        except Exception as e:
            print(f"  ⚠ Save after revision failed: {e}")

        # Re-validate revised code with a fresh compliance review
        self._revalidate_after_revision()

        # Only report success if all failed files were fully fixed
        return len(still_failing) == 0 and revised_count > 0

    def _revalidate_after_revision(self):
        """
        Run a fresh compliance review after revision to verify fixes.
        Creates a new compliance_review task, executes it, and appends results.
        """
        print(f"\n🔍 Re-validating revised code...")

        revalidation_task = Task(
            task_id=f"T_revalidation_{int(time.time())}",
            task_type="compliance_review",
            description="Post-revision compliance re-validation",
            assigned_role=AgentRole.REVIEWER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={
                "user_request": self.state["context"].get("user_request", ""),
                "is_revalidation": True
            }
        )

        try:
            self.execute_task(revalidation_task)

            if revalidation_task.status == TaskStatus.COMPLETED:
                self.completed_tasks.append(revalidation_task)

                still_needs_revision = revalidation_task.metadata.get("needs_revision", False)
                failed_files = revalidation_task.metadata.get("failed_files", [])

                if still_needs_revision and failed_files:
                    print(f"  ⚠ Re-validation: {len(failed_files)} file(s) still have issues: {', '.join(failed_files)}")
                else:
                    print(f"  ✓ Re-validation: all files pass compliance review")

                # Update stored review results in context
                if revalidation_task.result:
                    self.state["context"]["compliance_review_reviewer"] = {
                        "task_id": revalidation_task.task_id,
                        "result": revalidation_task.result,
                        "completed_at": revalidation_task.completed_at or time.time()
                    }
            else:
                print(f"  ⚠ Re-validation task did not complete (status: {revalidation_task.status.value})")
        except Exception as e:
            print(f"  ⚠ Re-validation failed: {e}")

    def run_workflow(self, user_request: str, workflow_type: str = "standard", stop_after: str = None):
        """Execute a complete workflow.

        Args:
            stop_after: Stop after this stage completes. Valid values for collaborative:
                        'clarify', 'draft_plan', 'plan_review', 'finalize_plan',
                        'build', 'compliance'. If None, runs all stages.
        """
        print("=" * 80)
        print(f"ADVANCED SWARM COORDINATOR v2")
        print(f"Workflow: {workflow_type}")
        if stop_after:
            print(f"Stop after: {stop_after}")
        print("=" * 80)
        
        # Start Docker sandbox if enabled
        if self.sandbox and self.use_docker:
            self.sandbox.start()
        
        # FIXED: Store user request in context for all downstream tasks
        self.state["context"]["user_request"] = user_request
        
        # Setup project directory (skip for custom/bugfix/import as they handle their own)
        if workflow_type not in ("custom", "bugfix", "import") or not self.task_queue:
            project_name = self._create_project_name(user_request) if user_request else "custom_project"
            project_dir = self._setup_project_directory(project_name, user_request or "Custom workflow")
            
            print(f"\n📁 Project: {os.path.basename(project_dir)}")
            print(f"   Location: {project_dir}")
            print(f"   Number: {self.state['project_info']['project_number']:03d}")
            print(f"   Version: {self.state['project_info']['version']}")
        else:
            # RETRIEVE EXISTING DIR FROM STATE
            project_dir = self.state["project_info"].get("project_dir")
            # If for some reason it's missing (rare edge case), default to current dir to prevent crash
            if not project_dir:
                project_dir = os.getcwd()
        
        # Create initial tasks based on workflow type
        if workflow_type == "standard":
            self._create_standard_workflow(user_request)
        elif workflow_type == "full":
            self._create_full_workflow(user_request)
        elif workflow_type == "review_only":
            self._create_review_workflow(user_request)
        elif workflow_type == "custom":
            # FIXED: Custom workflow - tasks already added by caller
            if not self.task_queue:
                print("⚠ No tasks in custom workflow queue")
                return
        elif workflow_type == "planned":
            if not PLAN_EXECUTOR_AVAILABLE:
                raise ValueError("Plan executor not available - ensure plan_executor.py is in the same directory")
            create_planned_workflow(self, user_request)
        elif workflow_type == "import":
            # Import workflow - tasks already added by project_import.create_import_workflow
            # Project directory already set up by project_import.setup_import_project_directory
            if not self.task_queue:
                print("⚠ No tasks in import workflow queue")
                return
        elif workflow_type == "bugfix":
            # Bugfix workflow - tasks already added by bugfix_workflow.create_bugfix_workflow
            # Project directory already set up by bugfix_workflow.setup_bugfix_project_directory
            if not self.task_queue:
                print("⚠ No tasks in bugfix workflow queue")
                return
        elif workflow_type == "collaborative":
            self._create_collaborative_workflow(user_request, stop_after=stop_after)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Execute tasks
        iteration = 0
        max_iterations = self.state["max_iterations"]
        
        while self.task_queue and iteration < max_iterations * 10:  # Safety limit
            iteration += 1
            ready_tasks = self.get_ready_tasks()
            
            if not ready_tasks:
                pending_tasks = [t for t in self.task_queue if t.status == TaskStatus.PENDING]
                if pending_tasks:
                    print(f"⚠ {len(pending_tasks)} tasks blocked waiting for dependencies")
                break
            
            print(f"\n▶ Iteration {iteration}: Executing {len(ready_tasks)} tasks")
            
            completed = self.execute_tasks_parallel(ready_tasks)
            
            for task in completed:
                if task in self.task_queue:
                    self.task_queue.remove(task)
                self.completed_tasks.append(task)
                
                status_symbol = "✓" if task.status == TaskStatus.COMPLETED else "✗"
                print(f"  {status_symbol} {task.task_id} ({task.assigned_role.value}): {task.status.value}")
                
                # Handle clarification tasks
                if task.task_type == "clarification" and task.status == TaskStatus.COMPLETED:
                    self._handle_clarification_interactive(task)
            
            # FIXED: Check for revision cycle after reviews complete
            review_tasks = [t for t in completed if "review" in t.task_type]
            if review_tasks:
                self._handle_revision_cycle(review_tasks)
        
        # Generate final report
        self._generate_workflow_report()
        
        # Save all outputs
        self._save_project_outputs()
        init_file = os.path.join(project_dir, "src", "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("")

        # Print project summary
        self._print_project_summary()
        
        # Cleanup Docker sandbox
        if self.sandbox:
            self.sandbox.stop()
    
    def _print_project_summary(self):
        """Print summary of project outputs"""
        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir:
            return
            
        project_name = os.path.basename(project_dir)
        
        print("\n" + "=" * 80)
        print("PROJECT OUTPUTS")
        print("=" * 80)
        print(f"\n📦 {project_name}/")
        
        for root, dirs, files in os.walk(project_dir):
            level = root.replace(project_dir, '').count(os.sep)
            indent = '  ' * level
            rel_path = os.path.basename(root)
            if level > 0:
                print(f"{indent}├── {rel_path}/")
            
            sub_indent = '  ' * (level + 1)
            for file in files:
                print(f"{sub_indent}├── {file}")
        
        print(f"\n✓ All outputs saved to: {project_dir}")
        print(f"✓ Project number: {self.state['project_info']['project_number']:03d}")
        print(f"✓ Version: {self.state['project_info']['version']}")
    
    def _create_standard_workflow(self, user_request: str):
        """Create standard coding workflow: clarify -> architect -> code -> review -> document -> verify"""
        
        # Task 1: Clarification
        self.add_task(Task(
            task_id="T001_clarify",
            task_type="clarification",
            description="Clarify requirements",
            assigned_role=AgentRole.CLARIFIER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={"user_request": user_request}
        ))
        
        # Task 2: Architecture
        self.add_task(Task(
            task_id="T002_architect",
            task_type="architecture",
            description="Design system architecture",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=9,
            dependencies=["T001_clarify"],
            metadata={"user_request": user_request}
        ))
        
        # Task 3: Coding
        self.add_task(Task(
            task_id="T003_code",
            task_type="coding",
            description="Implement the code",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=8,
            dependencies=["T002_architect"],
            metadata={"user_request": user_request}
        ))
        
        # Tasks 4-6: Multiple reviewers in parallel
        for i in range(1, 4):
            self.add_task(Task(
                task_id=f"T00{3+i}_review{i}",
                task_type="review",
                description=f"Code review #{i}",
                assigned_role=AgentRole.REVIEWER,
                status=TaskStatus.PENDING,
                priority=7,
                dependencies=["T003_code"],
                metadata={"reviewer_number": i, "user_request": user_request}
            ))

        # Task 7: Test generation (parallel with reviews)
        self.add_task(Task(
            task_id="T007_tests",
            task_type="test_generation",
            description="Generate tests",
            assigned_role=AgentRole.TESTER,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_code"],
            metadata={"user_request": user_request}
        ))

        # Task 8: Documentation
        self.add_task(Task(
            task_id="T008_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T004_review1", "T005_review2", "T006_review3"],
            metadata={"user_request": user_request}
        ))

        # Task 9: Verification
        self.add_task(Task(
            task_id="T009_verify",
            task_type="verification",
            description="Verify docs match code",
            assigned_role=AgentRole.VERIFIER,
            status=TaskStatus.PENDING,
            priority=5,
            dependencies=["T007_tests", "T008_document"]
        ))
    
    def _create_full_workflow(self, user_request: str):
        """Create comprehensive workflow with all agent types"""
        
        # Phase 1: Planning
        self.add_task(Task(
            task_id="T001_clarify",
            task_type="clarification",
            description="Clarify requirements",
            assigned_role=AgentRole.CLARIFIER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T002_architect",
            task_type="architecture",
            description="Design architecture",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=9,
            dependencies=["T001_clarify"],
            metadata={"user_request": user_request}
        ))
        
        # Phase 2: Implementation
        self.add_task(Task(
            task_id="T003_code",
            task_type="coding",
            description="Implement code",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=8,
            dependencies=["T002_architect"],
            metadata={"user_request": user_request}
        ))
        
        # Phase 3: Quality assurance (parallel)
        self.add_task(Task(
            task_id="T004_review",
            task_type="review",
            description="Code review",
            assigned_role=AgentRole.REVIEWER,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_code"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T005_security",
            task_type="security_audit",
            description="Security analysis",
            assigned_role=AgentRole.SECURITY,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_code"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T006_tests",
            task_type="test_generation",
            description="Generate tests",
            assigned_role=AgentRole.TESTER,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_code"],
            metadata={"user_request": user_request}
        ))
        
        # Phase 4: Optimization & Documentation
        self.add_task(Task(
            task_id="T007_optimize",
            task_type="optimization",
            description="Optimize code",
            assigned_role=AgentRole.OPTIMIZER,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T004_review", "T005_security"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T008_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T003_code", "T006_tests"],
            metadata={"user_request": user_request}
        ))
        
        # Phase 5: Final verification
        self.add_task(Task(
            task_id="T009_verify",
            task_type="verification",
            description="Verify docs match code",
            assigned_role=AgentRole.VERIFIER,
            status=TaskStatus.PENDING,
            priority=5,
            dependencies=["T008_document"]
        ))
    
    def _create_collaborative_workflow(self, user_request: str, stop_after: str = None):
        """Create collaborative workflow: clarify → draft plan → coder reviews → final plan → build → compliance

        Args:
            stop_after: Stop after this stage. Valid: 'clarify', 'draft_plan',
                        'plan_review', 'finalize_plan', 'build', 'compliance'.
        """

        stages = [
            ("clarify", Task(
                task_id="T001_clarify",
                task_type="clarification",
                description="Clarify requirements and produce structured job spec",
                assigned_role=AgentRole.CLARIFIER,
                status=TaskStatus.PENDING,
                priority=10,
                metadata={"user_request": user_request, "output_format": "structured"}
            )),
            ("draft_plan", Task(
                task_id="T002_draft_plan",
                task_type="draft_plan",
                description="Produce draft architecture plan from job spec",
                assigned_role=AgentRole.ARCHITECT,
                status=TaskStatus.PENDING,
                priority=9,
                dependencies=["T001_clarify"],
                metadata={"user_request": user_request}
            )),
            ("plan_review", Task(
                task_id="T003_plan_review",
                task_type="plan_review",
                description="Coder reviews draft plan before finalization",
                assigned_role=AgentRole.CODER,
                status=TaskStatus.PENDING,
                priority=8,
                dependencies=["T002_draft_plan"],
                metadata={"user_request": user_request}
            )),
            ("finalize_plan", Task(
                task_id="T004_final_plan",
                task_type="finalize_plan",
                description="Architect finalizes plan with coder feedback",
                assigned_role=AgentRole.ARCHITECT,
                status=TaskStatus.PENDING,
                priority=7,
                dependencies=["T003_plan_review"],
                metadata={"user_request": user_request}
            )),
            ("build", Task(
                task_id="T005_build",
                task_type="build_from_plan",
                description="Build all code from finalized plan",
                assigned_role=AgentRole.CODER,
                status=TaskStatus.PENDING,
                priority=6,
                dependencies=["T004_final_plan"],
                metadata={"user_request": user_request}
            )),
            ("compliance", Task(
                task_id="T006_compliance",
                task_type="compliance_review",
                description="Verify code implements the plan correctly",
                assigned_role=AgentRole.REVIEWER,
                status=TaskStatus.PENDING,
                priority=5,
                dependencies=["T005_build"],
                metadata={"user_request": user_request}
            )),
        ]

        for stage_name, task in stages:
            self.add_task(task)
            if stop_after and stage_name == stop_after:
                print(f"  ⏸ Will stop after: {stage_name}")
                break

    def _create_review_workflow(self, code: str):
        """Create workflow for reviewing existing code"""
        
        # FIXED: Store code in context
        self.state["context"]["code_to_review"] = code
        
        review_types = [
            ("correctness", AgentRole.REVIEWER),
            ("security", AgentRole.SECURITY),
            ("performance", AgentRole.REVIEWER),
            ("style", AgentRole.REVIEWER)
        ]
        
        for i, (review_type, role) in enumerate(review_types, 1):
            self.add_task(Task(
                task_id=f"T00{i}_review_{review_type}",
                task_type=f"review_{review_type}",
                description=f"Review: {review_type}",
                assigned_role=role,
                status=TaskStatus.PENDING,
                priority=10,
                metadata={"code": code, "review_focus": review_type}
            ))
    
    def _get_system_prompt(self, role: AgentRole, task_type: str, task_metadata: Dict = None) -> str:
        """Get system prompt for agent role and task type"""
        
        is_revision = task_metadata.get("is_revision", False) if task_metadata else False
        if role == AgentRole.CODER and is_revision:
            return """You are an expert programmer performing a CODE REVISION.

YOUR TASK: Fix the SPECIFIC BUGS identified by the code reviewer.

CRITICAL REVISION RULES:
1. The reviewer has found bugs in your code - fix ONLY those bugs
2. Do NOT rewrite working code. If a function works correctly, leave it alone
3. IGNORE any replacement code the reviewer suggests - write your own fix based on the bug description
4. If the reviewer's complaint is vague or doesn't describe an actual bug, keep your original code
5. Before changing anything, mentally test: does the current code actually fail? If not, don't change it
6. Output COMPLETE files, not partial patches

OUTPUT FORMAT:
- Use ### FILE: filename.py ### headers for each file
- No markdown code blocks (```)
- Config files must be pure JSON/YAML with no comments

QUALITY REQUIREMENTS:
1. The revised code must be immediately executable
2. Fix actual bugs, not style preferences
3. Preserve ALL working code - do not replace working approaches with different ones
4. Mentally test your code with real inputs before outputting
5. If the original code passes "2 + 3" and returns 5, the revised code must too

The goal is to fix bugs, not rewrite the program."""
# Here
        is_modify = task_metadata.get("mode") == "modify_existing" if task_metadata else False
        if role == AgentRole.CODER and is_modify:
            return """You are an expert programmer MODIFYING EXISTING CODE.

YOUR TASK: Make ONLY the specific changes requested. Do NOT rewrite the file.

CRITICAL RULES:
1. The existing code is provided below - PRESERVE everything not related to the requested changes
2. Make ONLY the changes explicitly requested
3. Do NOT refactor, reorganize, or "improve" unrelated code
4. Do NOT replace working implementations with mocks or stubs
5. Keep all existing imports, functions, and logic that work

OUTPUT FORMAT:
- Output the COMPLETE modified file(s)
- Use ### FILE: filename.py ### headers
- No markdown code blocks

If you're unsure whether to change something, DON'T. Only change what was explicitly requested."""       
# here   
        prompts = {
            AgentRole.ARCHITECT: """You are an expert software architect. Your role is to:
- Design solutions that MATCH the actual requirements (don't over-engineer)
- Choose the SIMPLEST approach that solves the problem
- Avoid unnecessary complexity, patterns, or frameworks
- Design for the stated use case, not imagined future needs
- Ensure the design can actually be implemented and tested

CRITICAL: If the requirements are simple, the architecture should be simple.
Don't add Singleton patterns, config files, or enterprise features unless explicitly required.

Provide clear, structured architecture documentation that matches the scope of the request.""",

            AgentRole.CLARIFIER: """You are a Requirements Clarification Agent. Your ONLY job is to ask questions.

CRITICAL INSTRUCTIONS:
1. Ask as many clarifying questions as needed to fully understand the requirements
   - Simple requests may need only 1-2 questions
   - Complex multi-component systems may need 8-10 questions
   - Don't pad with unnecessary questions, don't skip important ones
2. You MUST NOT provide solutions, code, or implementation details
3. You MUST NOT say the requirements are clear unless they truly are
4. Format your response EXACTLY as shown below

FOCUS YOUR QUESTIONS ON:
- Input/output formats and data types
- Error handling expectations (what happens when X fails?)
- Edge cases the user cares about
- External dependencies or environment constraints
- Performance or scale requirements (if relevant)

AVOID ASKING ABOUT:
- Future features not mentioned
- Multiple language support unless relevant
- Hypothetical scenarios unlikely to occur

RESPONSE FORMAT (use this exact format):
CLARIFYING QUESTIONS:
1. [Specific question about missing requirement]
2. [Specific question about unclear specification]
3. [Additional questions as needed]

ONLY if ALL requirements are crystal clear and complete, respond with:
STATUS: CLEAR - All requirements are well-defined.

Remember: Your job is to ASK QUESTIONS, not solve problems.""",

            AgentRole.CODER: """You are an expert programmer. Your role is to:
- Write code that WORKS correctly on the first try
- Test your code mentally before outputting it
- Include ONLY what's needed to meet the requirements
- Handle errors that can realistically occur
- Add comments explaining WHY, not WHAT
- Provide a simple usage example at the end

CRITICAL FILE STRUCTURE RULES:
1. If the architect specifies MULTIPLE FILES, you MUST create ALL of them
2. Output each file with a clear header: ### FILE: filename.py ###
3. Do NOT combine everything into one file unless that's what the architect designed
4. Each module/file the architect specifies MUST be created separately
5. Imports between your files must be correct (from module import X)

CRITICAL OUTPUT RULES:
1. Output ONLY valid, executable code
2. Do NOT include markdown code blocks (```) - use ### FILE: ### headers instead
3. Start each file directly with imports or docstring
4. Main entry point should have: if __name__ == "__main__":
5. Config files (JSON, YAML, etc.) must contain ONLY valid data - no comments, no explanations, no markdown

CRITICAL QUALITY RULES:
1. Your code must be EXECUTABLE and CORRECT
2. Keep it SIMPLE - match complexity to requirements
3. Do not add unrequested features
4. Handle edge cases: None inputs, missing files, etc.
5. If external files are needed, handle their absence gracefully
6. For FastAPI/uvicorn servers: use `uvicorn.run()` directly (NOT inside asyncio.run()). The if __name__ block should call uvicorn.run(app, host=host, port=port) synchronously.
7. Match ALL bracket types carefully: every ( needs ), every [ needs ], every { needs }

EXAMPLE OUTPUT FOR CONFIG FILES:
### FILE: config.json ###
{
    "color": "blue"
}

### FILE: main.py ###
\"\"\"Main entry point\"\"\"
import json
...

Focus on code that works correctly and follows the architect's file structure exactly.""",

            AgentRole.FALLBACK_CODER: """You are a SENIOR debugger called in to fix code that failed validation.

CONTEXT: Previous attempts have FAILED. You are the fallback - the last line of defense.

BEFORE WRITING ANY CODE:
1. Read the error message EXACTLY - what file, what line, what type of error?
2. Identify the ROOT CAUSE, not just the symptom
3. Check: Is this a typo? Import issue? Type mismatch? Logic error? Missing dependency?
4. Consider: Is the original approach overcomplicated? Sometimes delete and simplify.

COMMON FAILURE PATTERNS TO CHECK:
- Missing imports or wrong import paths
- Type hint mismatches (Optional[], Union[], etc.)
- Pydantic/dataclass field issues
- Enum values that don't exist
- Async/await mismatches
- Circular imports
- Off-by-one errors, wrong variable names

FIX STRATEGY:
1. Make the MINIMAL change that fixes the error
2. Don't refactor unrelated code - you'll introduce new bugs
3. If the architecture is fundamentally broken, simplify ruthlessly
4. When in doubt, add defensive checks and clear error messages

OUTPUT RULES:
1. Output ONLY valid, executable Python
2. Use ### FILE: filename.py ### headers
3. NO markdown, NO explanations outside code comments
4. Include ALL necessary imports at the top of each file

Make it work.""",

            AgentRole.REVIEWER: """You are a code reviewer. Your job is to determine if the code WORKS CORRECTLY.

REVIEW PROCESS:
1. Mentally trace through the code with sample inputs. Does it produce correct output?
2. Check for actual bugs: syntax errors, logic errors, missing imports, crashes
3. Verify it matches what the user asked for

APPROVE if:
- The code runs without errors
- It produces correct results for normal inputs
- It handles obvious error cases (empty input, division by zero, etc.)
- It meets the stated requirements

REJECT only for:
- Code that will CRASH or produce WRONG results
- Missing functionality that was explicitly required
- Bugs that would be caught by running the code

DO NOT reject for:
- Style preferences or "best practices" that don't affect correctness
- Theoretical security concerns when input is already validated
- Using standard library features (eval, exec, etc.) when the input is sanitized
- Code that works but could be written differently

CRITICAL: If the code works correctly, APPROVE IT. Working code that uses eval with validated input is better than a broken custom parser. Do not suggest replacements unless the current code is actually broken.

Return STATUS: APPROVED if the code works and meets requirements.
Return STATUS: NEEDS_REVISION only for ACTUAL BUGS with specific details of what fails and why.

Do NOT include replacement code in your review. Describe the bug, not the fix.""",

            AgentRole.TESTER: """⚠️ CRITICAL FORMAT REQUIREMENT ⚠️
Your output will be saved DIRECTLY as a .py file. The FIRST character MUST be 'i' or 'f' (from 'import' or 'from').

🚫 WRONG OUTPUT EXAMPLE (DO NOT DO THIS):
---
Here is a test for the calculator:

### FILE: test_main.py ###
```python
import pytest
from src.main import main

def test_main():
    assert main() == "Hello"
```

### TEST DESIGN RATIONALE:
This test verifies...
---

✅ CORRECT OUTPUT EXAMPLE (DO THIS):
---
import pytest
from src.main import main

def test_main():
    assert main() == "Hello"
---

See the difference? WRONG output has preamble text and markdown. CORRECT output starts immediately with 'import'.

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES - VIOLATIONS WILL CAUSE SYSTEM FAILURE:
═══════════════════════════════════════════════════════════════════════════════

1. NO PREAMBLE TEXT - Do NOT write "Here is", "This test", or ANY text before the code
2. NO MARKDOWN - Do NOT use ```python or ``` or ### FILE: or any markdown formatting
3. NO EXPLANATIONS - Do NOT add "RATIONALE", "NOTES", "HOW TO RUN", or commentary
4. NO DIRECTORY TREES - Do NOT show file structures like tests/├── test_foo.py
5. START WITH IMPORT - First line MUST be: import pytest OR from src.X import Y

If you output ANYTHING except raw Python code, the file will break and tests will fail.

═══════════════════════════════════════════════════════════════════════════════
WHAT YOUR OUTPUT MUST LOOK LIKE:
═══════════════════════════════════════════════════════════════════════════════

import pytest
from src.calculator import add, subtract

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_subtract_negative_result():
    assert subtract(3, 5) == -2

def test_add_handles_zero():
    assert add(0, 5) == 5

That's it. Just code. No commentary. No explanations. No markdown.

═══════════════════════════════════════════════════════════════════════════════
IMPORT RULES:
═══════════════════════════════════════════════════════════════════════════════

- Source code is in 'src/' directory - ALWAYS use: from src.MODULE import FUNCTION
- Example: Testing main.py → from src.main import main
- Example: Testing utils.py → from src.utils import helper_function
- Use EXACT names from the exports list provided in context - do NOT invent function names
- NEVER copy/paste source code into tests - import it

═══════════════════════════════════════════════════════════════════════════════
TEST REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════════════

- Test each public function/class listed in exports
- Include edge cases: None, empty strings, empty lists, boundary values
- Include error cases: invalid input, missing files, malformed data
- Every test function needs assertions or 'pass' - NO empty functions
- Use unittest.mock for file I/O, network calls, datetime, random, etc.
- For databases: use temp databases or mock connections

═══════════════════════════════════════════════════════════════════════════════
CRITICAL TEST ISOLATION RULES:
═══════════════════════════════════════════════════════════════════════════════

- NEVER create class instances at module level (outside functions)
  WRONG: settings = Settings()  # runs at import time, may crash
  RIGHT: def test_settings(): settings = Settings()
- NEVER share variables between test functions via module globals
  WRONG: metrics = None; def test_a(): global metrics; metrics = collect()
  RIGHT: Each test creates its own data independently
- Each test function MUST be self-contained: create its own objects, call its own methods
- For FastAPI apps, use: from fastapi.testclient import TestClient; client = TestClient(app)
  NEVER use .test_client() — that is Flask, not FastAPI
- Mock external dependencies (psutil, subprocess, network, databases) so tests run anywhere
  Example: @patch('src.collector.psutil.cpu_percent', return_value=50.0)
- For SQLite databases, use ':memory:' or tempfile.mkstemp() — never write to real files
- For classes that need Settings, mock or create Settings with test values inside each test

═══════════════════════════════════════════════════════════════════════════════
INTERACTIVE INPUT - CRITICAL:
═══════════════════════════════════════════════════════════════════════════════

- NEVER call a function that uses input() without mocking it first
- Functions like main loops, get_input(), get_number(), get_operation() call input() internally
- If you call them without mocking, the test will HANG forever waiting for stdin
- ALWAYS use: @unittest.mock.patch('builtins.input', side_effect=[...])
- Example for testing an interactive function:
    from unittest.mock import patch
    @patch('builtins.input', side_effect=['5', '+', '3', 'n'])
    def test_main_loop(mock_input):
        main()  # Now input() returns '5', '+', '3', 'n' in sequence

═══════════════════════════════════════════════════════════════════════════════
METHOD VERIFICATION - CRITICAL:
═══════════════════════════════════════════════════════════════════════════════

- ONLY call methods/functions that ACTUALLY EXIST in the source code
- Read the ACTUAL SOURCE CODE provided below carefully before writing tests
- If a function is defined at module level (e.g. get_number()), do NOT call it as a class method (calculator.get_number())
- If a class has methods add(), subtract(), multiply(), divide() - ONLY test those exact methods
- Do NOT invent methods that don't exist in the source code
- Match function signatures exactly: if get_number(prompt: str) takes a prompt, pass a string argument

NAMING: test_<function>_<scenario>
Example: test_calculate_total_with_empty_list

═══════════════════════════════════════════════════════════════════════════════
PRE-SUBMISSION CHECKLIST - VERIFY BEFORE RESPONDING:
═══════════════════════════════════════════════════════════════════════════════

☐ First character is 'i' or 'f' (from import/from)
☐ No text before the first import statement
☐ No ``` or ```python anywhere
☐ No ### FILE: headers
☐ No explanation sections (RATIONALE, NOTES, etc.)
☐ All imports use 'from src.X import Y' format
☐ Every test function has a body (not empty)
☐ Output is valid Python that can run with pytest
☐ Every function/method I call ACTUALLY EXISTS in the source code
☐ Any function that calls input() is mocked with @patch('builtins.input', ...)

If ANY checkbox is unchecked, FIX YOUR OUTPUT before submitting.

NOW OUTPUT ONLY THE RAW PYTHON TEST CODE - START WITH 'import' OR 'from':""",

            AgentRole.OPTIMIZER: """You are a performance optimization expert.

YOUR TASK:
Review the provided code and suggest concrete optimizations.

ANALYZE FOR:
- Algorithmic inefficiency (O(n²) that could be O(n), repeated lookups)
- Unnecessary operations (redundant file reads, repeated calculations)
- Memory waste (loading entire files when streaming works, large intermediate lists)
- I/O bottlenecks (synchronous operations that could be batched)

OUTPUT FORMAT:
### OPTIMIZATION ANALYSIS ###

CURRENT PERFORMANCE ISSUES:
1. [Issue]: [Explanation]
   Location: [function/line]
   Impact: [High/Medium/Low]

RECOMMENDED CHANGES:
1. [Change description]

   BEFORE_CODE_START
   [current code]
   BEFORE_CODE_END

   AFTER_CODE_START
   [optimized code]
   AFTER_CODE_END

   Benefit: [Expected improvement]

### OPTIMIZED CODE ###
[Full optimized code with all changes applied]

CRITICAL: Only suggest optimizations that preserve correctness.
Do not optimize prematurely - focus on actual bottlenecks.""",

            AgentRole.DOCUMENTER: """You are a technical documentation writer.

CRITICAL RULES:
1. Document ONLY what actually exists in the code
2. Do NOT describe features that don't exist
3. Do NOT reference files that don't exist
4. Use code examples that actually work with the provided code
5. When showing example output, describe it in words - don't show raw ANSI codes

FORMAT:
# Project Name

## Description
[1-2 sentences describing what this does]

## Installation
[Step by step setup instructions]
[List actual dependencies from imports]

## Usage
```bash
python -m src.main <args>
```
[Describe what happens when you run it]

## API Reference
### `function_name(params) -> return_type`
[Description]
- Parameters: [list with types]
- Returns: [description]

## Examples
[Working examples with expected output described in plain English]
[Example: "The text will appear in blue on supported terminals"]

## Edge Cases
[Document error handling behavior]

Base ALL documentation on the actual code provided. Verify every claim.""",

            AgentRole.DEBUGGER: """You are a debugging specialist. Your role is to:
- Analyze error messages and stack traces
- Identify root causes of bugs
- Provide specific fixes with code
- Explain why the bug occurred

Focus on actionable fixes, not general advice.""",

            AgentRole.SECURITY: """You are a security analyst. Your role is to:
- Identify security vulnerabilities (injection, XSS, path traversal, etc.)
- Check input validation and sanitization
- Verify proper error handling (no sensitive data in errors)
- Check for hardcoded credentials or secrets

Provide specific, actionable security recommendations with code fixes.""",

            AgentRole.VERIFIER: """You are a FINAL VERIFICATION AGENT.

Your job is to verify that ALL deliverables are correct and consistent:

1. DOCUMENTATION ACCURACY
   - Does README describe actual code structure?
   - Are all file paths mentioned actually present?
   - Are setup instructions accurate and complete?
   - Do examples match actual code behavior?

2. CODE COMPLETENESS
   - Are all required files present?
   - Does code match what documentation claims?
   - Are all imports valid and available?

3. TEST VALIDITY
   - Do test files import from source (not copy source code)?
   - Are tests actually runnable with pytest?
   - Do tests use correct mocking patterns?
   - Is there meaningful test coverage?

4. USER ACCEPTANCE
   - Can a user follow the README and use this project?
   - Will the code run without modification?

VERIFICATION RESULTS:
1. Documentation Accuracy: [PASS/FAIL]
2. Code Completeness: [PASS/FAIL]
3. Test Validity: [PASS/FAIL]
4. User Acceptance: [PASS/FAIL]

OUTPUT FORMAT (MANDATORY):
First non-empty line MUST be exactly one of:
VERIFICATION: PASS
VERIFICATION: FAIL

ISSUES (if any):
- [List specific issues with file names and line numbers]

RECOMMENDATION: [APPROVE / REJECT - needs revision]

SMOKE TEST INTERPRETATION:
- If pytest fails with actual test failures: FAIL
- If pytest fails due to import errors: Check if imports match actual file structure
- If entry point test is missing: Project may be a library (no entry point needed)
- If entry point test fails: Check project type - not all projects are CLI apps

Be strict on code correctness, but flexible on project structure.
Broken tests or inaccurate docs are grounds for FAIL.
Missing CLI is NOT a failure if the project is a library or subprocess tool."""
        }
        
        return prompts.get(role, "You are a helpful AI assistant.")
    
    def _build_user_message(self, task: Task) -> str:
        """Build user message with context for the task"""
        
        # Check if we need to reduce context (retry after overflow)
        reduce_context = task.metadata.get("reduce_context", False)
        max_context_per_dep = 2000 if reduce_context else 8000
        
        message_parts = [f"TASK: {task.description}\n"]
        
        # FIXED: Always include user request from context or metadata
        user_request = (
            task.metadata.get('user_request') or 
            self.state["context"].get("user_request", "")
        )
        if user_request:
            # Truncate user request if reducing context
            if reduce_context and len(user_request) > 1000:
                user_request = user_request[:1000] + "..."
            message_parts.append(f"USER REQUEST:\n{user_request}\n")
        
        # Add clarification context if available
        if 'clarification' in self.state["context"]:
            clarification = self.state["context"]['clarification']
            if clarification and clarification != "Requirements are clear and complete.":
                message_parts.append(f"\nCLARIFICATION:\n{clarification}\n")
        # Add existing code for modify tasks
        if task.metadata.get("mode") == "modify_existing":
            existing_code = self.state["context"].get("existing_code", "")
            if existing_code:
                message_parts.append(f"\nEXISTING CODE TO MODIFY:\n{existing_code}\n")
        # Add code if in metadata (for reviews)
        if "code" in task.metadata:
            message_parts.append(f"\nCODE TO ANALYZE:\n{task.metadata['code']}\n")
        
        # Add revision feedback if present
# --- START CHANGE 1 ---
        # Add revision feedback if present
        if "revision_feedback" in task.metadata:
            feedback = task.metadata['revision_feedback']
            original_code = task.metadata.get('original_code', '')
            
            # Try to extract corrected code if reviewer provided it
            corrected_code_match = re.search(r'```python\s*(.*?)```', feedback, re.DOTALL)
            
            message_parts.append("""
================================================================================
                           ⚠️  REVISION REQUIRED  ⚠️
================================================================================

Your previous code was REJECTED by the reviewer. You MUST fix the issues.

CRITICAL INSTRUCTIONS:
1. Read the reviewer's feedback carefully - every issue mentioned must be fixed
2. If the reviewer provided corrected code, USE IT as your starting point
3. Do NOT ignore any feedback - all issues are blockers
4. Output COMPLETE revised code using ### FILE: filename.py ### format
5. The revised code must be immediately executable

================================================================================
""")
            
            if corrected_code_match:
                corrected_code = corrected_code_match.group(1).strip()
                message_parts.append(f"""
REVIEWER PROVIDED CORRECTED CODE - USE THIS AS YOUR BASE:
================================================================================
{corrected_code}
================================================================================

Apply this corrected code. Verify it addresses all issues mentioned in the review.
If needed, make additional improvements while preserving the reviewer's fixes.
""")
            
            message_parts.append(f"""
FULL REVIEWER FEEDBACK:
--------------------------------------------------------------------------------
{feedback}
--------------------------------------------------------------------------------
""")
            
            message_parts.append(f"""
YOUR ORIGINAL CODE (for reference - DO NOT just repeat this):
--------------------------------------------------------------------------------
{original_code}
--------------------------------------------------------------------------------

OUTPUT YOUR COMPLETE REVISED CODE BELOW:
""")
# --- END CHANGE 1 ---        

        # Add context from dependent tasks
        if task.dependencies:
            message_parts.append("\nCONTEXT FROM PREVIOUS TASKS:")
            for dep_id in task.dependencies:
                dep_task = next((t for t in self.completed_tasks if t.task_id == dep_id), None)
                if dep_task and dep_task.result:
                    message_parts.append(f"\n[{dep_task.task_id} - {dep_task.assigned_role.value}]")
                    # Use reduced context limit if retrying
                    limit = max_context_per_dep
                    result = dep_task.result[:limit] + "..." if len(dep_task.result) > limit else dep_task.result
                    message_parts.append(result)
        
        # FIXED: For documenter, ensure code is included (with smart truncation)
        if task.task_type == "documentation":
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")), None)
            if code_task and code_task.result and "coding" not in str(task.dependencies):
                # For multi-file projects, provide structure summary + key file contents
                files_dict = self._parse_multi_file_output(code_task.result)
                
                # Adjust limits based on reduce_context flag
                if reduce_context:
                    priority_limit = 1200
                    signature_limit = 800
                    single_file_limit = 3000
                else:
                    priority_limit = 2500
                    signature_limit = 1500
                    single_file_limit = 6000
                
                if files_dict:
                    message_parts.append("\n" + "=" * 60)
                    message_parts.append("CODE TO DOCUMENT (Summarized for context)")
                    message_parts.append("=" * 60)
                    
                    # List all files with their exports
                    message_parts.append("\n### FILES AND EXPORTS ###")
                    for fname, content in files_dict.items():
                        exports = self._extract_exports_from_code(content)
                        message_parts.append(f"\n{fname}: exports {exports}")
                    
                    # Provide full content for key files only (main.py, config, etc.)
                    priority_files = ['main.py', 'config.py', 'api.py']
                    for pfile in priority_files:
                        if pfile in files_dict:
                            content = files_dict[pfile]
                            # Truncate if very long
                            if len(content) > priority_limit:
                                content = content[:priority_limit] + "\n# ... (truncated)"
                            message_parts.append(f"\n### {pfile} (FULL) ###")
                            message_parts.append(content)
                    
                    # For other files, just show docstrings/signatures
                    message_parts.append("\n### OTHER FILES (Signatures Only) ###")
                    for fname, content in files_dict.items():
                        if fname not in priority_files:
                            # Extract just the function/class signatures
                            signatures = self._extract_signatures(content)
                            message_parts.append(f"\n--- {fname} ---")
                            message_parts.append(signatures[:signature_limit] if len(signatures) > signature_limit else signatures)
                else:
                    # Single file - include with truncation
                    code = code_task.result
                    if len(code) > single_file_limit:
                        code = code[:single_file_limit] + "\n# ... (truncated)"
                    message_parts.append(f"\nACTUAL CODE TO DOCUMENT:\n{code}\n")
        
        # FIXED: For security audit, ensure code is included
        if task.task_type == "security_audit":
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")), None)
            if code_task and code_task.result:
                message_parts.append(f"\nCODE TO ANALYZE FOR SECURITY:\n{code_task.result}\n")
        
        # Add review focus if specified
        if "review_focus" in task.metadata:
            message_parts.append(f"\nFOCUS YOUR REVIEW ON: {task.metadata['review_focus']}")
        
# Add reviewer number if specified
        if "reviewer_number" in task.metadata:
            message_parts.append(f"\nYou are reviewer #{task.metadata['reviewer_number']}")
        
        # --- FIX FOR TESTER: Provide actual source filenames ---
        # --- FIX FOR TESTER: Provide actual source code and exports ---
        if task.task_type == "test_generation":
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")), None)
            if code_task and code_task.result:
                files_dict = self._parse_multi_file_output(code_task.result)
                project_name = self.state["project_info"].get("project_name", "project")
                
                message_parts.append("\n" + "=" * 70)
                message_parts.append("CRITICAL: ACTUAL SOURCE CODE TO TEST")
                message_parts.append("=" * 70)
                
                if files_dict:
                    py_files = [f for f in files_dict.keys() if f.endswith('.py')]
                    message_parts.append(f"\nSource files: {', '.join(py_files)}")
                    
                    # Provide actual exports for each file
                    message_parts.append("\n### ACTUAL EXPORTS BY FILE ###")
                    for fname in py_files:
                        content = files_dict[fname]
                        exports = self._extract_exports_from_code(content)

                        module_path = fname.replace("\\", "/").replace(".py", "")
                        if module_path.startswith("src/"):
                            module_path = module_path[len("src/"):]
                        module_path = module_path.replace("/", ".")
                        import_mod = f"src.{module_path}"

                        message_parts.append(f"\n{fname}:")
                        message_parts.append(f"  Import as: from {import_mod} import {', '.join(exports) if exports else 'N/A'}")
                        message_parts.append(f"  Exports: {exports}")


                        # Map file path -> python import module, assuming all sources live under project_root/src/
                        module_path = fname.replace("\\", "/").replace(".py", "").lstrip("./")

                        # If the file key already includes 'src/', strip it; otherwise assume it's under src/
                        if module_path.startswith("src/"):
                            module_path = module_path[len("src/"):]

                        module_path = module_path.replace("/", ".")
                        import_mod = f"src.{module_path}"

                        message_parts.append(
                            f"  Import as: from {import_mod} import {', '.join(exports) if exports else 'N/A'}"
                        )

                    
                    # Include actual code for key files (main.py, services, etc.)
                    message_parts.append("\n### ACTUAL SOURCE CODE ###")
                    priority_files = ['main.py'] + [f for f in py_files if 'service' in f.lower()]
                    other_files = [f for f in py_files if f not in priority_files]
                    
                    for fname in priority_files + other_files:
                        if fname in files_dict:
                            content = files_dict[fname]
                            # Truncate very long files but show structure
                            if len(content) > 3000:
                                content = content[:3000] + "\n... (truncated)"
                            message_parts.append(f"\n--- {fname} ---")
                            message_parts.append(content)
                else:
                    message_parts.append(f"\nSingle source file: {project_name}.py")
                    # Include the actual code
                    message_parts.append("\n--- Source Code ---")
                    message_parts.append(code_task.result[:5000])
                
                message_parts.append("\n" + "=" * 70)
                message_parts.append("IMPORT CORRECTLY: Use the EXACT export names shown above!")
                message_parts.append("DO NOT invent functions that don't exist!")
                message_parts.append("=" * 70)
        
        # --- FIX FOR DOCUMENTER: Provide actual file paths ---
        if task.task_type == "documentation":
            project_name = self.state["project_info"].get("project_name", "project")
            
            message_parts.append("\n" + "=" * 70)
            message_parts.append("CRITICAL: ACTUAL FILE PATHS FOR DOCUMENTATION")
            message_parts.append("=" * 70)
            
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")), None)
            if code_task and code_task.result:
                files_dict = self._parse_multi_file_output(code_task.result)
                if files_dict:
                    message_parts.append("\nSource files (in src/ directory):")
                    for fname in files_dict.keys():
                        message_parts.append(f"  - src/{fname}")
                    message_parts.append("\nIMPORTANT: Run commands from the PROJECT ROOT (the folder that contains src/).")
                    message_parts.append("Use module execution:")
                    message_parts.append("  python -m src.main <command> [args]")
                    message_parts.append("Do NOT `cd src/` and run `python main.py` unless the project explicitly supports it.")
                else:
                    message_parts.append(f"\nSource file: src/{project_name}.py")
            
            message_parts.append(f"\nTest file: tests/test_{project_name}.py")
            message_parts.append("\nDO NOT reference task IDs like 'T006_tests.py' - use actual filenames!")
            message_parts.append("=" * 70)
        
        return "\n".join(message_parts)
    
    def _update_context(self, task):
        """Update shared context with task results - ENHANCED VERSION."""
        context_key = f"{task.task_type}_{task.assigned_role.value}"
        self.state["context"][context_key] = {
            "task_id": task.task_id,
            "result": task.result,
            "completed_at": task.completed_at
        }
    
        # Store latest code separately
        if task.task_type in ("coding", "revision", "plan_execution", "build_from_plan") and task.result:
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

    def _get_verifier_prompt(self, task: Task) -> tuple[str, str]:
        """Generate prompt for verifier agent (includes smoke test results)."""

        project_dir = self.state["project_info"].get("project_dir", "")
        project_name = os.path.basename(project_dir) if project_dir else "project"

        # Get documentation from completed tasks
        doc_task = next(
            (t for t in self.completed_tasks if t.task_type == "documentation"),
            None,
        )
        readme_content = (
            doc_task.result
            if (doc_task and doc_task.status == TaskStatus.COMPLETED)
            else ""
        )
        doc_failed = bool(doc_task and doc_task.status == TaskStatus.FAILED)

        # Get actual code (multi-file output string from coder/plan executor)
        code_task = next(
            (t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")),
            None,
        )
        code_content = code_task.result if code_task else ""

        # Build file inventory from disk
        file_inventory: list[str] = []
        if project_dir and os.path.exists(project_dir):
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, project_dir)
                    file_inventory.append(rel_path)

        system_prompt = self._get_system_prompt(AgentRole.VERIFIER, "verification")

        # Use smoke tests from task metadata (preferred) or run them
        smoke_tests = task.metadata.get("smoke_tests")
        if smoke_tests is None:
            smoke_tests = self._run_smoke_tests()

        def _fmt_smoke(name: str, info: dict) -> str:
            cmd = info.get("cmd", "")
            rc = info.get("returncode", None)
            stdout = (info.get("stdout") or "").strip()
            stderr = (info.get("stderr") or "").strip()
            if len(stdout) > 1500:
                stdout = stdout[:1500] + "\n... (truncated)"
            if len(stderr) > 1500:
                stderr = stderr[:1500] + "\n... (truncated)"
            parts = [
                f"- {name}",
                f"  cmd: {cmd}",
                f"  returncode: {rc}",
            ]
            if stdout:
                parts.append("  stdout:")
                parts.append("  " + "\n  ".join(stdout.splitlines()))
            if stderr:
                parts.append("  stderr:")
                parts.append("  " + "\n  ".join(stderr.splitlines()))
            return "\n".join(parts)

        smoke_section = "SMOKE TESTS:\n"
        if isinstance(smoke_tests, dict) and smoke_tests:
            smoke_section += "\n".join(_fmt_smoke(k, v) for k, v in smoke_tests.items())
        else:
            smoke_section += "(no smoke test results available)"

        # Check for integration issues in entry point (guarded)
        integration_issues = ""
        try:
            integration_issues = self._check_entry_point_integration(code_content)
        except Exception as e:
            integration_issues = f"Integration check error: {e}"

        # Adjust verification scope if documentation failed
        if doc_failed:
            readme_section = """README CONTENT:
Documentation generation failed - cannot verify documentation.
Skip documentation accuracy checks and focus on code completeness and runtime behavior only."""
        elif readme_content:
            readme_section = f"""README CONTENT:
{readme_content[:4000]}"""
        else:
            readme_section = """README CONTENT:
No README found - documentation may not have been generated yet."""

        user_message = f"""PROJECT: {project_name}

FILES IN PROJECT:
{chr(10).join(f"  • {f}" for f in sorted(file_inventory)) if file_inventory else "  (files not yet saved)"}

INTEGRATION CHECK:
{integration_issues if integration_issues else "No obvious integration issues detected."}

{smoke_section}

{readme_section}

ACTUAL CODE (truncated):
{code_content[:8000] if code_content else "No code found"}

Perform final verification:

1. {("Skip - documentation not available" if doc_failed or not readme_content else "Does the README accurately describe the actual code and usage?")}
2. Are there any integration issues noted above that need fixing?
3. Do smoke tests indicate runtime/test failures? If yes: FAIL.

OUTPUT FORMAT (MANDATORY):
- First non-empty line MUST be exactly one of:
  VERIFICATION: PASS
  VERIFICATION: FAIL
- Then provide 3-8 bullet points explaining why.


NOTE: {("Documentation generation failed, so focus only on code quality and runtime behavior." if doc_failed else "Verify both documentation and code.")}"""

        return system_prompt, user_message


    def _generate_workflow_report(self):
        """Generate and print workflow execution report"""
        print("\n" + "=" * 80)
        print("WORKFLOW EXECUTION REPORT")
        print("=" * 80)
        
        total_tasks = len(self.completed_tasks)
        successful = sum(1 for t in self.completed_tasks if t.status == TaskStatus.COMPLETED)
        failed = total_tasks - successful
        
        print(f"\n📊 Task Summary:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   ✓ Successful: {successful}")
        print(f"   ✗ Failed: {failed}")
        
        if self.completed_tasks:
            completed_with_time = [t for t in self.completed_tasks if t.completed_at]
            if completed_with_time:
                total_time = max(t.completed_at for t in completed_with_time) - \
                            min(t.created_at for t in self.completed_tasks)
                print(f"   ⏱ Total time: {total_time:.2f}s")
        
        # Agent metrics
        print(f"\n📈 Agent Performance:")
        for agent_name, metrics in self.executor.metrics.items():
            success_rate = (metrics.successful_calls / metrics.total_calls * 100) if metrics.total_calls > 0 else 0
            print(f"   {agent_name}:")
            print(f"      Calls: {metrics.total_calls} (✓ {metrics.successful_calls}, ✗ {metrics.failed_calls})")
            print(f"      Success rate: {success_rate:.1f}%")
            print(f"      Avg response time: {metrics.avg_response_time:.2f}s")
            print(f"      Total tokens: {metrics.total_tokens}")
        
        # Task details
        print(f"\n📋 Task Details:")
        for task in self.completed_tasks:
            status_symbol = "✓" if task.status == TaskStatus.COMPLETED else "✗"
            duration = (task.completed_at - task.created_at) if task.completed_at else 0
            print(f"   {status_symbol} {task.task_id}: {task.description}")
            print(f"      Role: {task.assigned_role.value} | Duration: {duration:.2f}s")
            if task.error:
                print(f"      Error: {task.error}")
    
    def save_state(self, filename: Optional[str] = None):
        """Save workflow state to file"""
        project_dir = self.state.get("project_info", {}).get("project_dir")
        
        if filename is None:
            if project_dir:
                filename = os.path.join(project_dir, "session_state.json")
            else:
                filename = f"swarm_state_{self.state['workflow_id']}.json"
        
        # Serialize tasks properly
        def serialize_task(t: Task) -> dict:
            d = asdict(t)
            d['assigned_role'] = t.assigned_role.value
            d['status'] = t.status.value
            return d
        
        state_data = {
            "workflow_id": self.state["workflow_id"],
            "phase": self.state["phase"],
            "iteration": self.state["iteration"],
            "context": {k: v for k, v in self.state["context"].items() 
                       if not isinstance(v, dict) or 'result' not in v or len(str(v.get('result', ''))) < 10000},
            "history": self.state["history"],
            "project_info": self.state.get("project_info", {}),
            "completed_tasks": [serialize_task(t) for t in self.completed_tasks],
            "metrics": {k: asdict(v) for k, v in self.executor.metrics.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        # Also save to sessions directory
        sessions_dir = "sessions"
        if not os.path.exists(sessions_dir):
            os.makedirs(sessions_dir)
        session_file = os.path.join(sessions_dir, f"swarm_state_{self.state['workflow_id']}.json")
        with open(session_file, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            "agents": {k: asdict(v) for k, v in self.executor.metrics.items()},
            "tasks": {
                "total": len(self.completed_tasks),
                "completed": sum(1 for t in self.completed_tasks if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in self.completed_tasks if t.status == TaskStatus.FAILED)
            }
        }


def main():
    """Example usage"""
    coordinator = SwarmCoordinator()
    
    user_request = """
    Create a Python function that takes a list of numbers and returns:
    1. The median value
    2. The standard deviation
    3. Outliers (values > 2 std devs from mean)
    
    Include error handling for edge cases.
    """
    
    coordinator.run_workflow(user_request, workflow_type="standard")


if __name__ == "__main__":
    main()