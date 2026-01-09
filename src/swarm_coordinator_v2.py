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
    from plan_executor_v2 import (
        PlanExecutor, 
        create_planned_workflow,
        execute_plan_task,
        ARCHITECT_PLAN_SYSTEM_PROMPT,
        get_architect_plan_prompt,
        extract_yaml_from_response
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
        
    def _get_agent_config(self, role: AgentRole) -> Dict:
        """Get configuration for specific agent role"""
        role_str = role.value
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
    
    def execute_agent(self, role: AgentRole, system_prompt: str, user_message: str, 
                      agent_params: Optional[Dict] = None) -> str:
        """Execute a single agent with the given prompts"""
        
        # Get agent configuration
        agent_config = self._get_agent_config(role)
        url = agent_config['url']
        model = agent_config.get('model', 'local-model')
        api_type = agent_config.get('api_type', 'openai')
        timeout = agent_config.get('timeout', 7200)
        
        # Merge parameters
        params = self.config.get('agent_parameters', {}).get(role.value, {}).copy()
        if agent_params:
            params.update(agent_params)
        
        # Initialize metrics if needed
        agent_key = role.value
        with self.metrics_lock:
            if agent_key not in self.metrics:
                self.metrics[agent_key] = AgentMetrics(agent_name=agent_key, role=role.value)
        
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
            raise Exception(f"Agent {role.value} failed: {str(e)}")
            
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
        code_task = next((t for t in reversed(self.completed_tasks) if t.task_type in ("coding", "plan_execution", "revision")), None)
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
    
        # Pattern 1: ### FILE: filename.py ###
        file_pattern = r'###\s*FILE:\s*([^\s#]+)\s*###'
        matches = list(re.finditer(file_pattern, code_output))
    
        # Pattern 2: #### File: `filename.py` ... ```python ... ```
        # More robust version handles case variations and optional language markers
        if not matches:
            md_pattern = r'#{1,4}\s*[Ff]ile:\s*`([^`]+)`.*?\n\s*```(?:python|py)?\s*\n(.*?)```'
            md_matches = list(re.finditer(md_pattern, code_output, re.DOTALL))
            if md_matches:
                files = {}
                for match in md_matches:
                    filename = match.group(1).strip()
                    content = match.group(2).strip()
                    if content:
                        files[filename] = content
                return files
    
        if not matches:
            return {}  # Single file output
    
        files = {}
        for i, match in enumerate(matches):
            filename = match.group(1).strip()
            start_pos = match.end()
            
            # Find end position (start of next file or end of string)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(code_output)
            
            content = code_output[start_pos:end_pos].strip()
            
            # Clean any remaining markdown artifacts
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

    def _clean_file_content(self, content: str) -> str:
        """Clean individual file content from markdown artifacts"""
        lines = content.split('\n')
        cleaned = []
        
        for line in lines:
            # Skip markdown code block markers
            if line.strip().startswith('```'):
                continue
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
        
        # Save requirements.txt
        self._create_requirements_txt(project_dir)
        
        # Save session state
        self.save_state()
    
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
                parts = line.split()
                if len(parts) > 1:
                    mod = parts[1].split('.')[0].split(',')[0]
                    imports.add(mod)

        stdlib = {'os', 'sys', 'json', 'time', 'datetime', 're', 'typing', 'logging', 'argparse', 'abc', 'enum', 'uuid', 'subprocess'}

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
            "pytest": "pytest>=8.0.0"
        }

        package_map = {
            "dotenv": "python_dotenv",
            "multipart": "python_multipart",
            "jose": "python_jose",
            "jwt": "python_jose",
            "psycopg2": "psycopg2" 
        }

        final_reqs = set()
        for imp in imports:
            if imp in stdlib: continue
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

            # 3. Run CLI Help in Docker
            entry_point = self.state.get("project_info", {}).get("entry_point", "src/main.py")
            # Convert src/main.py -> src.main
            module_path = entry_point.replace("/", ".").replace("\\", ".").rstrip(".py")
            if module_path.endswith("."): module_path = module_path[:-1]
            if not module_path.startswith("src."): module_path = f"src.{module_path}"

            cli_res = self.sandbox._exec(
                ["python", "-m", module_path, "--help"],
                timeout=10
            )
            results["cli_help"] = {
                "returncode": cli_res.returncode,
                "stdout": cli_res.stdout,
                "stderr": cli_res.stderr,
                "cmd": f"python -m {module_path} --help (docker)"
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

        # Run CLI Help
        try:
            entry_point = self.state.get("project_info", {}).get("entry_point", "src/main.py")
            module_path = entry_point.replace("/", ".").replace("\\", ".").rstrip(".py")
            if module_path.endswith("."): module_path = module_path[:-1]
            
            # Patch 3 Logic: Ensure src. prefix
            if os.path.exists(os.path.join(project_dir, "src")) and not module_path.startswith("src."):
                 module_path = f"src.{module_path}"
            
            res = subprocess.run(
                [sys.executable, "-m", module_path, "--help"],
                cwd=project_dir, env=env, capture_output=True, text=True, timeout=10
            )
            results["cli_help"] = {
                "returncode": res.returncode, 
                "stdout": res.stdout, 
                "stderr": res.stderr, 
                "cmd": f"python -m {module_path} --help"
            }
        except Exception as e:
            results["cli_help"] = {"returncode": -1, "stderr": str(e), "cmd": "cli_help check"}
            
        return results
    # -------------------------------------------------------------------------
    # UPDATED EXECUTE_TASK
    # -------------------------------------------------------------------------
    def execute_task(self, task: Task) -> Task:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS
        
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
                    print("\n" + "=" * 80)
                    print("❌ SMOKE TESTS FAILED - AUTO-REJECTING")
                    print("=" * 80)
                    print("Smoke tests failed:")
                    for item in failing_list:
                        print(f" - {item}")
                    print("\nSmoke test details:")
                    for name, info in smoke_tests.items():
                        print("-" * 40)
                        print(f"{name}: {info.get('cmd')}")
                        print(f"returncode: {info.get('returncode')}")
                        stdout = (info.get("stdout") or "").strip()
                        stderr = (info.get("stderr") or "").strip()
                        if stdout:
                            print("stdout:")
                            print(stdout[:4000])
                        if stderr:
                            print("stderr:")
                            print(stderr[:4000])
                    print("=" * 80)

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
                        "max_tokens": 2000,
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
                user_message = get_architect_plan_prompt(
                    task.metadata.get('user_request', ''), 
                    self.state["context"].get('clarification', '')
                    )
                architect_config = self.executor._get_agent_config(AgentRole.ARCHITECT)
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
                    raise Exception(output.get("error"))
                result_text = output.get("result", "")
                plan_yaml = extract_yaml_from_response(result_text)
                self.state["context"]["plan_yaml"] = plan_yaml
                task.result = plan_yaml
                self._set_project_type_from_plan(plan_yaml)
                task.status = TaskStatus.COMPLETED
            else:
                system_prompt = self._get_system_prompt(AgentRole.ARCHITECT, task.task_type, task.metadata)
                user_message = self._build_user_message(task)
                task.result = self.executor.execute_agent(AgentRole.ARCHITECT, system_prompt, user_message)
                task.status = TaskStatus.COMPLETED
                
                task.completed_at = time.time()
                self._update_context(task)
                return task

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
                        "max_tokens": 6000,
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
                            passed, missing = self._verify_coder_files(architect_task.result, result_text)
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
                        "temperature": 0.8,
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
                        "temperature": 0.7,
                        "max_tokens": 4000,
                        "timeout": tester_config.get("timeout", 600)
                    }
                }

                try:
                    output = self._run_external_agent("tester", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))
                    
                    result_text = self._clean_test_output(output.get("result", ""))
                    task.result = result_text
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                task.completed_at = time.time()
                self._update_context(task)
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
                        "max_tokens": 4000,
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
        except Exception as e:
            # Global exception handler
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
        return '\n'.join(cleaned_lines).rstrip()

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
                "max_tokens": 3000,
                "timeout": clarifier_config.get("timeout", 300)
            }
        }
        
        try:
            synthesis_output = self._run_external_agent("clarifier", synthesis_payload)
            job_scope = synthesis_output.get("job_scope", "")
            
            if not job_scope:
                # Fallback to raw Q&A if synthesis failed
                print("⚠ Synthesis failed, using raw Q&A")
                job_scope = f"ORIGINAL REQUEST:\n{user_request}\n\nCLARIFICATION Q&A:\n{result}\n\nANSWERS:\n{answers_text}"
            else:
                print("✓ Job scope synthesized")
                
        except Exception as e:
            print(f"⚠ Synthesis error: {e}, using raw Q&A")
            job_scope = f"ORIGINAL REQUEST:\n{user_request}\n\nCLARIFICATION Q&A:\n{result}\n\nANSWERS:\n{answers_text}"
        
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
        
        # Find the coding task
        code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "plan_execution")), None)
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
        
# --- START CHANGE 2 ---
        # Create revision task
        revision_task = Task(
            task_id=f"T_revision_{code_task.revision_count + 1}",
            task_type="revision",  # Changed from "coding" to "revision"
            description="Revise code based on review feedback",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={
                "revision_feedback": "\n\n".join(feedback),
                "original_code": code_task.result,
                "user_request": self.state["context"].get("user_request", ""),
                "is_revision": True  # Flag for system prompt
            }
        )
# --- END CHANGE 2 ---
        revision_task.revision_count = code_task.revision_count + 1
        
        # Execute revision
        self.execute_task(revision_task)
        
        if revision_task.status == TaskStatus.COMPLETED:
            # Update the context with revised code
            self._update_context(revision_task)
            self.completed_tasks.append(revision_task)
            return True
        
        return False
    
    def run_workflow(self, user_request: str, workflow_type: str = "standard"):
        """Execute a complete workflow"""
        print("=" * 80)
        print(f"ADVANCED SWARM COORDINATOR v2")
        print(f"Workflow: {workflow_type}")
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
        
        # Task 7: Documentation
        self.add_task(Task(
            task_id="T007_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T004_review1", "T005_review2", "T006_review3"],
            metadata={"user_request": user_request}
        ))
        
        # Task 8: Verification
        self.add_task(Task(
            task_id="T008_verify",
            task_type="verification",
            description="Verify docs match code",
            assigned_role=AgentRole.VERIFIER,
            status=TaskStatus.PENDING,
            priority=5,
            dependencies=["T007_document"]
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

YOUR TASK: Fix the issues identified by the code reviewer.

CRITICAL REVISION RULES:
1. The reviewer has REJECTED your previous code - you MUST address their feedback
2. If the reviewer provided corrected code, USE IT - don't reinvent
3. Every issue mentioned in the review MUST be fixed
4. Do not argue with the review - implement the fixes
5. Output COMPLETE files, not partial patches

OUTPUT FORMAT:
- Use ### FILE: filename.py ### headers for each file
- No markdown code blocks (```)
- Config files must be pure JSON/YAML with no comments

QUALITY REQUIREMENTS:
1. The revised code must be immediately executable
2. All reviewer concerns must be addressed
3. Preserve working parts of the original code
4. Test mentally before outputting

Focus on implementing the reviewer's feedback completely and correctly."""  
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

            AgentRole.REVIEWER: """You are a code reviewer. Your role is to:
- Find BUGS - syntax errors, logic errors, missing imports, unhandled exceptions
- Verify code actually WORKS - trace through execution mentally
- Check if code matches requirements (not over-engineered, not under-engineered)
- Identify missing error handling for realistic failures
- Verify external dependencies are handled correctly

CRITICAL CHECKS:
1. Will this code run without errors? Test mentally.
2. What if required files don't exist? What if inputs are None/empty?
3. Does this match what the user asked for?
4. Are there syntax errors, typos, or undefined variables?
5. Will the code produce the expected output?

Return STATUS: APPROVED if code is correct and matches requirements.
Return STATUS: NEEDS_REVISION with SPECIFIC bugs/issues if found.

Be thorough - broken code getting through is a critical failure.""",

            AgentRole.TESTER: """You are a test engineer generating pytest test suites.

CRITICAL OUTPUT RULES:
1. Output ONLY raw Python code - no markdown, no ```python blocks, no explanations
2. NEVER copy source code into test files - always import from the source module
3. The first line must be an import statement, not a markdown fence
4. The output must be directly saveable as a .py file and runnable with pytest
5. NO EMPTY FUNCTIONS. If you need a placeholder, you MUST use 'pass'.
   WRONG: def test_foo():
   RIGHT: def test_foo(): pass

IMPORT RULES:
- Import the functions you're testing from the ACTUAL source modules provided in context
- Use the EXACT function/class names shown in the exports list - do not invent names
- For nested modules like models/asset.py, import as: from models.asset import Asset
- NEVER redefine or copy the functions being tested
- The project source code is located in a 'src' directory.
- ALWAYS prefix your imports with 'src.'.
- Example: If testing 'main.py', import as 'from src.main import main'
- Example: If testing 'utils.py', import as 'from src.utils import helper'
- NEVER import directly like 'from main import main' - this will fail.

STRUCTURE:
- Start with imports: pytest, unittest.mock, then the source modules
- Use pytest fixtures for common setup (mock configs, temp files, temp databases)
- Group related tests in classes if appropriate
- Ensure every test function has a body (assertions or pass)

TEST COVERAGE REQUIREMENTS:
- Unit tests for each public function/method listed in the exports
- Edge cases: None, empty strings, empty collections, boundary values
- Error conditions: missing files, invalid input, malformed data
- For database operations: use temporary databases or mock the connection

MOCKING GUIDELINES:
- Use unittest.mock.patch for external dependencies (file I/O, datetime, etc.)
- Use mock_open for file operations
- Mock database connections to avoid side effects
- When testing CLI, mock sys.argv
- When mocking datetime, patch it where it's used: @patch('module.datetime')

REGEX AND CONFIG TESTING:
- NEVER assert exact string equality on regex patterns from config files
- String escaping differs between YAML parsing and Python literals
- Instead, test that regex patterns WORK correctly:
  WRONG: assert config['regex'] == "\\\\bERROR\\\\b"
  RIGHT: assert re.search(config['regex'], "ERROR occurred") is not None
  RIGHT: assert re.search(config['regex'], "ERRORS") is None  # word boundary works
- For config values, test behavior not exact string representation
- If testing config loading, verify the loaded config can be USED correctly

NAMING:
- test_<function_name>_<scenario>
- Example: test_create_asset_valid_name, test_create_asset_empty_name_raises

OUTPUT: Raw Python code only. No preamble. No markdown. No explanation.""",

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