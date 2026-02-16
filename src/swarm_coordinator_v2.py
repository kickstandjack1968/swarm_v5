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
    CODER_2 = "coder_2"
    CODER_3 = "coder_3"
    CODER_4 = "coder_4"
    TOOLSMITH = "toolsmith"


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
                AgentRole.VERIFIER: 'verifier',
                AgentRole.CODER_2: 'coder_2',
                AgentRole.CODER_3: 'coder_3',
                AgentRole.CODER_4: 'coder_4',
                AgentRole.TOOLSMITH: 'toolsmith'
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
                "architect": {"temperature": 0.5, "max_tokens": 35000, "top_p": 0.85},
                "clarifier": {"temperature": 0.7, "max_tokens": 35000, "top_p": 0.9},
                "coder": {"temperature": 0.2, "max_tokens": 35000, "top_p": 0.85},
                "reviewer": {"temperature": 0.8, "max_tokens": 35000, "top_p": 0.95},
                "tester": {"temperature": 0.7, "max_tokens": 35000, "top_p": 0.9},
                "optimizer": {"temperature": 0.6, "max_tokens": 35000, "top_p": 0.85},
                "documenter": {"temperature": 0.7, "max_tokens": 35000, "top_p": 0.9},
                "debugger": {"temperature": 0.6, "max_tokens": 35000, "top_p": 0.85},
                "security": {"temperature": 0.8, "max_tokens": 35000, "top_p": 0.95},
                "verifier": {"temperature": 0.3, "max_tokens": 35000, "top_p": 0.85}
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
        
        # Generate empty __init__.py for each (PlanExecutor creates proper ones with
        # explicit named imports — we just ensure the file exists for packages that need it)
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
        
        # Save code - merge all code-producing tasks (build + revisions)
        # Earlier tasks provide the baseline, later revisions override specific files
        code_tasks = [t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan") and t.result]
        files_dict = {}
        for code_task in code_tasks:
            parsed = self._parse_multi_file_output(code_task.result)
            if parsed:
                files_dict.update(parsed)  # Later tasks override earlier ones

        if not files_dict and code_tasks:
            # Single file output fallback — use last task
            code_task = code_tasks[-1]

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

        elif code_tasks:
            # Single file output (legacy behavior) - assume source
            project_name = self.state["project_info"]["project_name"]
            code_file = os.path.join(project_dir, "src", f"{project_name}.py")

            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code_tasks[-1].result)
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


    def _build_file_plan_spec_string(self, file_spec) -> str:
        """Convert a FileSpec to a human-readable string for the reviewer."""
        parts = [f"File: {file_spec.name}"]
        parts.append(f"Purpose: {file_spec.purpose}")

        if file_spec.exports:
            parts.append("Required exports:")
            for exp in file_spec.exports:
                methods_str = ""
                if exp.methods:
                    method_names = list(exp.methods.keys())
                    methods_str = f" (methods: {', '.join(method_names)})"
                parts.append(f"  - {exp.name} ({exp.type}){methods_str}")

        if file_spec.requirements:
            parts.append("Requirements:")
            for req in file_spec.requirements:
                parts.append(f"  - {req}")

        if file_spec.imports_from:
            parts.append("Imports from:")
            for src, names in file_spec.imports_from.items():
                parts.append(f"  - {src}: {', '.join(names)}")

        return "\n".join(parts)

    def _build_dependency_code_for_review(self, file_spec, all_files_dict: dict) -> str:
        """Build dependency context for per-file review.

        Direct deps get full code (8K cap each), other files get signatures only.
        """
        parts = []
        for dep_name in file_spec.dependencies:
            dep_code = all_files_dict.get(dep_name, "")
            if dep_code:
                if len(dep_code) > 8000:
                    dep_code = dep_code[:8000] + "\n# ... (truncated)"
                parts.append(f"### {dep_name} ###\n{dep_code}")
        return "\n\n".join(parts)

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
        code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")), None)
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
    def _log_prompt(self, step: str, agent_name: str, payload: Dict, mode: str = ""):
        """Save full prompt payload to projects/<project>/prompt_logs/ as JSON."""
        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir:
            return

        log_dir = os.path.join(project_dir, "prompt_logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        # Sanitize step name: replace / with _ to avoid creating subdirectories
        safe_step = step.replace("/", "_")
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
        for key in ("job_spec", "plan_yaml", "plan_spec", "code", "draft_plan", "coder_feedback",
                     "environment_context", "revision_feedback", "original_code", "result"):
            if key in payload:
                log_entry[key] = payload[key]

        log_path = os.path.join(log_dir, filename)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)
            print(f"  [LOG] {filename}")
        except Exception as e:
            print(f"  [LOG] Warning: could not save prompt log: {e}")

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

            # 1. Sync files to container (recursive for subdirectories)
            src_local = os.path.join(project_dir, "src")
            tests_local = os.path.join(project_dir, "tests")

            # Create remote dirs
            self.sandbox._exec(["mkdir", "-p", "/workspace/src", "/workspace/tests"])

            # Copy src files (recursive — handles subdirectories like src/agents/)
            if os.path.exists(src_local):
                for root, dirs, files in os.walk(src_local):
                    rel_root = os.path.relpath(root, src_local)
                    remote_root = f"/workspace/src/{rel_root}" if rel_root != "." else "/workspace/src"
                    if rel_root != ".":
                        self.sandbox._exec(["mkdir", "-p", remote_root])
                    for f in files:
                        if f.endswith(".py"):
                            self.sandbox.copy_to_container(os.path.join(root, f), f"{remote_root}/{f}")

            # Copy test files
            if os.path.exists(tests_local):
                for f in os.listdir(tests_local):
                    if f.endswith(".py"):
                        self.sandbox.copy_to_container(os.path.join(tests_local, f), f"/workspace/tests/{f}")

            # 2. Install requirements.txt if present
            req_file = os.path.join(project_dir, "requirements.txt")
            if os.path.exists(req_file):
                self.sandbox.copy_to_container(req_file, "/workspace/requirements.txt")
                print("   📦 Installing project dependencies...")
                install_res = self.sandbox._exec(
                    ["pip", "install", "-q", "--no-cache-dir", "-r", "/workspace/requirements.txt"],
                    timeout=120
                )
                if install_res.returncode != 0:
                    print(f"   ⚠ Some deps failed to install (rc={install_res.returncode})")
                    if install_res.stderr:
                        # Show last few lines of error
                        err_lines = install_res.stderr.strip().split("\n")
                        for line in err_lines[-5:]:
                            print(f"      {line}")

            # 3. Run Pytest in Docker
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

        # Install requirements.txt if present
        req_file = os.path.join(project_dir, "requirements.txt")
        if os.path.exists(req_file):
            print("   📦 Installing project dependencies...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-r", req_file],
                    env=env, capture_output=True, text=True, timeout=120
                )
            except Exception as e:
                print(f"   ⚠ Dependency install failed: {e}")

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
                plan_yaml = (
                    self.state["context"].get("final_plan") or
                    self.state["context"].get("architecture_plan") or
                    self.state["context"].get("plan_yaml") or
                    ""
                )
                if not plan_yaml: raise Exception("No YAML plan found")
                job_scope = self.state["context"].get("job_spec") or self.state["context"].get("job_scope", "")
                task.result = execute_plan_task(self, task, plan_yaml, task.metadata.get("user_request",""), job_scope)
                self.state["context"]["inline_reviewer_done"] = True
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self._update_context(task)
                # Save files immediately so they're on disk even if later steps fail
                # Temporarily add task to completed_tasks so _save_project_outputs can find it
                self.completed_tasks.append(task)
                try:
                    self._save_project_outputs()
                except Exception as e:
                    print(f"  ⚠ Error saving after plan_execution: {e}")
                self.completed_tasks.remove(task)
                return task
            
            # --- COLLABORATIVE WORKFLOW TASK HANDLERS ---

            # DRAFT_PLAN: Architect produces draft plan from job_spec
            if task.task_type == "draft_plan":
                job_spec = self.state["context"].get("job_spec", "")
                if not job_spec:
                    job_spec = self.state["context"].get("job_scope", "")
                architect_config = self.executor._get_agent_config(AgentRole.ARCHITECT)

                payload = {
                    "mode": "draft",
                    "job_spec": job_spec,
                    "config": {
                        "model_url": architect_config.get("url", "http://localhost:1233/v1"),
                        "model_name": architect_config.get("model", "local-model"),
                        "api_type": architect_config.get("api_type", "openai"),
                        "temperature": architect_config.get("temperature", 0.6),
                        "max_tokens": architect_config.get("max_tokens", 35000),
                        "timeout": architect_config.get("timeout", 600)
                    }
                }

                self._log_prompt("DRAFT", "ARCHITECT", {"system_prompt": "DRAFT_PLAN_PROMPT", "user_message": job_spec, "job_spec": job_spec}, mode="draft")

                try:
                    output = self._run_external_agent("architect", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    plan_yaml = output.get("plan_yaml") or output.get("result", "")
                    self.state["context"]["draft_plan"] = plan_yaml
                    task.result = plan_yaml
                    task.status = TaskStatus.COMPLETED
                    print(f"  ✓ Draft plan produced ({len(plan_yaml)} chars)")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # PLAN_REVIEW: Coder reviews draft plan (no code)
            if task.task_type == "plan_review":
                draft_plan = self.state["context"].get("draft_plan", "")
                coder_config = self.executor._get_agent_config(AgentRole.CODER)

                payload = {
                    "mode": "plan_review",
                    "plan_yaml": draft_plan,
                    "config": {
                        "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                        "model_name": coder_config.get("model", "local-model"),
                        "api_type": coder_config.get("api_type", "openai"),
                        "temperature": coder_config.get("temperature", 0.3),
                        "max_tokens": coder_config.get("max_tokens", 35000),
                        "timeout": coder_config.get("timeout", 1200)
                    }
                }

                self._log_prompt("PLAN_REVIEW", "CODER", {"system_prompt": "PLAN_REVIEW_PROMPT", "user_message": draft_plan, "plan_yaml": draft_plan}, mode="plan_review")

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

            # FINALIZE_PLAN: Architect incorporates coder feedback
            if task.task_type == "finalize_plan":
                draft_plan = self.state["context"].get("draft_plan", "")
                plan_review = self.state["context"].get("plan_review", "")
                architect_config = self.executor._get_agent_config(AgentRole.ARCHITECT)

                payload = {
                    "mode": "finalize",
                    "draft_plan": draft_plan,
                    "coder_feedback": plan_review,
                    "config": {
                        "model_url": architect_config.get("url", "http://localhost:1233/v1"),
                        "model_name": architect_config.get("model", "local-model"),
                        "api_type": architect_config.get("api_type", "openai"),
                        "temperature": architect_config.get("temperature", 0.6),
                        "max_tokens": architect_config.get("max_tokens", 35000),
                        "timeout": architect_config.get("timeout", 600)
                    }
                }

                self._log_prompt("FINALIZE", "ARCHITECT", {"system_prompt": "FINALIZE_PROMPT", "user_message": plan_review, "draft_plan": draft_plan, "coder_feedback": plan_review}, mode="finalize")

                try:
                    output = self._run_external_agent("architect", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    final_plan = output.get("plan_yaml") or output.get("result", "")
                    self.state["context"]["final_plan"] = final_plan
                    self.state["context"]["plan_yaml"] = final_plan  # For backward compat
                    task.result = final_plan
                    task.status = TaskStatus.COMPLETED
                    self._set_project_type_from_plan(final_plan)
                    print(f"  ✓ Final plan produced ({len(final_plan)} chars)")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # TOOL_CHECK: Toolsmith checks Tool Forge for reusable tools before coding
            if task.task_type == "tool_check":
                user_request = self.state["context"].get("user_request", "")
                job_scope = self.state["context"].get("job_spec") or self.state["context"].get("job_scope", "")
                architecture = self.state["context"].get("final_plan") or self.state["context"].get("draft_plan", "")
                toolsmith_config = self.executor._get_agent_config(AgentRole.TOOLSMITH)

                payload = {
                    "task_description": job_scope or user_request,
                    "user_request": user_request,
                    "architecture_plan": architecture,
                    "config": {
                        "model_url": toolsmith_config.get("url", "http://192.168.40.100:1234/v1"),
                        "model_name": toolsmith_config.get("model", "qwen/qwen3-coder-30b"),
                        "api_type": toolsmith_config.get("api_type", "openai"),
                        "temperature": toolsmith_config.get("temperature", 0.3),
                        "max_tokens": toolsmith_config.get("max_tokens", 4000),
                        "timeout": toolsmith_config.get("timeout", 300)
                    }
                }

                self._log_prompt("TOOL_CHECK", "TOOLSMITH", payload, mode="tool_check")

                try:
                    output = self._run_external_agent("toolsmith", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    decision = output.get("decision", "skip")
                    tool_context = output.get("tool_context", "")
                    self.state["context"]["tool_context"] = tool_context
                    self.state["context"]["tool_decision"] = decision
                    task.result = f"Decision: {decision}\n{tool_context}"
                    task.status = TaskStatus.COMPLETED
                    print(f"  ✓ Tool check: {decision}" +
                          (f" ({len(output.get('reuse_tools', []))} reuse, {len(output.get('built_tools', []))} built)" if decision != "skip" else ""))
                except Exception as e:
                    # Tool check failure is non-fatal - skip and continue
                    print(f"  ⚠ Tool check failed: {e} — continuing without tools")
                    self.state["context"]["tool_context"] = ""
                    self.state["context"]["tool_decision"] = "skip"
                    task.result = "Tool check skipped due to error"
                    task.status = TaskStatus.COMPLETED

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # BUILD_FROM_PLAN: Coder builds all code (owns imports — no _fix_relative_imports)
            if task.task_type == "build_from_plan":
                final_plan = self.state["context"].get("final_plan", "")
                job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))
                coder_config = self.executor._get_agent_config(AgentRole.CODER)

                # Build environment context so coder knows how to reach LLMs
                env_context = self._build_environment_context()

                # Inject tool context from toolsmith if available
                tool_context = self.state["context"].get("tool_context", "")

                payload = {
                    "mode": "build",
                    "plan_yaml": final_plan,
                    "job_spec": job_spec,
                    "environment_context": env_context,
                    "tool_context": tool_context,
                    "config": {
                        "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                        "model_name": coder_config.get("model", "local-model"),
                        "api_type": coder_config.get("api_type", "openai"),
                        "temperature": coder_config.get("temperature", 0.2),
                        "max_tokens": coder_config.get("max_tokens", 35000),
                        "timeout": coder_config.get("timeout", 1200)
                    }
                }

                self._log_prompt("BUILD", "CODER", {"system_prompt": "BUILD_PROMPT", "user_message": f"PLAN:\n{final_plan}\n\nJOB_SPEC:\n{job_spec}\n\nENV:\n{env_context}\n\nTOOL_CONTEXT:\n{tool_context}", "plan_yaml": final_plan, "job_spec": job_spec, "environment_context": env_context, "tool_context": tool_context}, mode="build")

                try:
                    output = self._run_external_agent("coder", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    result_text = output.get("result", "")
                    result_text = self._clean_code_output(result_text)

                    # NOTE: We do NOT call _fix_relative_imports — the coder owns imports
                    self.state["context"]["latest_code"] = result_text
                    task.result = result_text
                    task.status = TaskStatus.COMPLETED
                    print(f"  ✓ Code built ({len(result_text)} chars)")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                task.completed_at = time.time()
                self._update_context(task)
                return task

            # COMPLIANCE_REVIEW: Reviewer checks code against plan (file-by-file)
            if task.task_type == "compliance_review":
                final_plan = self.state["context"].get("final_plan", "")
                latest_code = self.state["context"].get("latest_code", "")
                job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))
                reviewer_config = self.executor._get_agent_config(AgentRole.REVIEWER)

                reviewer_llm_config = {
                    "model_url": reviewer_config.get("url", "http://localhost:1233/v1"),
                    "model_name": reviewer_config.get("model", "local-model"),
                    "api_type": reviewer_config.get("api_type", "openai"),
                    "temperature": reviewer_config.get("temperature", 0.3),
                    "max_tokens": reviewer_config.get("max_tokens", 8000),
                    "timeout": reviewer_config.get("timeout", 600)
                }

                # Try file-by-file compliance review
                use_file_by_file = False
                plan_parsed = None
                all_files_dict = self._parse_multi_file_output(latest_code) if latest_code else {}

                if PLAN_EXECUTOR_AVAILABLE and final_plan and all_files_dict:
                    try:
                        plan_exec = PlanExecutor(executor=self.executor, config=self.config)
                        plan_parsed = plan_exec.parse_plan(final_plan)
                        use_file_by_file = True
                    except Exception as e:
                        print(f"  ⚠ Could not parse plan for file-by-file review: {e}")

                try:
                    if use_file_by_file and plan_parsed:
                        # Check if inline reviewer already ran during build
                        if self.state["context"].get("inline_reviewer_done"):
                            print(f"  ✓ Inline reviewer already ran during build — skipping batch review")
                            per_file_results = {}
                            failed_files = []
                            report_parts = ["COMPLIANCE REVIEW (inline during build):\n"]
                            for file_spec in plan_parsed.files:
                                fname = file_spec.name
                                if fname in all_files_dict:
                                    per_file_results[fname] = f"FILE: {fname}\nSTATUS: PASS\n(Passed inline reviewer during build)"
                                    report_parts.append(per_file_results[fname])
                                else:
                                    per_file_results[fname] = "MISSING"
                                    failed_files.append(fname)
                                    report_parts.append(f"FILE: {fname}\nSTATUS: FAIL\nFile missing.\n")

                            if failed_files:
                                report_parts.append(f"\nOVERALL: STATUS: NEEDS_REVISION")
                                report_parts.append(f"Failed files: {', '.join(failed_files)}")
                            else:
                                report_parts.append(f"\nOVERALL: STATUS: APPROVED")

                            result_text = "\n".join(report_parts)
                            task.result = result_text
                            task.metadata["per_file_results"] = per_file_results
                            task.metadata["failed_files"] = failed_files
                            if failed_files:
                                task.metadata["needs_revision"] = True
                                task.metadata["final_plan"] = final_plan
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = time.time()
                            self._update_context(task)
                            return task

                        # No inline review — do batch file-by-file compliance review
                        print(f"  📋 File-by-file compliance review ({len(plan_parsed.files)} files)")
                        per_file_results = {}
                        failed_files = []
                        report_parts = ["COMPLIANCE REVIEW (file-by-file):\n"]

                        for file_spec in plan_parsed.files:
                            fname = file_spec.name
                            file_code = all_files_dict.get(fname, "")

                            if not file_code:
                                # File is in plan but missing from code
                                per_file_results[fname] = "MISSING — file not found in generated code"
                                failed_files.append(fname)
                                report_parts.append(f"FILE: {fname}\nSTATUS: FAIL\nFile missing from generated code.\n")
                                print(f"    ✗ {fname}: MISSING")
                                continue

                            plan_spec_str = self._build_file_plan_spec_string(file_spec)
                            dep_code_str = self._build_dependency_code_for_review(file_spec, all_files_dict)

                            payload = {
                                "mode": "compliance_file",
                                "file_name": fname,
                                "file_code": file_code,
                                "plan_spec": plan_spec_str,
                                "dependency_code": dep_code_str,
                                "job_spec": job_spec,
                                "config": reviewer_llm_config
                            }

                            self._log_prompt(f"COMPLIANCE_FILE_{fname}", "REVIEWER", {
                                "system_prompt": "FILE_COMPLIANCE_PROMPT",
                                "user_message": f"file={fname} ({len(file_code)} chars)",
                                "plan_spec": plan_spec_str,
                                "code": file_code,
                                "job_spec": job_spec
                            }, mode="compliance_file")

                            output = self._run_external_agent("reviewer", payload)

                            if output.get("status") == "error":
                                per_file_results[fname] = f"ERROR: {output.get('error')}"
                                print(f"    ✗ {fname}: ERROR — {output.get('error')}")
                                continue

                            file_review = output.get("result", "")
                            per_file_results[fname] = file_review

                            self._log_prompt(f"COMPLIANCE_FILE_{fname}_RESULT", "REVIEWER", {
                                "system_prompt": "FILE_COMPLIANCE_PROMPT",
                                "user_message": "RESULT",
                                "result": file_review
                            }, mode="compliance_file")

                            # Parse STATUS: PASS|FAIL from result
                            file_status_upper = file_review.upper()
                            if "STATUS: FAIL" in file_status_upper or "STATUS:FAIL" in file_status_upper:
                                failed_files.append(fname)
                                print(f"    ✗ {fname}: FAIL")
                            else:
                                print(f"    ✓ {fname}: PASS")

                            report_parts.append(file_review)
                            report_parts.append("")

                        # Check for files in code but not in plan (extra files — just note them)
                        plan_file_names = {f.name for f in plan_parsed.files}
                        for fname in all_files_dict:
                            if fname not in plan_file_names and not fname.endswith("__init__.py"):
                                report_parts.append(f"FILE: {fname}\nNOTE: Extra file not in plan (not reviewed)\n")

                        # Assemble overall report
                        if failed_files:
                            report_parts.append(f"\nOVERALL: STATUS: NEEDS_REVISION")
                            report_parts.append(f"Failed files: {', '.join(failed_files)}")
                        else:
                            report_parts.append(f"\nOVERALL: STATUS: APPROVED")

                        result_text = "\n".join(report_parts)
                        task.result = result_text

                        # Store per-file results and failed files in metadata for revision
                        task.metadata["per_file_results"] = per_file_results
                        task.metadata["failed_files"] = failed_files

                        if failed_files:
                            task.metadata["needs_revision"] = True
                            task.metadata["final_plan"] = final_plan
                            print(f"  ⚠ Compliance review: NEEDS_REVISION ({len(failed_files)} files)")
                        else:
                            print(f"  ✓ Compliance review: APPROVED (all {len(plan_parsed.files)} files pass)")

                    else:
                        # FALLBACK: single-call compliance review (backward compat)
                        payload = {
                            "mode": "compliance",
                            "plan_yaml": final_plan,
                            "code": latest_code,
                            "job_spec": job_spec,
                            "config": reviewer_llm_config
                        }
                        # Override max_tokens for full-code review
                        payload["config"]["max_tokens"] = reviewer_config.get("max_tokens", 35000)

                        self._log_prompt("COMPLIANCE", "REVIEWER", {
                            "system_prompt": "COMPLIANCE_PROMPT",
                            "user_message": f"PLAN + CODE + JOB_SPEC ({len(latest_code)} chars code)",
                            "plan_yaml": final_plan, "code": latest_code, "job_spec": job_spec
                        }, mode="compliance")

                        output = self._run_external_agent("reviewer", payload)
                        if output.get("status") == "error":
                            raise Exception(output.get("error"))

                        result_text = output.get("result", "")
                        task.result = result_text

                        self._log_prompt("COMPLIANCE_RESULT", "REVIEWER", {
                            "system_prompt": "COMPLIANCE_PROMPT",
                            "user_message": "RESULT", "result": result_text
                        }, mode="compliance")

                        if "NEEDS_REVISION" in result_text.upper() or "STATUS: NEEDS_REVISION" in result_text.upper():
                            task.metadata["needs_revision"] = True
                            task.metadata["final_plan"] = final_plan
                            print(f"  ⚠ Compliance review: NEEDS_REVISION")
                        else:
                            print(f"  ✓ Compliance review: APPROVED")

                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

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

                # RUN SMOKE TESTS LOCALLY (informational — not a hard gate)
                smoke_tests = self._run_smoke_tests()
                task.metadata["smoke_tests"] = smoke_tests

                # Log smoke results for visibility
                for name, info in (smoke_tests or {}).items():
                    rc = info.get("returncode", None)
                    status_icon = "PASS" if rc == 0 else "FAIL"
                    print(f"   Smoke: {name} → {status_icon} (rc={rc})")

                # Always proceed to LLM verification — smoke results are passed as context
                system_prompt, user_message = self._get_verifier_prompt(task)
                ver_config = self.executor._get_agent_config(AgentRole.VERIFIER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": ver_config.get("url", "http://localhost:1233/v1"),
                        "model_name": ver_config.get("model", "local-model"),
                        "api_type": ver_config.get("api_type", "openai"),
                        "temperature": ver_config.get("temperature", 0.3),
                        "max_tokens": ver_config.get("max_tokens", 35000),
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
                        print("VERIFICATION FAILED")
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
                        "temperature": clarifier_config.get("temperature", 0.7),
                        "max_tokens": clarifier_config.get("max_tokens", 35000),
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
                            "temperature": architect_config.get("temperature", 0.6),
                            "max_tokens": architect_config.get("max_tokens", 35000),
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

                # Inject tool context from toolsmith if available
                tool_context = self.state["context"].get("tool_context", "")
                if tool_context:
                    user_message = user_message + f"\n\n{tool_context}"

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "tool_context": tool_context,
                    "config": {
                        "model_url": coder_config.get("url", "http://localhost:1233/v1"),
                        "model_name": coder_config.get("model", "local-model"),
                        "api_type": coder_config.get("api_type", "openai"),
                        "temperature": coder_config.get("temperature", 0.2),
                        "max_tokens": coder_config.get("max_tokens", 35000),
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
                        "temperature": reviewer_config.get("temperature", 0.8),
                        "max_tokens": reviewer_config.get("max_tokens", 35000),
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
                        "temperature": tester_config.get("temperature", 0.7),
                        "max_tokens": tester_config.get("max_tokens", 35000),
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
                        "temperature": doc_config.get("temperature", 0.7),
                        "max_tokens": doc_config.get("max_tokens", 35000),
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
                        "temperature": sec_config.get("temperature", 0.8),
                        "max_tokens": sec_config.get("max_tokens", 35000),
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
                        "temperature": opt_config.get("temperature", 0.6),
                        "max_tokens": opt_config.get("max_tokens", 35000),
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
            # NOTE: Smoke tests run locally but failures are informational, not hard gates
            if task.task_type == "verification" and task.assigned_role == AgentRole.VERIFIER:
                # 1. Save outputs & Run Smoke Tests (informational)
                try:
                    self._save_project_outputs()
                except Exception as e:
                    print(f"   ! Warning: failed to save before verification: {e}")

                smoke_tests = self._run_smoke_tests()
                task.metadata["smoke_tests"] = smoke_tests

                # Log smoke results for visibility
                for name, info in (smoke_tests or {}).items():
                    rc = info.get("returncode", None)
                    status_icon = "PASS" if rc == 0 else "FAIL"
                    print(f"   Smoke: {name} → {status_icon} (rc={rc})")

                # 2. Always send to LLM for holistic evaluation (smoke results included in prompt)
                system_prompt, user_message = self._get_verifier_prompt(task)
                ver_config = self.executor._get_agent_config(AgentRole.VERIFIER)

                payload = {
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "config": {
                        "model_url": ver_config.get("url", "http://localhost:1233/v1"),
                        "model_name": ver_config.get("model", "local-model"),
                        "api_type": ver_config.get("api_type", "openai"),
                        "temperature": ver_config.get("temperature", 0.3),
                        "max_tokens": ver_config.get("max_tokens", 35000),
                        "timeout": ver_config.get("timeout", 600)
                    }
                }

                try:
                    output = self._run_external_agent("verifier", payload)
                    if output.get("status") == "error":
                        raise Exception(output.get("error"))

                    result = output.get("result", "")
                    task.result = result

                    resp_upper = (result or "").upper()
                    explicit_fail = "VERIFICATION: FAIL" in resp_upper
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
    
    def _build_environment_context(self) -> str:
        """Build a description of the runtime environment (LLM endpoints, etc.) for the coder."""
        lines = []
        lines.append("RUNTIME ENVIRONMENT (use this info to write real LLM integration):")
        lines.append("-" * 60)

        model_config = self.config.get("model_config", {})
        mode = model_config.get("mode", "single")

        if mode == "single":
            sm = model_config.get("single_model", {})
            lines.append(f"LLM Endpoint: {sm.get('url', 'http://localhost:11434')}")
            lines.append(f"Model: {sm.get('model', 'local-model')}")
            lines.append(f"API Type: {sm.get('api_type', 'ollama')}")
        else:
            # Multi-model: pick the coder's endpoint as the representative one
            mm = model_config.get("multi_model", {})
            coder_m = mm.get("coder", {})
            lines.append(f"LLM Endpoint: {coder_m.get('url', 'http://localhost:1234/v1')}")
            lines.append(f"Model: {coder_m.get('model', 'local-model')}")
            lines.append(f"API Type: {coder_m.get('api_type', 'openai')}")

        lines.append("")
        lines.append("To call the LLM from generated code, use:")
        lines.append("  import requests")
        lines.append("  response = requests.post(f\"{endpoint}/chat/completions\", json={")
        lines.append("      \"model\": model_name,")
        lines.append("      \"messages\": [{\"role\": \"user\", \"content\": prompt}],")
        lines.append("      \"temperature\": 0.7")
        lines.append("  })")
        lines.append("  result = response.json()[\"choices\"][0][\"message\"][\"content\"]")
        lines.append("")
        lines.append("This is a LOCAL LLM — no API key needed. Use it for any agent that")
        lines.append("needs to analyze text, generate hypotheses, summarize papers, etc.")
        lines.append("Do NOT mock or simulate LLM calls — actually call the endpoint above.")
        lines.append("-" * 60)

        return "\n".join(lines)

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
                "temperature": clarifier_config.get("temperature", 0.7),
                "max_tokens": clarifier_config.get("max_tokens", 35000),
                "timeout": clarifier_config.get("timeout", 300)
            }
        }

        # Pass output_format for structured job spec
        if output_format:
            synthesis_payload["output_format"] = output_format

        # Log the prompt
        self._log_prompt("SYNTHESIZE", "CLARIFIER", {
            "system_prompt": "SYNTHESIZE_STRUCTURED_PROMPT" if output_format == "structured" else "SYNTHESIZE_PROMPT",
            "user_message": f"Request: {user_request}\n\nQ&A: {result}\n\nAnswers: {answers_text}"
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
    
    def _handle_file_by_file_revision(self, review_task: Task) -> bool:
        """
        Handle revision using PlanExecutor for each failed file.

        Uses PlanExecutor._handle_revision() for anti-stub prompt, AST detection,
        export validation, retries, and fallback coder — matching the build phase.

        Returns True if revision was performed successfully.
        """
        if not PLAN_EXECUTOR_AVAILABLE:
            return False

        failed_files = review_task.metadata.get("failed_files", [])
        per_file_results = review_task.metadata.get("per_file_results", {})
        final_plan = review_task.metadata.get("final_plan") or self.state["context"].get("final_plan", "")
        latest_code = self.state["context"].get("latest_code", "")
        user_request = self.state["context"].get("user_request", "")
        job_spec = self.state["context"].get("job_spec", self.state["context"].get("job_scope", ""))

        if not failed_files or not final_plan:
            return False

        # Parse current code and plan
        all_files_dict = self._parse_multi_file_output(latest_code) if latest_code else {}
        if not all_files_dict:
            return False

        try:
            project_dir = self.state["project_info"].get("project_dir", "")
            plan_exec = PlanExecutor(
                executor=self.executor,
                config=self.config,
                project_dir=project_dir
            )
            plan_exec.plan = plan_exec.parse_plan(final_plan)
            plan_exec.job_scope = job_spec or user_request
        except Exception as e:
            print(f"  ⚠ Could not initialize PlanExecutor for revision: {e}")
            return False

        # Pre-populate completed_files with all current code
        for fname, fcode in all_files_dict.items():
            file_spec = plan_exec._get_file_spec(fname)
            actual_exports = plan_exec._extract_exports(fcode)
            plan_exec.completed_files[fname] = FileResult(
                name=fname,
                content=fcode,
                actual_exports=actual_exports,
                status=FileStatus.COMPLETED
            )

        print(f"\n🔄 File-by-file revision ({len(failed_files)} files to revise)")

        revised_any = False
        for fname in failed_files:
            file_spec = plan_exec._get_file_spec(fname)
            reviewer_feedback = per_file_results.get(fname, "")

            if not file_spec:
                print(f"    ⚠ {fname}: not found in plan, skipping")
                continue

            self._log_prompt(f"COMPLIANCE_REVISION_{fname}", "CODER", {
                "system_prompt": "(PlanExecutor revision)",
                "user_message": f"Revising {fname}",
                "revision_feedback": reviewer_feedback[:3000]
            }, mode="compliance_revision")

            context = plan_exec._build_context_for_file(file_spec)

            existing_code = all_files_dict.get(fname, "")

            if not existing_code:
                # MISSING file — generate from scratch
                print(f"    ▶ {fname}: generating (was missing)")
                try:
                    result = plan_exec._generate_file(file_spec, user_request, context)
                    if result.status == FileStatus.PLAN_MISMATCH:
                        result = plan_exec._handle_revision(file_spec, result, context, user_request)
                except Exception as e:
                    print(f"    ✗ {fname}: generation failed — {e}")
                    continue
            else:
                # FAILED file — revise via PlanExecutor._handle_revision()
                print(f"    ▶ {fname}: revising")
                actual_exports = plan_exec._extract_exports(existing_code)

                # Build a FileResult with the reviewer's feedback as compliance issues
                issues = []
                if reviewer_feedback:
                    # Extract ISSUES REQUIRING REVISION section if present
                    issues_match = re.search(
                        r'ISSUES REQUIRING REVISION[:\s]*\n(.*?)(?:\n\n|\nFILE:|\nOVERALL:|\Z)',
                        reviewer_feedback, re.DOTALL | re.IGNORECASE
                    )
                    if issues_match:
                        for line in issues_match.group(1).strip().split('\n'):
                            line = line.strip().lstrip('-').lstrip('0123456789.').strip()
                            if line and line.lower() != "none":
                                issues.append(line)
                    # Also extract STUB AUDIT entries
                    stub_match = re.search(
                        r'STUB AUDIT[:\s]*\n(.*?)(?:\n\n|\nEXPORT|\nREQUIREMENTS|\nISSUES|\Z)',
                        reviewer_feedback, re.DOTALL | re.IGNORECASE
                    )
                    if stub_match:
                        for line in stub_match.group(1).strip().split('\n'):
                            line = line.strip().lstrip('-').strip()
                            if line and "no stub" not in line.lower() and "none" not in line.lower():
                                issues.append(f"STUB: {line}")

                if not issues:
                    issues = [reviewer_feedback[:2000]]

                current_result = FileResult(
                    name=fname,
                    content=existing_code,
                    actual_exports=actual_exports,
                    status=FileStatus.PLAN_MISMATCH,
                    plan_compliance={"passed": False, "issues": issues, "warnings": []}
                )

                try:
                    result = plan_exec._handle_revision(file_spec, current_result, context, user_request)
                except Exception as e:
                    print(f"    ✗ {fname}: revision failed — {e}")
                    continue

            self._log_prompt(f"COMPLIANCE_REVISION_{fname}_RESULT", "CODER", {
                "system_prompt": "(PlanExecutor revision result)",
                "user_message": "RESULT",
                "result": result.content[:5000] if result.content else "(empty)"
            }, mode="compliance_revision")

            if result.status == FileStatus.COMPLETED and result.content:
                plan_exec.completed_files[fname] = result
                all_files_dict[fname] = result.content
                revised_any = True
                print(f"    ✓ {fname}: revised successfully")
            else:
                print(f"    ⚠ {fname}: revision did not pass (keeping original)")

        if not revised_any:
            print(f"  ⚠ No files were successfully revised")
            return False

        # Merge revised files back into latest_code
        combined_parts = []
        for fname in plan_exec.plan.execution_order:
            if fname in all_files_dict:
                combined_parts.append(f"### FILE: {fname} ###\n{all_files_dict[fname]}")
        # Include any extra files (e.g. __init__.py)
        for fname, fcode in all_files_dict.items():
            if fname not in plan_exec.plan.execution_order:
                combined_parts.append(f"### FILE: {fname} ###\n{fcode}")
        revised_code = "\n\n".join(combined_parts)

        # Create synthetic revision task for _save_project_outputs
        revision_task = Task(
            task_id="T_compliance_revision",
            task_type="revision",
            description="File-by-file compliance revision",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.COMPLETED,
            priority=10,
            metadata={"is_revision": True}
        )
        revision_task.result = revised_code
        self.completed_tasks.append(revision_task)

        # Update context
        self.state["context"]["latest_code"] = revised_code
        self._update_context(revision_task)

        # Save to disk immediately
        try:
            self._save_project_outputs()
        except Exception as e:
            print(f"  ⚠ Error saving revised outputs: {e}")

        print(f"  ✓ File-by-file revision complete")
        return True

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

        # Find the coding task (also check build_from_plan for collaborative workflow)
        code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")), None)
        if not code_task:
            return False

        # Check revision count
        if code_task.revision_count >= code_task.max_revisions:
            print(f"\n⚠ Max revisions ({code_task.max_revisions}) reached, proceeding anyway")
            return False

        # Try file-by-file revision if compliance review has per_file_results
        for rt in review_tasks:
            if rt.task_type == "compliance_review" and rt.metadata.get("per_file_results"):
                try:
                    if self._handle_file_by_file_revision(rt):
                        return True
                except Exception as e:
                    print(f"  ⚠ File-by-file revision failed: {e}, falling back to generic revision")

        print(f"\n🔄 Revision cycle {code_task.revision_count + 1}/{code_task.max_revisions}")

        # Collect review feedback
        feedback = []
        for rt in review_tasks:
            if rt.result and rt.metadata.get("needs_revision"):
                feedback.append(f"[{rt.task_id}]: {rt.result}")

        # Build revision metadata — only include files needing revision to avoid context blowup
        full_feedback = "\n\n".join(feedback)
        all_files = self._parse_multi_file_output(code_task.result)
        if all_files and len(code_task.result) > 30000:
            # Extract which files need revision from the feedback
            revision_files = {}
            reference_signatures = []
            feedback_lower = full_feedback.lower()
            for fname, fcontent in all_files.items():
                if fname.lower() in feedback_lower or fname.replace("/", ".").replace(".py","") in feedback_lower:
                    revision_files[fname] = fcontent
                else:
                    # Include only class/function signatures for reference
                    sig_lines = [f"# --- {fname} (reference only) ---"]
                    for line in fcontent.split('\n'):
                        stripped = line.strip()
                        if stripped.startswith(('class ', 'def ', 'import ', 'from ')):
                            sig_lines.append(line)
                    reference_signatures.append('\n'.join(sig_lines[:30]))

            # Assemble reduced original_code
            parts = []
            for fname, fcontent in revision_files.items():
                parts.append(f"### FILE: {fname} ###\n{fcontent}")
            if reference_signatures:
                parts.append("\n# === OTHER FILES (signatures only — do NOT rewrite these) ===")
                parts.append('\n\n'.join(reference_signatures))
            reduced_code = '\n\n'.join(parts)
            print(f"  Revision: sending {len(revision_files)} files to revise ({len(reduced_code)//1024}KB), {len(all_files) - len(revision_files)} as reference sigs")
        else:
            reduced_code = code_task.result

        revision_metadata = {
            "revision_feedback": full_feedback,
            "original_code": reduced_code,
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

        # Log the revision prompt
        self._log_prompt(f"REVISION_{revision_task.revision_count}", "CODER", {
            "system_prompt": "(revision system prompt)",
            "user_message": f"Revision #{revision_task.revision_count}",
            "revision_feedback": revision_metadata.get("revision_feedback", ""),
            "original_code": revision_metadata.get("original_code", "")[:5000] + "..." if len(revision_metadata.get("original_code", "")) > 5000 else revision_metadata.get("original_code", ""),
            "plan_yaml": revision_metadata.get("final_plan", "")
        }, mode="revision")

        # Execute revision
        self.execute_task(revision_task)

        if revision_task.status == TaskStatus.COMPLETED:
            # Log the revision result
            self._log_prompt(f"REVISION_{revision_task.revision_count}_RESULT", "CODER", {
                "system_prompt": "(revision result)",
                "user_message": "RESULT",
                "result": revision_task.result
            }, mode="revision")

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
        elif workflow_type == "collaborative":
            self._create_collaborative_workflow(user_request)
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
        
        # Save all outputs first (before anything else that might fail)
        try:
            self._save_project_outputs()
            init_file = os.path.join(project_dir, "src", "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write("")
        except Exception as e:
            print(f"  ⚠ Error saving project outputs: {e}")

        # Hand off to Tier 2 finisher pipeline
        try:
            self._handoff_to_tier2()
        except Exception as e:
            print(f"  ⚠ Tier 2 handoff failed (non-fatal): {e}")

        # Generate final report
        try:
            self._generate_workflow_report()
        except Exception as e:
            print(f"  ⚠ Error generating report: {e}")

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

    def _handoff_to_tier2(self):
        """
        Hand off the completed project to the Tier 2 finisher pipeline.

        Scans the project's src/ directory, builds a manifest.json, and copies
        everything into the tier2 handoff directory.  The tier2 watcher (or a
        manual orchestrator run) picks it up from there.
        """
        import ast as _ast
        import shutil
        from pathlib import Path
        from datetime import timezone

        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir or not os.path.isdir(project_dir):
            return

        # Use the full directory name (e.g. "135_swarm_project_eve_v1") so tier2
        # output folders match the swarm_v5 naming convention.
        project_name = os.path.basename(os.path.normpath(project_dir))

        # ---- configurable paths ------------------------------------------------
        tier2_finisher_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "..", "tier2_finisher")
        tier2_finisher_root = os.path.normpath(tier2_finisher_root)

        # Fall back to hard-coded sibling path if the relative lookup misses
        if not os.path.isdir(tier2_finisher_root):
            tier2_finisher_root = os.path.expanduser("~/projects/tier2_finisher")
        if not os.path.isdir(tier2_finisher_root):
            print("  ⚠ tier2_finisher not found, skipping handoff")
            return

        handoff_dir = os.path.join(tier2_finisher_root, "handoff")
        # -----------------------------------------------------------------------

        # The swarm writes generated code into src/, so that's what tier2 processes.
        src_dir = os.path.join(project_dir, "src")
        if not os.path.isdir(src_dir):
            src_dir = project_dir  # fallback: project root

        # ---- scan the source tree for modules / classes / entry points ----------
        expected_modules = []
        expected_classes = []
        expected_entry_points = []

        src_path = Path(src_dir)
        for py_file in sorted(src_path.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel = str(py_file.relative_to(src_path))
            expected_modules.append(rel)
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = _ast.parse(source)
            except Exception:
                continue
            for node in _ast.walk(tree):
                if isinstance(node, _ast.ClassDef):
                    expected_classes.append(node.name)
            if py_file.name == "main.py" or ("if __name__" in source and "__main__" in source):
                expected_entry_points.append(rel)

        if not expected_entry_points:
            expected_entry_points = ["main.py"]

        # ---- read original spec from PROJECT_INFO.txt ---------------------------
        spec_text = ""
        info_path = os.path.join(project_dir, "PROJECT_INFO.txt")
        if os.path.isfile(info_path):
            try:
                spec_text = Path(info_path).read_text(encoding="utf-8")
            except Exception:
                pass

        # If no PROJECT_INFO.txt, fall back to user_request from state
        if not spec_text:
            spec_text = self.state.get("context", {}).get("user_request", "")

        # ---- build manifest ------------------------------------------------------
        manifest = {
            "project_name": project_name,
            "tier1_timestamp": datetime.now(timezone.utc).isoformat(),
            "tier1_version": "swarm_v5",
            "entry_point": expected_entry_points[0] if expected_entry_points else "main.py",
            "language": "python",
            "framework": None,
            "original_spec": spec_text,
            "expected_modules": expected_modules,
            "expected_classes": list(dict.fromkeys(expected_classes)),
            "expected_entry_points": expected_entry_points,
            "max_repair_cycles": 3,
            "eve_callback_url": None,
        }

        # ---- copy to handoff dir -------------------------------------------------
        dest = os.path.join(handoff_dir, project_name)
        output_dest = os.path.join(dest, "output")

        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src_dir, output_dest)

        manifest_path = os.path.join(dest, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Tier 2 handoff: {dest}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Modules:  {len(expected_modules)}  Classes: {len(manifest['expected_classes'])}  Entries: {expected_entry_points}")

        # ---- run tier2 orchestrator directly ------------------------------------
        # Add tier2_finisher to sys.path so we can import the orchestrator
        if tier2_finisher_root not in sys.path:
            sys.path.insert(0, tier2_finisher_root)

        try:
            from tier2_orchestrator import Tier2Orchestrator

            # Load tier2 config from YAML if available, otherwise use defaults
            tier2_config_path = os.path.join(tier2_finisher_root, "tier2_config.yaml")
            tier2_config = {}
            if os.path.isfile(tier2_config_path):
                import yaml as _yaml
                with open(tier2_config_path, "r") as f:
                    tier2_config = _yaml.safe_load(f) or {}

            # Override dirs to use local paths
            tier2_config["workspace_dir"] = os.path.join(tier2_finisher_root, "workspace")
            tier2_config["completed_dir"] = os.path.join(tier2_finisher_root, "completed")
            tier2_config["failed_dir"] = os.path.join(tier2_finisher_root, "failed")
            tier2_config["projects_dir"] = os.path.join(tier2_finisher_root, "projects")

            print(f"\n{'='*60}")
            print("TIER 2 FINISHER — PROCESSING")
            print(f"{'='*60}\n")

            orchestrator = Tier2Orchestrator(tier2_config)
            orchestrator.process(output_dest, manifest)

            # Show result — look in projects/{name}_tier2/ first
            tier2_project_name = f"{project_name}_tier2"
            projects_summary = os.path.join(
                tier2_config["projects_dir"], tier2_project_name, "tier2_summary.json"
            )
            if os.path.isfile(projects_summary):
                with open(projects_summary, "r") as f:
                    summary = json.load(f)
                output_path = os.path.join(tier2_config["projects_dir"], tier2_project_name)
                print(f"\n{'='*60}")
                print("TIER 2 RESULT")
                print(f"{'='*60}")
                print(f"  Verdict:        {summary.get('audit_verdict', 'UNKNOWN')}")
                print(f"  Health score:   {summary.get('final_health_score', 'N/A')}")
                print(f"  Repair cycles:  {summary.get('total_repair_cycles', 0)}")
                print(f"  Test status:    {summary.get('final_test_summary', {}).get('overall_status', 'N/A')}")
                print(f"  Output:         {output_path}")
            else:
                # Check failed dir
                failed_path = os.path.join(tier2_config["failed_dir"], project_name)
                if os.path.isdir(failed_path):
                    print(f"\n  ⚠ Tier 2 moved project to failed: {failed_path}")

        except ImportError as e:
            print(f"  ⚠ Could not import tier2_orchestrator: {e}")
            print(f"    Files are staged at {dest} for manual processing.")
        except Exception as e:
            print(f"  ⚠ Tier 2 processing failed: {e}")
            print(f"    Files are staged at {dest} for manual processing.")

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
        
        # Task 3: Tool Check (search Tool Forge before coding)
        self.add_task(Task(
            task_id="T003_tool_check",
            task_type="tool_check",
            description="Check Tool Forge for reusable tools",
            assigned_role=AgentRole.TOOLSMITH,
            status=TaskStatus.PENDING,
            priority=8,
            dependencies=["T002_architect"],
            metadata={"user_request": user_request}
        ))

        # Task 4: Coding (with tool context from toolsmith)
        self.add_task(Task(
            task_id="T004_code",
            task_type="coding",
            description="Implement the code",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_tool_check"],
            metadata={"user_request": user_request}
        ))

        # Tasks 5-7: Multiple reviewers in parallel
        for i in range(1, 4):
            self.add_task(Task(
                task_id=f"T00{4+i}_review{i}",
                task_type="review",
                description=f"Code review #{i}",
                assigned_role=AgentRole.REVIEWER,
                status=TaskStatus.PENDING,
                priority=6,
                dependencies=["T004_code"],
                metadata={"reviewer_number": i, "user_request": user_request}
            ))

        # Task 8: Documentation
        self.add_task(Task(
            task_id="T008_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=5,
            dependencies=["T005_review1", "T006_review2", "T007_review3"],
            metadata={"user_request": user_request}
        ))

        # Task 9: Verification
        self.add_task(Task(
            task_id="T009_verify",
            task_type="verification",
            description="Verify docs match code",
            assigned_role=AgentRole.VERIFIER,
            status=TaskStatus.PENDING,
            priority=4,
            dependencies=["T008_document"]
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
    
    def _create_collaborative_workflow(self, user_request: str):
        """Create collaborative workflow: clarify → draft plan → coder reviews → final plan → build → compliance"""

        # T001: Clarification (structured output)
        self.add_task(Task(
            task_id="T001_clarify",
            task_type="clarification",
            description="Clarify requirements and produce structured job spec",
            assigned_role=AgentRole.CLARIFIER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={"user_request": user_request, "output_format": "structured"}
        ))

        # T002: Draft Plan
        self.add_task(Task(
            task_id="T002_draft_plan",
            task_type="draft_plan",
            description="Produce draft architecture plan from job spec",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=9,
            dependencies=["T001_clarify"],
            metadata={"user_request": user_request}
        ))

        # T003: Plan Review (Coder reviews the draft)
        self.add_task(Task(
            task_id="T003_plan_review",
            task_type="plan_review",
            description="Coder reviews draft plan before finalization",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=8,
            dependencies=["T002_draft_plan"],
            metadata={"user_request": user_request}
        ))

        # T004: Final Plan (Architect incorporates coder feedback)
        self.add_task(Task(
            task_id="T004_final_plan",
            task_type="finalize_plan",
            description="Architect finalizes plan with coder feedback",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_plan_review"],
            metadata={"user_request": user_request}
        ))

        # T005: Tool Check (Toolsmith checks for reusable tools before coding)
        self.add_task(Task(
            task_id="T005_tool_check",
            task_type="tool_check",
            description="Check Tool Forge for reusable tools",
            assigned_role=AgentRole.TOOLSMITH,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T004_final_plan"],
            metadata={"user_request": user_request}
        ))

        # T006: Build (Coder builds all code from final plan + tool context)
        self.add_task(Task(
            task_id="T006_build",
            task_type="build_from_plan",
            description="Build all code from finalized plan",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=5,
            dependencies=["T005_tool_check"],
            metadata={"user_request": user_request}
        ))

        # T007: Compliance Review
        self.add_task(Task(
            task_id="T007_compliance",
            task_type="compliance_review",
            description="Verify code implements the plan correctly",
            assigned_role=AgentRole.REVIEWER,
            status=TaskStatus.PENDING,
            priority=4,
            dependencies=["T006_build"],
            metadata={"user_request": user_request}
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
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")), None)
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
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")), None)
            if code_task and code_task.result:
                message_parts.append(f"\nCODE TO ANALYZE FOR SECURITY:\n{code_task.result}\n")
        
        # Add review focus if specified
        if "review_focus" in task.metadata:
            message_parts.append(f"\nFOCUS YOUR REVIEW ON: {task.metadata['review_focus']}")
        
# Add reviewer number if specified
        if "reviewer_number" in task.metadata:
            message_parts.append(f"\nYou are reviewer #{task.metadata['reviewer_number']}")
        
        # --- FIX FOR TESTER: Provide actual source code, exports, plan, and job spec ---
        if task.task_type == "test_generation":
            # Include job spec if available
            job_spec = self.state["context"].get("job_spec") or self.state["context"].get("job_scope", "")
            if job_spec:
                message_parts.append("\n" + "=" * 70)
                message_parts.append("PROJECT REQUIREMENTS (job spec)")
                message_parts.append("=" * 70)
                message_parts.append(job_spec[:4000])

            # Include plan YAML if available (for file structure reference)
            plan_yaml = (
                self.state["context"].get("final_plan") or
                self.state["context"].get("architecture_plan") or
                ""
            )
            if plan_yaml:
                message_parts.append("\n" + "=" * 70)
                message_parts.append("ARCHITECTURE PLAN (for file structure reference)")
                message_parts.append("=" * 70)
                message_parts.append(plan_yaml[:4000])

            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")), None)
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

                        # Map file path -> python import module
                        module_path = fname.replace("\\", "/").replace(".py", "").lstrip("./")
                        if module_path.startswith("src/"):
                            module_path = module_path[len("src/"):]
                        module_path = module_path.replace("/", ".")
                        import_mod = f"src.{module_path}"

                        message_parts.append(f"\n{fname}:")
                        message_parts.append(f"  Import as: from {import_mod} import {', '.join(exports) if exports else 'N/A'}")
                        message_parts.append(f"  Exports: {exports}")

                    # Include actual code for key files (main.py, services, etc.)
                    message_parts.append("\n### ACTUAL SOURCE CODE ###")
                    priority_files = ['main.py'] + [f for f in py_files if 'service' in f.lower()]
                    other_files = [f for f in py_files if f not in priority_files]

                    for fname in priority_files + other_files:
                        if fname in files_dict:
                            content = files_dict[fname]
                            if len(content) > 3000:
                                content = content[:3000] + "\n... (truncated)"
                            message_parts.append(f"\n--- {fname} ---")
                            message_parts.append(content)
                else:
                    message_parts.append(f"\nSingle source file: {project_name}.py")
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
            
            code_task = next((t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")), None)
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
    
        # Store latest code separately — merge revisions into existing code
        if task.task_type in ("coding", "revision", "plan_execution", "build_from_plan") and task.result:
            if task.task_type == "revision" and self.state["context"].get("latest_code"):
                # Merge: parse both, overlay revision files onto existing
                existing_files = self._parse_multi_file_output(self.state["context"]["latest_code"])
                revision_files = self._parse_multi_file_output(task.result)
                if existing_files and revision_files:
                    existing_files.update(revision_files)
                    # Reassemble into ### FILE: format
                    merged_parts = []
                    for fname, content in existing_files.items():
                        merged_parts.append(f"### FILE: {fname} ###\n{content}")
                    self.state["context"]["latest_code"] = "\n\n".join(merged_parts)
                else:
                    # Fallback: just use the revision result as-is
                    self.state["context"]["latest_code"] = task.result
            else:
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
            (t for t in self.completed_tasks if t.task_type in ("coding", "revision", "plan_execution", "build_from_plan")),
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
3. Review smoke test results. Distinguish between:
   - Core code failures (import errors, syntax errors in main source) → these are real issues
   - Test scaffolding failures (test import paths wrong, missing test fixtures) → these are minor/cosmetic
   Only FAIL for core code issues, not test scaffolding problems.
4. Does the code implement the requested functionality completely?

OUTPUT FORMAT (MANDATORY):
- First non-empty line MUST be exactly one of:
  VERIFICATION: PASS
  VERIFICATION: WARN
  VERIFICATION: FAIL
- If WARN, also include: RECOMMENDATION: APPROVE or RECOMMENDATION: REJECT
- Then provide 3-8 bullet points explaining why.

Use WARN + APPROVE for projects where core code works but tests have minor issues.
Use FAIL only when the core application code has fundamental problems.

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