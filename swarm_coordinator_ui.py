#!/usr/bin/env python3
"""
SwarmCoordinator UI Version - Designed for web interface integration.

Based on SwarmCoordinator v2 but modified for non-interactive use:
- No input() calls - uses callbacks for user interaction
- Progress streaming via callbacks
- Clarification handled through UI
- Same agent logic and workflows

This file is separate from swarm_coordinator_v2.py to avoid breaking
the terminal-based workflow.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from enum import Enum
import requests
from threading import Lock
import re


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

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
    DATA_ANALYST = "data_analyst"  # New for Excel/data tasks


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_REVISION = "needs_revision"
    NEEDS_INPUT = "needs_input"  # New: waiting for user input


class SwarmState(Enum):
    """Overall swarm execution state"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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


@dataclass
class ProgressUpdate:
    """Progress update for streaming to UI"""
    update_type: str  # 'agent_start', 'agent_complete', 'status', 'question', 'result', 'error'
    agent: Optional[str] = None
    task_id: Optional[str] = None
    message: str = ""
    data: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.update_type,
            'agent': self.agent,
            'task_id': self.task_id,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }


# =============================================================================
# AGENT EXECUTOR
# =============================================================================

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
            fallback_map = {
                AgentRole.ARCHITECT: 'architect',
                AgentRole.CLARIFIER: 'clarifier',
                AgentRole.CODER: 'coder',
                AgentRole.REVIEWER: 'reviewer',
                AgentRole.TESTER: 'tester',
                AgentRole.OPTIMIZER: 'optimizer',
                AgentRole.DOCUMENTER: 'documenter',
                AgentRole.DEBUGGER: 'debugger',
                AgentRole.SECURITY: 'security',
                AgentRole.VERIFIER: 'verifier',
                AgentRole.DATA_ANALYST: 'coder'
            }
            fallback_role = fallback_map.get(role, 'coder')
            if fallback_role in self.config['model_config'].get('multi_model', {}):
                return self.config['model_config']['multi_model'][fallback_role]
        
        return self.config['model_config']['single_model']
    
    def _call_api(self, url: str, api_type: str, model: str, system_prompt: str, 
                  user_message: str, params: Dict, timeout: int) -> tuple:
        """Make API call to LLM"""
        
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
        
        agent_config = self._get_agent_config(role)
        url = agent_config['url']
        model = agent_config.get('model', 'local-model')
        api_type = agent_config.get('api_type', 'openai')
        timeout = agent_config.get('timeout', 7200)
        
        params = self.config.get('agent_parameters', {}).get(role.value, {}).copy()
        if agent_params:
            params.update(agent_params)
        
        agent_key = role.value
        with self.metrics_lock:
            if agent_key not in self.metrics:
                self.metrics[agent_key] = AgentMetrics(agent_name=agent_key, role=role.value)
        
        start_time = time.time()
        success = False
        tokens = 0
        response = ""
        
        try:
            response, tokens = self._call_api(url, api_type, model, system_prompt, 
                                             user_message, params, timeout)
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


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS = {
    AgentRole.ARCHITECT: """You are a software architect. Your role is to:
- Design clear, appropriate architecture for the given requirements
- Specify file structure and module organization
- Define interfaces between components
- Choose appropriate patterns (but don't over-engineer)

CRITICAL: Match complexity to requirements. Simple request = simple architecture.
Don't add enterprise patterns unless explicitly required.

Provide clear, structured architecture documentation.""",

    AgentRole.CLARIFIER: """You are a Requirements Clarification Agent.

CRITICAL INSTRUCTIONS:
1. Ask clarifying questions to fully understand requirements
2. You MUST NOT provide solutions or code
3. Format your response EXACTLY as shown below

When analyzing a request, identify:
- Missing technical specifications
- Unclear input/output formats
- Ambiguous requirements
- Unspecified edge cases

RESPONSE FORMAT:
CLARIFYING QUESTIONS:
1. [Specific question]
2. [Specific question]
...

ONLY if ALL requirements are crystal clear, respond with:
STATUS: CLEAR - All requirements are well-defined.""",

    AgentRole.CODER: """You are an expert programmer. Your role is to:
- Write code that WORKS correctly
- Include ONLY what's needed
- Handle errors gracefully
- Add comments explaining WHY

CRITICAL FILE STRUCTURE RULES:
1. If architect specifies MULTIPLE FILES, create ALL of them
2. Use header: ### FILE: filename.py ###
3. Do NOT combine into one file unless designed that way

CRITICAL OUTPUT RULES:
1. Output ONLY valid, executable Python code
2. Do NOT include markdown code blocks
3. Main entry should have: if __name__ == "__main__":

Focus on code that works correctly.""",

    AgentRole.REVIEWER: """You are a code reviewer. Your role is to:
- Find BUGS - syntax errors, logic errors, missing imports
- Verify code actually WORKS
- Check if code matches requirements

CRITICAL CHECKS:
1. Will this code run without errors?
2. What if inputs are None/empty?
3. Does this match what was asked for?

Return STATUS: APPROVED if code is correct.
Return STATUS: NEEDS_REVISION with SPECIFIC bugs if found.""",

    AgentRole.TESTER: """You are a test engineer. Your role is to:
- Generate comprehensive pytest test cases
- Include unit tests for each function
- Include edge case tests
- Provide clear test names

Output executable pytest code.""",

    AgentRole.DOCUMENTER: """You are a technical writer. Your role is to:
- Generate clear README documentation
- Document ONLY what exists in the code
- Include installation and usage
- Provide working examples

CRITICAL: Do NOT describe features that don't exist.
Base documentation ONLY on actual code provided.

Format as Markdown.""",

    AgentRole.VERIFIER: """You are a verification agent. Your role is to:
- Verify documentation matches actual code
- Check all referenced files exist
- Ensure examples work with the code

Respond with PASS or FAIL with specific issues.""",

    AgentRole.DATA_ANALYST: """You are a data analyst specializing in database schema design.

For Excel/CSV files, analyze and provide:
1. All columns and their data types
2. Primary key candidates
3. Foreign key relationships between sheets/tables
4. Recommended SQLite column types
5. Suggested indexes
6. Data quality notes

Output as structured schema design:
TABLE: table_name
  - column: TYPE [PRIMARY KEY] [NOT NULL]
  - column: TYPE [REFERENCES other_table(col)]
INDEX: column_name"""
}


# =============================================================================
# SWARM COORDINATOR UI
# =============================================================================

class SwarmCoordinatorUI:
    """
    Swarm Coordinator designed for web UI integration.
    
    Key differences from terminal version:
    - No input() calls
    - Progress streaming via callbacks
    - Clarification through callbacks
    - Non-blocking execution support
    """
    
    def __init__(self, config: Optional[Dict] = None, config_file: str = "config_v2.json"):
        """
        Initialize the coordinator.
        
        Args:
            config: Direct configuration dict (optional)
            config_file: Path to config file (used if config not provided)
        """
        self.config = config or self._load_config(config_file)
        self.executor = AgentExecutor(self.config)
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        
        self.state = {
            "workflow_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "phase": "initial",
            "swarm_state": SwarmState.IDLE,
            "iteration": 0,
            "max_iterations": self.config.get('workflow', {}).get('max_iterations', 3),
            "context": {
                "user_request": "",
                "clarification": "",
                "user_answers": ""
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
        
        # Callbacks for UI integration
        self._progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
        self._input_callback: Optional[Callable[[str, Dict], str]] = None
        self._pending_input: Optional[Dict] = None
        
        self._ensure_projects_dir()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
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
                    pass
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            "model_config": {
                "mode": "single",
                "single_model": {
                    "url": "http://localhost:1234/v1",
                    "model": "local-model",
                    "api_type": "openai",
                    "timeout": 7200
                }
            },
            "agent_parameters": {
                "architect": {"temperature": 0.6, "max_tokens": 3000},
                "clarifier": {"temperature": 0.7, "max_tokens": 2000},
                "coder": {"temperature": 0.5, "max_tokens": 6000},
                "reviewer": {"temperature": 0.8, "max_tokens": 3000},
                "tester": {"temperature": 0.7, "max_tokens": 4000},
                "documenter": {"temperature": 0.7, "max_tokens": 3000},
                "data_analyst": {"temperature": 0.3, "max_tokens": 4000},
                "verifier": {"temperature": 0.3, "max_tokens": 3000}
            },
            "workflow": {
                "max_iterations": 3,
                "max_parallel_agents": 4,
                "enable_parallel": True
            }
        }
    
    def _ensure_projects_dir(self):
        """Ensure projects directory exists"""
        if not os.path.exists(self.projects_root):
            os.makedirs(self.projects_root)
    
    def set_progress_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Set callback for progress updates"""
        self._progress_callback = callback
    
    def set_input_callback(self, callback: Callable[[str, Dict], str]):
        """
        Set callback for user input requests.
        
        Callback receives (question_type, data) and should return user's answer.
        For async UI, this might store the question and return None,
        then provide_input() is called later with the answer.
        """
        self._input_callback = callback
    
    def _emit_progress(self, update: ProgressUpdate):
        """Emit progress update to callback"""
        if self._progress_callback:
            self._progress_callback(update)
    
    def _request_input(self, question_type: str, data: Dict) -> Optional[str]:
        """
        Request input from user via callback.
        
        Returns answer if callback provides one, None if async.
        """
        if self._input_callback:
            return self._input_callback(question_type, data)
        return None

    # =========================================================================
    # PROJECT MANAGEMENT
    # =========================================================================
    
    def _get_next_project_number(self) -> int:
        """Get next project number"""
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
        """Generate project name from request"""
        words = user_request.lower().split()
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'create', 'make', 'build', 'write', 'i', 'need',
                     'want', 'please', 'help', 'that', 'this', 'which', 'what'}
        
        keywords = [w for w in words[:15] if w not in stop_words and len(w) > 2]
        name_parts = keywords[:3] if keywords else ['project']
        project_name = '_'.join(name_parts)
        project_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in project_name)
        project_name = re.sub(r'_+', '_', project_name).strip('_')
        
        return project_name or 'project'
    
    def _setup_project_directory(self, project_name: str, user_request: str) -> str:
        """Create project directory structure"""
        project_num = self._get_next_project_number()
        
        dir_name = f"{project_num:03d}_{project_name}_v1"
        project_dir = os.path.join(self.projects_root, dir_name)
        
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "tests"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "docs"), exist_ok=True)
        
        self.state["project_info"] = {
            "project_number": project_num,
            "project_name": project_name,
            "version": 1,
            "project_dir": project_dir
        }
        
        return project_dir
    
    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================
    
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
        """Get tasks ready to execute (dependencies met)"""
        ready = []
        completed_ids = {t.task_id for t in self.completed_tasks}
        
        for task in self.task_queue:
            if task.status == TaskStatus.PENDING:
                if all(dep_id in completed_ids for dep_id in task.dependencies):
                    ready.append(task)
        
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready
    
    def _get_system_prompt(self, role: AgentRole, task_type: str) -> str:
        """Get system prompt for an agent"""
        return SYSTEM_PROMPTS.get(role, f"You are a {role.value} agent.")
    
    def _build_user_message(self, task: Task) -> str:
        """Build user message for a task"""
        message_parts = []
        
        # Add user request
        user_request = self.state["context"].get("user_request", "")
        if user_request:
            message_parts.append(f"USER REQUEST:\n{user_request}\n")
        
        # Add clarification if available
        clarification = self.state["context"].get("clarification", "")
        if clarification:
            message_parts.append(f"CLARIFIED REQUIREMENTS:\n{clarification}\n")
        
        # Add task description
        message_parts.append(f"TASK: {task.description}")
        
        # Add context from dependencies
        if task.dependencies:
            message_parts.append("\nCONTEXT FROM PREVIOUS TASKS:")
            for dep_id in task.dependencies:
                dep_task = next((t for t in self.completed_tasks if t.task_id == dep_id), None)
                if dep_task and dep_task.result:
                    result = dep_task.result[:8000] + "..." if len(dep_task.result) > 8000 else dep_task.result
                    message_parts.append(f"\n[{dep_task.task_id} - {dep_task.assigned_role.value}]\n{result}")
        
        # Special handling for documenter
        if task.task_type == "documentation":
            code_task = next((t for t in self.completed_tasks if t.task_type == "coding"), None)
            if code_task and code_task.result:
                message_parts.append(f"\nACTUAL CODE TO DOCUMENT:\n{code_task.result}\n")
        
        return "\n".join(message_parts)
    
    def _clean_code_output(self, code: str) -> str:
        """Clean markdown from code output"""
        # Check for multi-file markers
        if '### FILE:' in code:
            lines = code.split('\n')
            cleaned = [line for line in lines if not line.strip().startswith('```')]
            return '\n'.join(cleaned).strip()
        
        # Extract from code blocks
        code_block_pattern = r'```(?:python)?\n(.*?)```'
        matches = re.findall(code_block_pattern, code, re.DOTALL)
        if matches:
            code = '\n\n'.join(matches)
        
        lines = code.split('\n')
        cleaned_lines = []
        found_code_start = False
        
        for line in lines:
            stripped = line.strip()
            
            if not found_code_start and not stripped:
                continue
            
            if re.match(r'^#{1,6}\s', line) and '### FILE:' not in line:
                continue
            
            code_indicators = (
                stripped.startswith('import '),
                stripped.startswith('from '),
                stripped.startswith('class '),
                stripped.startswith('def '),
                stripped.startswith('"""'),
                stripped.startswith('@'),
                stripped.startswith('#!'),
                '### FILE:' in stripped
            )
            
            if any(code_indicators):
                found_code_start = True
            
            if found_code_start:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _update_context(self, task: Task):
        """Update shared context with task results"""
        context_key = f"{task.task_type}_{task.assigned_role.value}"
        self.state["context"][context_key] = {
            "task_id": task.task_id,
            "result": task.result,
            "completed_at": task.completed_at
        }
        
        if task.task_type == "coding" and task.result:
            self.state["context"]["latest_code"] = task.result

    # =========================================================================
    # TASK EXECUTION
    # =========================================================================
    
    def execute_task(self, task: Task) -> Task:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS
        
        # Emit progress
        self._emit_progress(ProgressUpdate(
            update_type='agent_start',
            agent=task.assigned_role.value,
            task_id=task.task_id,
            message=f'{task.assigned_role.value} starting: {task.description}'
        ))
        
        try:
            system_prompt = self._get_system_prompt(task.assigned_role, task.task_type)
            user_message = self._build_user_message(task)
            
            result = self.executor.execute_agent(
                role=task.assigned_role,
                system_prompt=system_prompt,
                user_message=user_message
            )
            
            # Clean code output
            if task.task_type == "coding" and result:
                result = self._clean_code_output(result)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # Update context
            self._update_context(task)
            
            # Check for reviewer rejection
            if task.task_type == "review" and result:
                if "NEEDS_REVISION" in result.upper() or "STATUS: REJECT" in result.upper():
                    task.metadata["needs_revision"] = True
            
            # Emit completion
            self._emit_progress(ProgressUpdate(
                update_type='agent_complete',
                agent=task.assigned_role.value,
                task_id=task.task_id,
                message=f'{task.assigned_role.value} completed'
            ))
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            self._emit_progress(ProgressUpdate(
                update_type='agent_error',
                agent=task.assigned_role.value,
                task_id=task.task_id,
                message=f'{task.assigned_role.value} failed: {str(e)}'
            ))
        
        return task
    
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
    
    def _handle_clarification(self, task: Task) -> bool:
        """
        Handle clarification task - UI version.
        
        Instead of input(), emits a question event and waits for response.
        Returns True if clarification was handled, False if waiting.
        """
        if not task.result:
            return True
        
        result = task.result.strip()
        
        # Check if requirements are already clear
        if "STATUS: CLEAR" in result.upper():
            self.state["context"]['clarification'] = "Requirements are clear and complete."
            self._emit_progress(ProgressUpdate(
                update_type='status',
                message='Requirements are clear, proceeding...'
            ))
            return True
        
        # Emit question to UI
        self._emit_progress(ProgressUpdate(
            update_type='question',
            agent='clarifier',
            message='Clarification needed',
            data={
                'questions': result,
                'task_id': task.task_id
            }
        ))
        
        # Try to get input via callback
        answer = self._request_input('clarification', {'questions': result})
        
        if answer:
            # Got answer synchronously
            user_request = self.state["context"].get("user_request", "")
            clarified_req = f"ORIGINAL REQUEST:\n{user_request}\n\n"
            clarified_req += f"CLARIFICATION QUESTIONS:\n{result}\n\n"
            clarified_req += f"USER ANSWERS:\n{answer}"
            
            self.state["context"]['clarification'] = clarified_req
            self.state["context"]['user_answers'] = answer
            return True
        else:
            # Waiting for async input
            self._pending_input = {
                'type': 'clarification',
                'questions': result,
                'task_id': task.task_id
            }
            self.state["swarm_state"] = SwarmState.WAITING_FOR_INPUT
            return False
    
    def provide_input(self, input_type: str, answer: str):
        """
        Provide input that was requested asynchronously.
        
        Call this when UI receives user's answer to a question.
        """
        if input_type == 'clarification':
            user_request = self.state["context"].get("user_request", "")
            questions = self._pending_input.get('questions', '') if self._pending_input else ''
            
            clarified_req = f"ORIGINAL REQUEST:\n{user_request}\n\n"
            clarified_req += f"CLARIFICATION QUESTIONS:\n{questions}\n\n"
            clarified_req += f"USER ANSWERS:\n{answer}"
            
            self.state["context"]['clarification'] = clarified_req
            self.state["context"]['user_answers'] = answer
            self._pending_input = None
            self.state["swarm_state"] = SwarmState.RUNNING

    # =========================================================================
    # WORKFLOW CREATION
    # =========================================================================
    
    def _create_standard_workflow(self, user_request: str):
        """Create standard workflow: clarify -> architect -> code -> review -> document"""
        
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
            description="Design system architecture",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=9,
            dependencies=["T001_clarify"],
            metadata={"user_request": user_request}
        ))
        
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
        
        self.add_task(Task(
            task_id="T004_review",
            task_type="review",
            description="Review code",
            assigned_role=AgentRole.REVIEWER,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_code"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T005_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T004_review"],
            metadata={"user_request": user_request}
        ))
    
    def _create_data_workflow(self, user_request: str):
        """Create data-focused workflow for Excel/DB tasks"""
        
        self.add_task(Task(
            task_id="T001_clarify",
            task_type="clarification",
            description="Clarify data requirements",
            assigned_role=AgentRole.CLARIFIER,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T002_analyze",
            task_type="data_analysis",
            description="Analyze data structure and design schema",
            assigned_role=AgentRole.DATA_ANALYST,
            status=TaskStatus.PENDING,
            priority=9,
            dependencies=["T001_clarify"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T003_architect",
            task_type="architecture",
            description="Design code architecture",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=8,
            dependencies=["T002_analyze"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T004_code",
            task_type="coding",
            description="Implement parser and database code",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=7,
            dependencies=["T003_architect"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T005_review",
            task_type="review",
            description="Review code",
            assigned_role=AgentRole.REVIEWER,
            status=TaskStatus.PENDING,
            priority=6,
            dependencies=["T004_code"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T006_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=5,
            dependencies=["T005_review"],
            metadata={"user_request": user_request}
        ))
    
    def _create_quick_workflow(self, user_request: str):
        """Create quick workflow: architect -> code -> document (no clarification)"""
        
        self.add_task(Task(
            task_id="T001_architect",
            task_type="architecture",
            description="Design architecture",
            assigned_role=AgentRole.ARCHITECT,
            status=TaskStatus.PENDING,
            priority=10,
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T002_code",
            task_type="coding",
            description="Implement code",
            assigned_role=AgentRole.CODER,
            status=TaskStatus.PENDING,
            priority=9,
            dependencies=["T001_architect"],
            metadata={"user_request": user_request}
        ))
        
        self.add_task(Task(
            task_id="T003_document",
            task_type="documentation",
            description="Generate documentation",
            assigned_role=AgentRole.DOCUMENTER,
            status=TaskStatus.PENDING,
            priority=8,
            dependencies=["T002_code"],
            metadata={"user_request": user_request}
        ))

    # =========================================================================
    # MAIN WORKFLOW EXECUTION
    # =========================================================================
    
    def run_workflow(
        self,
        user_request: str,
        workflow_type: str = "standard",
        files: Optional[List[str]] = None,
        skip_clarification: bool = False
    ) -> Iterator[ProgressUpdate]:
        """
        Execute a workflow and yield progress updates.
        
        Args:
            user_request: The user's task description
            workflow_type: 'standard', 'data', 'quick'
            files: Optional list of input file paths
            skip_clarification: Skip clarification step
        
        Yields:
            ProgressUpdate objects for streaming to UI
        """
        self.state["swarm_state"] = SwarmState.RUNNING
        self.state["context"]["user_request"] = user_request
        
        # Add file info to request if provided
        if files:
            file_info = self._analyze_files(files)
            user_request = f"{user_request}\n\nINPUT FILES:\n{file_info}"
            self.state["context"]["user_request"] = user_request
        
        # Setup project
        project_name = self._create_project_name(user_request)
        project_dir = self._setup_project_directory(project_name, user_request)
        
        yield ProgressUpdate(
            update_type='status',
            message=f'Project created: {os.path.basename(project_dir)}'
        )
        
        # Create workflow tasks
        if workflow_type == "data":
            self._create_data_workflow(user_request)
        elif workflow_type == "quick" or skip_clarification:
            self._create_quick_workflow(user_request)
        else:
            self._create_standard_workflow(user_request)
        
        yield ProgressUpdate(
            update_type='status',
            message=f'Workflow created with {len(self.task_queue)} tasks'
        )
        
        # Execute tasks
        iteration = 0
        max_iterations = self.state["max_iterations"] * 10
        
        while self.task_queue and iteration < max_iterations:
            # Check if waiting for input
            if self.state["swarm_state"] == SwarmState.WAITING_FOR_INPUT:
                yield ProgressUpdate(
                    update_type='waiting',
                    message='Waiting for user input...'
                )
                break
            
            iteration += 1
            ready_tasks = self.get_ready_tasks()
            
            if not ready_tasks:
                pending = [t for t in self.task_queue if t.status == TaskStatus.PENDING]
                if pending:
                    yield ProgressUpdate(
                        update_type='status',
                        message=f'{len(pending)} tasks blocked waiting for dependencies'
                    )
                break
            
            yield ProgressUpdate(
                update_type='status',
                message=f'Executing {len(ready_tasks)} task(s)...'
            )
            
            # Execute tasks
            completed = self.execute_tasks_parallel(ready_tasks)
            
            for task in completed:
                if task in self.task_queue:
                    self.task_queue.remove(task)
                self.completed_tasks.append(task)
                
                # Handle clarification specially
                if task.task_type == "clarification" and task.status == TaskStatus.COMPLETED:
                    handled = self._handle_clarification(task)
                    if not handled:
                        # Waiting for input
                        yield ProgressUpdate(
                            update_type='waiting',
                            message='Waiting for clarification answers...',
                            data={'questions': task.result}
                        )
                        return  # Exit generator, will be resumed after input
                
                yield ProgressUpdate(
                    update_type='task_complete',
                    task_id=task.task_id,
                    agent=task.assigned_role.value,
                    message=f'{task.assigned_role.value}: {task.status.value}'
                )
        
        # Save outputs
        self._save_project_outputs()
        
        self.state["swarm_state"] = SwarmState.COMPLETED
        
        yield ProgressUpdate(
            update_type='complete',
            message='Workflow completed',
            data={
                'project_dir': project_dir,
                'files_created': self._get_created_files(project_dir),
                'tasks_completed': len(self.completed_tasks),
                'tasks_failed': len([t for t in self.completed_tasks if t.status == TaskStatus.FAILED])
            }
        )
    
    def resume_workflow(self) -> Iterator[ProgressUpdate]:
        """
        Resume workflow after user provided input.
        
        Call this after provide_input() to continue execution.
        """
        if self.state["swarm_state"] != SwarmState.RUNNING:
            return
        
        yield from self._continue_execution()
    
    def _continue_execution(self) -> Iterator[ProgressUpdate]:
        """Continue workflow execution"""
        iteration = 0
        max_iterations = self.state["max_iterations"] * 10
        
        while self.task_queue and iteration < max_iterations:
            if self.state["swarm_state"] == SwarmState.WAITING_FOR_INPUT:
                break
            
            iteration += 1
            ready_tasks = self.get_ready_tasks()
            
            if not ready_tasks:
                break
            
            completed = self.execute_tasks_parallel(ready_tasks)
            
            for task in completed:
                if task in self.task_queue:
                    self.task_queue.remove(task)
                self.completed_tasks.append(task)
                
                if task.task_type == "clarification" and task.status == TaskStatus.COMPLETED:
                    handled = self._handle_clarification(task)
                    if not handled:
                        yield ProgressUpdate(
                            update_type='waiting',
                            message='Waiting for clarification answers...'
                        )
                        return
                
                yield ProgressUpdate(
                    update_type='task_complete',
                    task_id=task.task_id,
                    agent=task.assigned_role.value,
                    message=f'{task.assigned_role.value}: {task.status.value}'
                )
        
        # Save outputs
        self._save_project_outputs()
        
        project_dir = self.state["project_info"].get("project_dir", "")
        self.state["swarm_state"] = SwarmState.COMPLETED
        
        yield ProgressUpdate(
            update_type='complete',
            message='Workflow completed',
            data={
                'project_dir': project_dir,
                'files_created': self._get_created_files(project_dir)
            }
        )

    # =========================================================================
    # FILE HANDLING AND OUTPUT
    # =========================================================================
    
    def _analyze_files(self, files: List[str]) -> str:
        """Analyze input files and return description"""
        descriptions = []
        
        for file_path in files:
            path = file_path if isinstance(file_path, str) else str(file_path)
            
            if not os.path.exists(path):
                descriptions.append(f"- {os.path.basename(path)}: (file not found)")
                continue
            
            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in ['.xlsx', '.xls']:
                descriptions.append(self._describe_excel(path))
            elif ext == '.csv':
                descriptions.append(self._describe_csv(path))
            elif ext == '.json':
                descriptions.append(self._describe_json(path))
            else:
                size = os.path.getsize(path)
                descriptions.append(f"- {filename}: {size} bytes")
        
        return '\n'.join(descriptions)
    
    def _describe_excel(self, path: str) -> str:
        """Describe Excel file structure"""
        try:
            import pandas as pd
            xl = pd.ExcelFile(path)
            desc = [f"- {os.path.basename(path)}: Excel with sheets: {', '.join(xl.sheet_names)}"]
            
            for sheet in xl.sheet_names[:3]:
                df = xl.parse(sheet, nrows=5)
                desc.append(f"  Sheet '{sheet}': columns = {list(df.columns)}")
            
            return '\n'.join(desc)
        except ImportError:
            return f"- {os.path.basename(path)}: Excel file (pandas not available)"
        except Exception as e:
            return f"- {os.path.basename(path)}: Excel file (error: {e})"
    
    def _describe_csv(self, path: str) -> str:
        """Describe CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
            return f"- {os.path.basename(path)}: CSV with {len(df.columns)} columns: {list(df.columns)}"
        except:
            return f"- {os.path.basename(path)}: CSV file"
    
    def _describe_json(self, path: str) -> str:
        """Describe JSON file"""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return f"- {os.path.basename(path)}: JSON array with {len(data)} items"
            elif isinstance(data, dict):
                return f"- {os.path.basename(path)}: JSON object with keys: {list(data.keys())[:5]}"
            return f"- {os.path.basename(path)}: JSON file"
        except:
            return f"- {os.path.basename(path)}: JSON file"
    
    def _get_created_files(self, project_dir: str) -> List[str]:
        """Get list of files created"""
        files = []
        if project_dir and os.path.exists(project_dir):
            for root, dirs, filenames in os.walk(project_dir):
                for filename in filenames:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, project_dir)
                    files.append(rel_path)
        return files
    
    def _save_project_outputs(self):
        """Save all outputs to project directory"""
        project_dir = self.state["project_info"].get("project_dir")
        if not project_dir:
            return
        
        # Save code
        code_task = next((t for t in self.completed_tasks if t.task_type == "coding"), None)
        if code_task and code_task.result:
            files_dict = self._parse_multi_file_output(code_task.result)
            
            if files_dict:
                for filename, content in files_dict.items():
                    code_file = os.path.join(project_dir, "src", filename)
                    os.makedirs(os.path.dirname(code_file), exist_ok=True)
                    with open(code_file, 'w') as f:
                        f.write(content)
            else:
                project_name = self.state["project_info"]["project_name"]
                code_file = os.path.join(project_dir, "src", f"{project_name}.py")
                with open(code_file, 'w') as f:
                    f.write(code_task.result)
        
        # Save documentation
        doc_task = next((t for t in self.completed_tasks if t.task_type == "documentation"), None)
        if doc_task and doc_task.result:
            doc_file = os.path.join(project_dir, "README.md")
            with open(doc_file, 'w') as f:
                f.write(doc_task.result)
        
        # Save tests
        test_task = next((t for t in self.completed_tasks if t.task_type == "test_generation"), None)
        if test_task and test_task.result:
            test_file = os.path.join(project_dir, "tests", "test_main.py")
            with open(test_file, 'w') as f:
                f.write(test_task.result)
        
        # Create requirements.txt
        self._create_requirements(project_dir)
    
    def _parse_multi_file_output(self, code_output: str) -> Dict[str, str]:
        """Parse multi-file output from coder"""
        file_pattern = r'###\s*FILE:\s*([^\s#]+)\s*###'
        matches = list(re.finditer(file_pattern, code_output))
        
        if not matches:
            return {}
        
        files = {}
        for i, match in enumerate(matches):
            filename = match.group(1).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(code_output)
            
            content = code_output[start_pos:end_pos].strip()
            content = '\n'.join(line for line in content.split('\n') 
                               if not line.strip().startswith('```'))
            
            if content:
                files[filename] = content
        
        return files
    
    def _create_requirements(self, project_dir: str):
        """Generate requirements.txt from imports"""
        code_task = next((t for t in self.completed_tasks if t.task_type == "coding"), None)
        if not code_task or not code_task.result:
            return
        
        stdlib = {'os', 'sys', 'json', 'time', 'datetime', 'collections', 're',
                 'math', 'random', 'itertools', 'functools', 'pathlib', 'typing',
                 'logging', 'argparse', 'threading', 'copy', 'dataclasses', 'enum'}
        
        imports = set()
        for line in code_task.result.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                module = line.replace('import ', '').split()[0].split('.')[0].split(',')[0]
                if module and module not in stdlib:
                    imports.add(module)
            elif line.startswith('from ') and ' import ' in line:
                module = line.replace('from ', '').split()[0].split('.')[0]
                if module and module not in stdlib:
                    imports.add(module)
        
        if imports:
            req_file = os.path.join(project_dir, "requirements.txt")
            with open(req_file, 'w') as f:
                for imp in sorted(imports):
                    f.write(f"{imp}\n")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_state(self) -> Dict:
        """Get current swarm state"""
        return {
            'workflow_id': self.state['workflow_id'],
            'swarm_state': self.state['swarm_state'].value,
            'project_info': self.state['project_info'],
            'pending_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'waiting_for_input': self._pending_input is not None
        }
    
    def get_metrics(self) -> Dict:
        """Get execution metrics"""
        return {
            'agents': {k: asdict(v) for k, v in self.executor.metrics.items()},
            'tasks': {
                'total': len(self.completed_tasks),
                'completed': len([t for t in self.completed_tasks if t.status == TaskStatus.COMPLETED]),
                'failed': len([t for t in self.completed_tasks if t.status == TaskStatus.FAILED])
            }
        }
    
    def cancel(self):
        """Cancel the workflow"""
        self.state["swarm_state"] = SwarmState.CANCELLED
        self.task_queue.clear()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_swarm_ui(config: Optional[Dict] = None) -> SwarmCoordinatorUI:
    """Create a SwarmCoordinatorUI instance"""
    return SwarmCoordinatorUI(config)
