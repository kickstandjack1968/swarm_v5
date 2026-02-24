# Swarm v5 Codebase Improvement Analysis

**Date:** February 11, 2026  
**Analyzed by:** opencode  
**Scope:** Core swarm coordination system

---

## Executive Summary

The Swarm v5 codebase shows signs of rapid development without proper refactoring. The main issues are overly large classes, poor error handling, hardcoded configurations, and lack of proper logging. This document outlines specific improvement opportunities with prioritized recommendations.

## Codebase Overview

```
Core Files Analysis:
├── src/swarm_coordinator_v2.py    4,703 lines (CRITICAL)
├── src/plan_executor_v2.py        1,895 lines (HIGH)  
├── swarm_coordinator_ui.py        1,436 lines (HIGH)
├── src/agent_base.py               532 lines (MEDIUM)
├── service.py                      328 lines (LOW)
```

---

## 🚨 Critical Issues

### 1. God Classes & Massive Files

**Problem:** Single files doing too much, violating Single Responsibility Principle

**Files Affected:**
- `src/swarm_coordinator_v2.py:4703` - Workflow orchestration, task management, Docker, metrics, state management
- `src/plan_executor_v2.py:1895` - Plan execution, file management, validation, Docker operations
- `swarm_coordinator_ui.py:1436` - UI logic mixed with business logic

**Impact:** 
- Difficult to maintain and test
- High coupling between components
- Code duplication across workflows

**Solution:**
```python
# Proposed Refactoring for SwarmCoordinator:
class SwarmCoordinator:
    def __init__(self):
        self.workflow_builder = WorkflowBuilder()
        self.task_manager = TaskManager() 
        self.state_manager = StateManager()
        self.metrics_collector = MetricsCollector()
        self.docker_manager = DockerManager()

# Split into focused classes:
- WorkflowBuilder: Creates workflow definitions
- TaskManager: Handles task execution and routing
- StateManager: Manages persistent state
- MetricsCollector: Tracks performance metrics
- DockerManager: Manages sandbox containers
```

### 2. Poor Error Handling

**Problem:** Bare except clauses and generic exception handling

**Locations:**
- `src/plan_executor_v2.py:1114` - `except:`
- `src/plan_executor_v2.py:1153` - `except:`
- `src/plan_executor_v2.py:1162` - `except:`
- `src/plan_executor_v2.py:1172` - `except:`

**Impact:**
- Silent failures
- Difficult debugging
- No error recovery

**Solution:**
```python
# Replace bare except with specific handling
try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    result.check_returncode()
except subprocess.CalledProcessError as e:
    self.logger.error(f"Command failed: {cmd}, error: {e.stderr}")
    raise ProcessError(f"Subprocess failed: {e}")
except FileNotFoundError:
    self.logger.error(f"Command not found: {cmd[0]}")
    raise ProcessError(f"Missing executable: {cmd[0]}")
```

### 3. Hardcoded Configuration

**Problem:** Configuration scattered throughout codebase

**Issues Found:**
- LLM URL: `http://localhost:1234/v1` (30+ occurrences)
- Timeouts: 60, 120, 30 seconds scattered throughout
- Docker image: `python:3.12-slim` hardcoded
- Package lists and paths embedded in code

**Solution:**
```python
# config.py
@dataclass
class SwarmConfig:
    # LLM Configuration
    llm_url: str = field(default_factory=lambda: os.getenv('LLM_URL', 'http://localhost:1234/v1'))
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'local-model'))
    default_timeout: int = field(default_factory=lambda: int(os.getenv('DEFAULT_TIMEOUT', '60')))
    max_retries: int = field(default_factory=lambda: int(os.getenv('MAX_RETRIES', '3')))
    
    # Docker Configuration  
    docker_image: str = field(default_factory=lambda: os.getenv('DOCKER_IMAGE', 'python:3.12-slim'))
    sandbox_name: str = "swarm_sandbox"
    
    # Paths
    workspace_dir: str = field(default_factory=lambda: os.getenv('WORKSPACE_DIR', './workspace'))
    log_dir: str = field(default_factory=lambda: os.getenv('LOG_DIR', './logs'))
```

---

## ⚠️ Code Quality Issues

### 4. Print Statements Instead of Logging

**Problem:** 100+ print() calls for status updates and debugging

**Files with print statements:**
- `src/agent_base.py` - Debug and error output
- `src/architect/architect_agent.py` - Progress updates
- `src/clarifier/clarifier_agent.py` - Status messages
- `src/reviewer/reviewer_agent.py` - Review progress
- `src/verifier/verifier_agent.py` - Verification status

**Impact:**
- No log levels or filtering
- Can't disable debug output
- Poor production monitoring

**Solution:**
```python
# Replace print statements with proper logging
import logging

class AgentBase:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, input_data):
        # Instead of: print(f"Processing request: {input_data}")
        self.logger.info(f"Processing request: {input_data.get('type', 'unknown')}")
        
        # Instead of: print(f"⚠ Warning: {msg}")
        self.logger.warning(msg)
        
        # Instead of: print(f"❌ Error: {error}")
        self.logger.error(error)
```

### 5. Long Complex Methods

**Problem:** Methods doing too much, hard to understand and test

**Examples:**
- `run_workflow()` (lines 3325-3450): 125+ lines handling initialization, execution, cleanup
- `execute_plan()` (lines 1540-1632): Complex execution logic with nested conditions
- `_create_full_workflow()` (lines 3548-3650): Repetitive task creation patterns

**Solution:**
```python
def run_workflow(self, user_request: str, workflow_type: str = "standard"):
    """Main workflow execution entry point."""
    self._initialize_workspace(user_request, workflow_type)
    workflow = self._get_workflow_builder(workflow_type)
    tasks = workflow.build(user_request)
    return self._execute_workflow(tasks)

def _initialize_workspace(self, user_request: str, workflow_type: str):
    """Initialize workspace and required directories."""
    
def _get_workflow_builder(self, workflow_type: str) -> WorkflowBuilder:
    """Get appropriate workflow builder for type."""
    
def _execute_workflow(self, tasks: List[Task]) -> WorkflowResult:
    """Execute all tasks and collect results."""
```

### 6. Missing Type Hints

**Problem:** 40+ methods lack proper type annotations

**Impact:**
- Poor IDE support
- Runtime errors instead of compile-time
- Difficult to understand interfaces

**Solution:**
```python
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

def execute_task(self, task: Task) -> TaskResult:
    """Execute a single task and return the result."""
    
def create_workflow(self, user_request: str) -> List[Task]:
    """Create workflow tasks from user request."""
    
def call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
    """Call LLM with message history."""
```

---

## 🐌 Performance Issues

### 7. Inefficient Patterns

**Problem:** Sequential execution despite having parallel infrastructure

**Issues Found:**
- Sequential LLM calls in workflow execution
- No caching of agent responses
- Loading entire files into memory
- Repeated context building

**Solution:**
```python
from functools import lru_cache
import asyncio

# Add caching for expensive operations
@lru_cache(maxsize=128)
def _get_agent_config(self, role: AgentRole) -> Dict:
    return self._load_agent_config(role)

# Batch LLM calls where possible
async def _batch_llm_calls(self, requests: List[LLMRequest]) -> List[str]:
    """Make multiple LLM calls in parallel."""
    semaphore = asyncio.Semaphore(5)  # Limit concurrent calls
    
    async def single_call(req):
        async with semaphore:
            return await self._call_llm_async(req)
    
    return await asyncio.gather(*[single_call(req) for req in requests])

# Stream file processing for large files
def _process_file_stream(self, file_path: str) -> Iterator[str]:
    """Process file line by line instead of loading entire file."""
    with open(file_path, 'r') as f:
        for line in f:
            yield self._process_line(line)
```

---

## 🔧 Maintainability Issues

### 8. Code Duplication

**Problem:** Same patterns repeated across multiple files

**Duplicated Areas:**
- Agent configuration loading (15+ occurrences)
- LLM calling logic (agent_base.py and individual agents)
- Task creation patterns in workflow methods
- Error handling patterns

**Solution:**
```python
# Create base agent with common functionality
class BaseAgent:
    def __init__(self, role: AgentRole, config: AgentConfig):
        self.role = role
        self.config = config
        self.llm_client = LLMClient(config)
    
    def call_llm(self, prompt: str, **kwargs) -> str:
        """Standardized LLM calling with retry logic."""
        return self.llm_client.call(prompt, **kwargs)
    
    def get_config(self, key: str, default=None):
        """Get role-specific configuration."""
        return self.config.get_agent_config(self.role, key, default)

# Standard workflow builders
class WorkflowBuilder:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.task_factory = TaskFactory()
    
    def create_agent_task(self, role: AgentRole, context: Dict) -> Task:
        """Standardized task creation for agents."""
        return self.task_factory.create_agent_task(role, context)
```

### 9. Missing Tests

**Problem:** Virtually no test coverage for core functionality

**Current State:**
- Only 1 test file: `tests/coder_agent.py`
- No unit tests for SwarmCoordinator
- No integration tests for workflows
- No tests for error conditions

**Solution:**
```python
# tests/test_swarm_coordinator.py
import pytest
from unittest.mock import Mock, patch

class TestSwarmCoordinator:
    def test_run_workflow_standard(self):
        """Test standard workflow execution."""
        
    def test_run_workflow_with_files(self):
        """Test workflow with file inputs."""
        
    def test_error_handling_llm_failure(self):
        """Test graceful handling of LLM failures."""
        
    def test_parallel_task_execution(self):
        """Test parallel execution of independent tasks."""

# tests/integration/test_workflows.py  
class TestWorkflowIntegration:
    def test_full_coding_workflow(self):
        """Test complete coding workflow from request to completion."""
        
    def test_workflow_with_rejections(self):
        """Test workflow when reviewer rejects code."""
```

---

## 🔒 Security Issues

### 10. Input Validation

**Problem:** No validation of user requests and LLM inputs

**Risks:**
- Injection attacks in generated code
- Path traversal in file operations
- Resource exhaustion attacks

**Solution:**
```python
import re
from pathlib import Path

class InputValidator:
    @staticmethod
    def validate_user_request(request: str) -> str:
        """Validate and sanitize user request."""
        if len(request) > 10000:
            raise ValueError("Request too long")
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'rm\s+-rf',
            r'sudo\s+',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, request, re.IGNORECASE):
                raise ValueError("Request contains potentially dangerous commands")
        
        return request.strip()
    
    @staticmethod  
    def validate_file_path(file_path: str) -> Path:
        """Validate file path to prevent directory traversal."""
        try:
            path = Path(file_path).resolve()
            if not str(path).startswith(Path.cwd()):
                raise ValueError("Path traversal detected")
            return path
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")
```

---

## 📋 Action Plan

### Phase 1: Critical Fixes (Week 1-2)
1. **Extract Configuration** - Create `config.py` with environment variables
2. **Implement Logging** - Replace all print statements with proper logging
3. **Fix Error Handling** - Replace bare except clauses with specific exceptions
4. **Add Type Hints** - All public methods and data structures

### Phase 2: Architecture Refactoring (Week 3-4)
1. **Split SwarmCoordinator** - Extract WorkflowBuilder, TaskManager, StateManager
2. **Create Base Classes** - Standardize agent interfaces and common functionality
3. **Implement Caching** - LLM responses and file operations
4. **Add Input Validation** - Security measures and validation

### Phase 3: Testing & Performance (Week 5-6)
1. **Add Unit Tests** - Core SwarmCoordinator functionality
2. **Add Integration Tests** - End-to-end workflows
3. **Performance Optimization** - Async operations, batching
4. **Monitoring & Metrics** - Structured logging and metrics

### Phase 4: Documentation & Polish (Week 7-8)
1. **API Documentation** - Comprehensive docstrings and type hints
2. **Architecture Guide** - High-level documentation
3. **CI/CD Pipeline** - Automated testing and quality checks
4. **Examples & Tutorials** - Usage examples

---

## 🎯 Success Metrics

- **Code Coverage:** Target 80%+ test coverage
- **Cyclomatic Complexity:** Reduce maximum complexity to <10 per method
- **File Size:** No single file >2000 lines
- **Type Coverage:** 95%+ of functions with type hints
- **Error Handling:** 0 bare except clauses
- **Logging:** 0 print statements in production code

---

## 📊 Technical Debt Summary

| Category | Current State | Target State | Effort |
|----------|---------------|--------------|--------|
| File Size | 4,703 lines max | <2,000 lines | High |
| Type Coverage | ~60% | >95% | Medium |
| Test Coverage | <5% | >80% | High |
| Error Handling | Poor | Comprehensive | Medium |
| Configuration | Hardcoded | Environment-based | Low |
| Logging | Print statements | Structured logging | Medium |

**Total Estimated Effort:** 6-8 weeks for full refactoring

---

## 📝 Notes

- All changes should be backward compatible where possible
- Consider creating v2.0 branch for major breaking changes
- Prioritize fixes that improve debugging and maintainability
- Document all breaking changes with migration guides

---

*This analysis was generated using automated code review and manual inspection. Priority should be given to critical issues first, followed by architecture improvements that will enable easier maintenance going forward.*