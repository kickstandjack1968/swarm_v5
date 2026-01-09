#!/usr/bin/env python3
"""
Agent Base Module for SwarmCoordinator
======================================

Shared infrastructure for all external agent scripts.
Provides common LLM calling, error handling, and response parsing.

Usage:
    from agent_base import AgentBase, AgentError
    
    class MyAgent(AgentBase):
        def process(self, input_data: dict) -> dict:
            # Custom processing logic
            response = self.call_llm(messages)
            return {"status": "success", "result": response}
    
    if __name__ == "__main__":
        MyAgent().run()
"""

import sys
import json
import re
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


class AgentError(Exception):
    """Custom exception for agent errors with error codes."""
    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""
    url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    api_type: str = "openai"  # "openai" or "ollama"
    timeout: int = 600
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 0.9
    
    @classmethod
    def from_dict(cls, config: dict) -> 'LLMConfig':
        """Create config from dictionary, handling various key formats."""
        return cls(
            url=config.get("model_url") or config.get("url", "http://localhost:1234/v1"),
            model=config.get("model_name") or config.get("model", "local-model"),
            api_type=config.get("api_type", "openai"),
            timeout=config.get("timeout", 600),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 4000),
            top_p=config.get("top_p", 0.9)
        )


class AgentBase(ABC):
    """
    Base class for all SwarmCoordinator agents.
    
    Provides:
    - Standardized input/output handling
    - Common LLM calling with retries
    - Error handling and reporting
    - Response cleaning utilities
    """
    
    # Error codes
    ERR_INVALID_INPUT = "INVALID_INPUT"
    ERR_MISSING_FIELD = "MISSING_FIELD"
    ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
    ERR_LLM_ERROR = "LLM_ERROR"
    ERR_PARSE_ERROR = "PARSE_ERROR"
    ERR_INTERNAL = "INTERNAL_ERROR"
    
    def __init__(self):
        self.config: Optional[LLMConfig] = None
        self.input_data: dict = {}
    
    def run(self):
        """Main entry point - reads stdin, processes, writes stdout."""
        try:
            # Read and parse input
            raw_input = sys.stdin.read()
            if not raw_input.strip():
                self._fail(self.ERR_INVALID_INPUT, "Empty input")
            
            try:
                self.input_data = json.loads(raw_input)
            except json.JSONDecodeError as e:
                self._fail(self.ERR_INVALID_INPUT, f"Invalid JSON: {e}")
            
            # Extract config
            config_dict = self.input_data.get("config", {})
            self.config = LLMConfig.from_dict(config_dict)
            
            # Process (implemented by subclass)
            result = self.process(self.input_data)
            
            # Output result
            print(json.dumps(result, indent=2))
            
        except AgentError as e:
            self._fail(e.code, e.message, e.details)
        except Exception as e:
            self._fail(self.ERR_INTERNAL, str(e))
    
    @abstractmethod
    def process(self, input_data: dict) -> dict:
        """
        Process the input and return result.
        Must be implemented by subclasses.
        
        Returns:
            dict with at minimum {"status": "success"|"error", ...}
        """
        pass
    
    def call_llm(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = None,
        max_tokens: int = None,
        retries: int = 2
    ) -> str:
        """
        Make API call to LLM with retry logic.
        
        Args:
            messages: List of {"role": str, "content": str}
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            retries: Number of retry attempts
            
        Returns:
            Response content string
            
        Raises:
            AgentError on failure
        """
        if not self.config:
            raise AgentError(self.ERR_INTERNAL, "LLM config not initialized")
        
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        for attempt in range(retries + 1):
            try:
                if self.config.api_type == 'ollama':
                    return self._call_ollama(messages, temp, tokens)
                else:
                    return self._call_openai(messages, temp, tokens)
                    
            except requests.exceptions.Timeout:
                if attempt < retries:
                    continue
                raise AgentError(
                    self.ERR_LLM_TIMEOUT, 
                    f"LLM request timed out after {self.config.timeout}s"
                )
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    continue
                raise AgentError(self.ERR_LLM_ERROR, f"LLM request failed: {e}")
    
    def _call_openai(self, messages: list, temperature: float, max_tokens: int) -> str:
        """Call OpenAI-compatible API (LM Studio, etc.)"""
        url = f"{self.config.url.rstrip('/')}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.config.top_p
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not content:
            raise AgentError(self.ERR_LLM_ERROR, "LLM returned empty response")
        
        return content
    
    def _call_ollama(self, messages: list, temperature: float, max_tokens: int) -> str:
        """Call Ollama API."""
        url = self.config.url.replace('/v1', '').rstrip('/')
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(
            f"{url}/api/chat",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        content = data.get('message', {}).get('content', '')
        
        if not content:
            raise AgentError(self.ERR_LLM_ERROR, "LLM returned empty response")
        
        return content
    
    def _fail(self, code: str, message: str, details: dict = None):
        """Output failure JSON and exit."""
        output = {
            "status": "error",
            "error_code": code,
            "error": message
        }
        if details:
            output["details"] = details
        print(json.dumps(output, indent=2))
        sys.exit(1)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def clean_response(self, response: str) -> str:
        """Remove thinking tags, markdown fences, etc."""
        # Remove <think>...</think> tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove markdown code fences (but keep content)
        if response.strip().startswith('```'):
            response = re.sub(r'^```\w*\n?', '', response.strip())
            response = re.sub(r'\n?```$', '', response.strip())
        
        return response.strip()
    
    def extract_code_block(self, text: str, language: str = "python") -> Optional[str]:
        """Extract code from markdown code block."""
        pattern = rf'```{language}\s*(.*?)```'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try without language specifier
        pattern = r'```\s*(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    def extract_yaml_block(self, text: str) -> Optional[str]:
        """Extract YAML from markdown code block or raw text."""
        # Try markdown block first
        match = re.search(r'```ya?ml\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try to find raw YAML starting with common keys
        match = re.search(r'(program:\s*\n.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return text.strip()
    
    def get_required_field(self, data: dict, field: str, default=None) -> Any:
        """Get field from dict, raising error if missing and no default."""
        if field in data:
            return data[field]
        if default is not None:
            return default
        raise AgentError(self.ERR_MISSING_FIELD, f"Required field missing: {field}")


# =============================================================================
# SIMPLE AGENT IMPLEMENTATION HELPER
# =============================================================================

class SimpleAgent(AgentBase):
    """
    Simple agent that just passes system/user prompts to LLM.
    
    Use for agents that don't need custom processing logic.
    """
    
    def process(self, input_data: dict) -> dict:
        system_prompt = input_data.get("system_prompt", "You are a helpful assistant.")
        user_message = self.get_required_field(input_data, "user_message")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.call_llm(messages)
        cleaned = self.clean_response(response)
        
        return {
            "status": "success",
            "result": cleaned
        }


# =============================================================================
# IMPORT/EXPORT ANALYSIS UTILITIES
# =============================================================================

def extract_exports_from_code(code: str) -> List[Dict[str, str]]:
    """
    Extract exported names from Python code.
    
    Returns list of {"name": str, "type": "class"|"function"|"constant"}
    """
    import ast
    
    exports = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to regex if AST fails
        return _extract_exports_regex(code)
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith('_'):
                exports.append({"name": node.name, "type": "class"})
        
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if not node.name.startswith('_'):
                exports.append({"name": node.name, "type": "function"})
        
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Module-level constants (UPPER_CASE)
                    if name.isupper() or (name[0].isupper() and '_' in name):
                        exports.append({"name": name, "type": "constant"})
    
    return exports


def _extract_exports_regex(code: str) -> List[Dict[str, str]]:
    """Fallback regex-based export extraction."""
    exports = []
    
    for line in code.split('\n'):
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
        
        # Constants
        elif re.match(r'^[A-Z][A-Z_0-9]*\s*=', stripped):
            var_name = stripped.split('=')[0].strip()
            exports.append({'name': var_name, 'type': 'constant'})
    
    return exports


def extract_function_signatures(code: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract function/method signatures from code.
    
    Returns:
        {
            "function_name": {
                "args": ["arg1", "arg2"],
                "defaults": 2,  # number of args with defaults
                "is_method": False
            }
        }
    """
    import ast
    
    signatures = {}
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            arg_names = [a.arg for a in args.args]
            
            # Check if it's a method (first arg is 'self' or 'cls')
            is_method = len(arg_names) > 0 and arg_names[0] in ('self', 'cls')
            
            signatures[node.name] = {
                "args": arg_names,
                "defaults": len(args.defaults),
                "is_method": is_method,
                "required_args": len(arg_names) - len(args.defaults) - (1 if is_method else 0)
            }
    
    return signatures


def extract_class_info(code: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract class information including methods and their signatures.
    
    Returns:
        {
            "ClassName": {
                "bases": ["BaseClass"],
                "methods": {
                    "__init__": {"args": [...], ...},
                    "process": {"args": [...], ...}
                }
            }
        }
    """
    import ast
    
    classes = {}
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr)
            
            methods = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = item.args
                    arg_names = [a.arg for a in args.args]
                    is_method = len(arg_names) > 0 and arg_names[0] in ('self', 'cls')
                    
                    methods[item.name] = {
                        "args": arg_names,
                        "defaults": len(args.defaults),
                        "is_method": is_method,
                        "required_args": len(arg_names) - len(args.defaults) - (1 if is_method else 0)
                    }
            
            classes[node.name] = {
                "bases": bases,
                "methods": methods
            }
    
    return classes


def validate_imports_exist(code: str, available_exports: Dict[str, List[str]]) -> List[str]:
    """
    Validate that all imports in code exist in available exports.
    
    Args:
        code: Python code to check
        available_exports: {"module_name": ["Export1", "Export2"]}
        
    Returns:
        List of error messages for invalid imports
    """
    import ast
    
    errors = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["Syntax error in code"]
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            
            # Handle relative imports
            module = node.module.lstrip('.')
            
            # Check if this is an internal module
            if module in available_exports:
                valid_names = set(available_exports[module])
                for alias in node.names:
                    import_name = alias.name
                    if import_name != '*' and import_name not in valid_names:
                        errors.append(
                            f"Invalid import: '{import_name}' from {module} "
                            f"(available: {', '.join(sorted(valid_names)) or 'none'})"
                        )
    
    return errors


if __name__ == "__main__":
    # If run directly, act as simple pass-through agent
    SimpleAgent().run()
