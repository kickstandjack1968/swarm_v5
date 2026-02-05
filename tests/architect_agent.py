#!/usr/bin/env python3
"""
Architect Agent for SwarmCoordinator
====================================

Creates YAML execution plans with strict export/import specifications.
Uses shared AgentBase infrastructure.
"""

import sys
import os
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agent_base import AgentBase, AgentError
    import yaml
except ImportError:
    import yaml
    import requests
    
    class AgentBase:
        def __init__(self):
            self.config = None
            self.input_data = {}
        
        def run(self):
            try:
                self.input_data = json.load(sys.stdin)
                config = self.input_data.get("config", {})
                self.config = type('Config', (), {
                    'url': config.get("model_url") or config.get("url", "http://localhost:1234/v1"),
                    'model': config.get("model_name") or config.get("model", "local-model"),
                    'api_type': config.get("api_type", "openai"),
                    'timeout': config.get("timeout", 600),
                    'temperature': config.get("temperature", 0.6),
                    'max_tokens': config.get("max_tokens", 8000),
                    'top_p': config.get("top_p", 0.9)
                })()
                result = self.process(self.input_data)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(json.dumps({"status": "error", "ok": False, "error": str(e)}))
                sys.exit(1)
        
        def call_llm(self, messages, temperature=None, max_tokens=None, retries=2):
            temp = temperature if temperature is not None else self.config.temperature
            tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": self.config.top_p
            }
            response = requests.post(
                f"{self.config.url.rstrip('/')}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')


# Forbidden filename
FORBIDDEN_FILENAME = "config_v2.json"


class ArchitectAgent(AgentBase):
    """
    Architecture planning agent.
    
    Creates YAML plans with:
    - Explicit export specifications (names and types)
    - Import mappings between files
    - Dependency graphs
    - Execution order
    """
    
    SYSTEM_PROMPT = """You are an expert software architect creating a YAML execution plan.

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
    imports_from: {}
    requirements:
      - "Load config from file or environment"

  - name: "core.py"
    purpose: "Core business logic"
    dependencies: ["config.py"]
    exports:
      - name: "Processor"
        type: "class"
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
      - "Initialize and run"

execution_order: ["config.py", "core.py", "main.py"]
```

CRITICAL RULES:
1. exports MUST list what each file defines (exact names!)
2. imports_from MUST only reference names in source file's exports
3. dependencies MUST include every file used in imports_from
4. execution_order MUST have dependencies before dependents

Keep it SIMPLE. Output ONLY the YAML."""
    
    def process(self, input_data: dict) -> dict:
        """Generate YAML execution plan."""
        
        # Get user request
        user_request = input_data.get("user_request") or input_data.get("user_message", "")
        if not user_request:
            return {"status": "error", "ok": False, "error": "No user_request provided"}
        
        clarification = input_data.get("clarification", "")
        
        # Use custom system prompt if provided
        system_prompt = input_data.get("system_prompt") or self.SYSTEM_PROMPT
        
        # Build user prompt
        user_prompt = f"Create a YAML execution plan for:\n\n{user_request}"
        if clarification:
            user_prompt += f"\n\nCLARIFICATION:\n{clarification}"
        user_prompt += "\n\nOutput the YAML plan:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        response = self.call_llm(messages, temperature=0.6)
        
        # Extract YAML
        yaml_str = self._extract_yaml(response)
        if not yaml_str:
            return {
                "status": "error",
                "ok": False,
                "error": "Could not extract YAML from response",
                "raw_response": response
            }
        
        # Check forbidden filename
        if FORBIDDEN_FILENAME in yaml_str:
            return {
                "status": "error",
                "ok": False,
                "error": f"Plan contains forbidden file: {FORBIDDEN_FILENAME}"
            }
        
        # Parse and validate
        try:
            plan = yaml.safe_load(yaml_str)
            self._validate_plan(plan)
            
            # Normalize
            normalized = yaml.dump(plan, default_flow_style=False, sort_keys=False)
            
            return {
                "status": "success",
                "ok": True,
                "result": normalized,
                "plan_yaml": normalized,
                "raw_response": response
            }
            
        except yaml.YAMLError as e:
            return {
                "status": "error",
                "ok": False,
                "error": f"YAML parse error: {e}",
                "extracted_yaml": yaml_str
            }
        except ValueError as e:
            return {
                "status": "error",
                "ok": False,
                "error": f"Validation error: {e}",
                "extracted_yaml": yaml_str
            }
    
    def _extract_yaml(self, response: str) -> str:
        """Extract YAML from response."""
        # Try markdown block
        match = re.search(r'```ya?ml\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try raw YAML
        match = re.search(r'(program:\s*\n.*)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return whole thing if it starts with program:
        if response.strip().startswith('program:'):
            return response.strip()
        
        return ""
    
    def _validate_plan(self, plan: dict):
        """Validate plan structure and consistency."""
        
        # Required sections
        if 'program' not in plan:
            raise ValueError("Missing 'program' section")
        if 'files' not in plan or not plan['files']:
            raise ValueError("Missing or empty 'files' section")
        if 'execution_order' not in plan or not plan['execution_order']:
            raise ValueError("Missing or empty 'execution_order' section")
        
        # Normalize program
        if isinstance(plan['program'], str):
            plan['program'] = {'name': plan['program'], 'description': plan['program'], 'type': 'cli'}
        
        # Normalize architecture
        if 'architecture' not in plan:
            plan['architecture'] = {'pattern': 'simple', 'entry_point': 'main.py'}
        elif isinstance(plan['architecture'], str):
            plan['architecture'] = {'pattern': plan['architecture'], 'entry_point': 'main.py'}
        
        # Build file lookup
        file_names = set()
        file_exports = {}  # filename -> set of export names
        
        for f in plan['files']:
            if not isinstance(f, dict) or 'name' not in f:
                raise ValueError("Invalid file entry")
            
            name = f['name']
            file_names.add(name)
            
            # Extract exports
            exports = set()
            for exp in f.get('exports', []):
                if isinstance(exp, dict):
                    exports.add(exp.get('name', ''))
                elif isinstance(exp, str):
                    exports.add(exp)
            file_exports[name] = exports
            
            # Ensure required fields with defaults
            f.setdefault('purpose', '')
            f.setdefault('dependencies', [])
            f.setdefault('exports', [])
            f.setdefault('imports_from', {})
            f.setdefault('requirements', [])
        
        # Validate cross-references
        for f in plan['files']:
            name = f['name']
            
            # Check dependencies exist
            for dep in f['dependencies']:
                if dep not in file_names:
                    raise ValueError(f"File '{name}' depends on unknown file '{dep}'")
            
            # Check imports_from references valid files and exports
            for source, imports in f.get('imports_from', {}).items():
                if source not in file_names:
                    raise ValueError(f"File '{name}' imports from unknown file '{source}'")
                
                # Check that imports exist in source exports
                source_exports = file_exports.get(source, set())
                for imp in imports:
                    if imp not in source_exports:
                        # Warning only - might be auto-generated
                        pass
                
                # Ensure source is in dependencies
                if source not in f['dependencies']:
                    f['dependencies'].append(source)
        
        # Validate execution order
        for item in plan['execution_order']:
            if item not in file_names:
                raise ValueError(f"execution_order contains unknown file '{item}'")


if __name__ == "__main__":
    ArchitectAgent().run()
