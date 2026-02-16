#!/usr/bin/env python3
"""
Coder Agent for SwarmCoordinator
================================

Generates code based on system/user prompts.
Uses shared AgentBase infrastructure.
"""

import sys
import os
import re

# Add parent to path for agent_base import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agent_base import AgentBase, AgentError
except ImportError:
    # Fallback if agent_base not available
    from typing import Dict, List
    import json
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
                    'url': config.get("model_url", "http://localhost:1233/v1"),
                    'model': config.get("model_name", "local-model"),
                    'api_type': config.get("api_type", "openai"),
                    'timeout': config.get("timeout", 1200),
                    'temperature': config.get("temperature", 0.2),
                    'max_tokens': config.get("max_tokens", 6000)
                })()
                result = self.process(self.input_data)
                print(json.dumps(result))
            except Exception as e:
                print(json.dumps({"status": "error", "error": str(e)}))
                sys.exit(1)
        
        def call_llm(self, messages, temperature=None, max_tokens=None, retries=2):
            temp = temperature if temperature is not None else self.config.temperature
            tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            
            if self.config.api_type == 'ollama':
                url = self.config.url.replace('/v1', '').rstrip('/')
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temp, "num_predict": tokens}
                }
                response = requests.post(f"{url}/api/chat", json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return response.json().get('message', {}).get('content', '')
            else:
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": tokens
                }
                response = requests.post(
                    f"{self.config.url}/chat/completions",
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        
        def clean_response(self, response):
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            return response.strip()


class CoderAgent(AgentBase):
    """
    Code generation agent.
    
    Receives system/user prompts and generates code.
    Low temperature for consistent, correct output.
    """
    
    PLAN_REVIEW_PROMPT = """You are an expert programmer reviewing an architecture plan BEFORE you build it.
Your job is to catch problems that will cause issues during implementation.

DO NOT write any code. Output ONLY your review.

REVIEW FOR:
1. Are file responsibilities clear and non-overlapping?
2. Are dependencies between files reasonable? Any circular risks?
3. Are exports sufficient for the files that need them?
4. Are requirements specific enough to implement?
5. Is execution order correct?
6. Any missing files (utils, __init__.py, config files)?
7. Will the entry point wire everything together correctly?
8. Is this over-engineered for what's being asked?

OUTPUT FORMAT:
PLAN REVIEW:

LOOKS GOOD:
- [Things that are correct]

CONCERNS:
1. [Issue with explanation]

SUGGESTIONS:
1. [Improvement with rationale]

VERDICT: APPROVED | NEEDS_CHANGES"""

    BUILD_PROMPT = """You are an expert programmer building a complete project from a finalized plan.

The plan tells you WHAT to build. You decide HOW to implement it.

RULES:
1. Create ALL files listed in the plan
2. Each file MUST export what the plan specifies in its exports
3. IMPORTS ARE MANDATORY — copy them exactly from the plan's imports_from section. The import paths shown in the plan are the ONLY valid way to import project files. Do NOT invent your own import paths.
4. Implement ALL requirements listed for each file
5. NO stubs, NO placeholders (pass, ..., NotImplementedError)
6. Code must be immediately runnable
7. Include __init__.py files where needed for packages
8. Include requirements.txt if external packages are used

OUTPUT FORMAT:
### FILE: filename.py ###
<complete file content>

### FILE: another.py ###
<complete file content>

Output ONLY the file contents. No explanations before or after."""

    # Common libraries for requirements.txt auto-detection
    COMMON_LIBS = {
        "PyPDF2": "PyPDF2",
        "pypdf": "pypdf",
        "PIL": "Pillow",
        "requests": "requests",
        "pandas": "pandas",
        "numpy": "numpy",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "flask": "flask",
        "fastapi": "fastapi",
        "sqlalchemy": "sqlalchemy",
        "cv2": "opencv-python-headless",
        "openai": "openai",
        "yaml": "PyYAML",
        "pydantic": "pydantic"
    }
    
    def process(self, input_data: dict) -> dict:
        """Generate code from prompts."""

        mode = input_data.get("mode", "default")

        if mode == "plan_review":
            return self._process_plan_review(input_data)
        elif mode == "build":
            return self._process_build(input_data)
        else:
            return self._process_default(input_data)

    def _process_plan_review(self, input_data: dict) -> dict:
        """Review an architecture plan before building."""
        plan_yaml = input_data.get("plan_yaml") or input_data.get("user_message", "")
        if not plan_yaml:
            return {"status": "error", "error": "No plan_yaml provided for plan_review mode"}

        user_message = f"Review this architecture plan:\n\n```yaml\n{plan_yaml}\n```"

        messages = [
            {"role": "system", "content": self.PLAN_REVIEW_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.call_llm(messages, temperature=0.3, max_tokens=4000)
        cleaned = self.clean_response(response)

        return {
            "status": "success",
            "review": cleaned,
            "result": cleaned
        }

    def _process_build(self, input_data: dict) -> dict:
        """Build all code from a finalized plan."""
        plan_yaml = input_data.get("plan_yaml", "")
        job_spec = input_data.get("job_spec", "")
        environment_context = input_data.get("environment_context", "")

        if not plan_yaml:
            return {"status": "error", "error": "No plan_yaml provided for build mode"}

        user_message = f"FINALIZED PLAN:\n```yaml\n{plan_yaml}\n```"
        if job_spec:
            user_message += f"\n\nORIGINAL JOB SPECIFICATION:\n{job_spec}"
        if environment_context:
            user_message += f"\n\n{environment_context}"
        user_message += "\n\nBuild all files now:"

        messages = [
            {"role": "system", "content": self.BUILD_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.call_llm(messages, temperature=0.2)
        cleaned = self.clean_response(response)
        detected_deps = self._detect_dependencies(cleaned)

        return {
            "status": "success",
            "result": cleaned,
            "detected_dependencies": detected_deps
        }

    def _process_default(self, input_data: dict) -> dict:
        """Default mode — original behavior for backward compatibility."""
        system_prompt = input_data.get("system_prompt", "You are an expert Python programmer.")
        user_message = input_data.get("user_message", "")

        if not user_message:
            return {"status": "error", "error": "No user_message provided"}

        # Enhance system prompt with coding standards
        enhanced_system = system_prompt + """

CODING STANDARDS:
1. Use standard library whenever possible
2. Prefer 'pypdf' for PDFs, 'Pillow' for images
3. Write robust error handling
4. Output complete, executable code"""

        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": user_message}
        ]

        # Use low temperature for code generation
        response = self.call_llm(messages, temperature=0.2, max_tokens=8000)

        # Clean response
        cleaned = self.clean_response(response)

        # Detect dependencies for requirements.txt
        detected_deps = self._detect_dependencies(cleaned)

        return {
            "status": "success",
            "result": cleaned,
            "detected_dependencies": detected_deps
        }
    
    def _detect_dependencies(self, code: str) -> list:
        """Detect external dependencies from code."""
        found = set()
        
        for line in code.split('\n'):
            # import X
            match = re.match(r'^\s*import\s+(\w+)', line)
            if match:
                lib = match.group(1)
                if lib in self.COMMON_LIBS:
                    found.add(self.COMMON_LIBS[lib])
            
            # from X import
            match = re.match(r'^\s*from\s+(\w+)', line)
            if match:
                lib = match.group(1)
                if lib in self.COMMON_LIBS:
                    found.add(self.COMMON_LIBS[lib])
        
        return list(found)


if __name__ == "__main__":
    CoderAgent().run()
