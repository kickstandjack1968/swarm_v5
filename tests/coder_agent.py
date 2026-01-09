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
                    'url': config.get("model_url", "http://localhost:1234/v1"),
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
    
    def _detect_dependencies(self, code: str) -> List[str]:
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
