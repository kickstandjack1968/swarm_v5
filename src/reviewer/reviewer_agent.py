#!/usr/bin/env python3
"""
Reviewer Agent for SwarmCoordinator
====================================

Code review and compliance checking.
Uses shared AgentBase infrastructure.
"""

import sys
import os
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agent_base import AgentBase
except ImportError:
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
                    'timeout': config.get("timeout", 600),
                    'temperature': config.get("temperature", 0.7),
                    'max_tokens': config.get("max_tokens", 3000)
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


class ReviewerAgent(AgentBase):
    """
    Code review agent.

    Modes:
    - default: Standard review (system_prompt + user_message passthrough)
    - compliance: Verify code implements the architecture plan
    """

    COMPLIANCE_PROMPT = """You are a code reviewer performing a COMPLIANCE REVIEW.
You have the architecture plan AND the generated code.

YOUR JOB: Verify the code correctly implements the plan.

CHECK EACH FILE:
1. Does each planned file exist in the output?
2. Does each file export what the plan specifies?
3. Does each file implement ALL requirements from the plan?
4. Do imports between files resolve correctly?
5. Does the entry point wire all components together?
6. Is there real implementation (no stubs, no placeholders)?
7. Will this code actually run without errors?

OUTPUT FORMAT:
COMPLIANCE REVIEW:

FILE-BY-FILE:
- filename.py: [PASS/FAIL] — [details]

IMPORT CHECK:
- [Any import issues found]

ISSUES REQUIRING REVISION:
1. [Specific issue in specific file with what needs to change]

OVERALL: STATUS: APPROVED | STATUS: NEEDS_REVISION"""

    FILE_COMPLIANCE_PROMPT = """You are a code reviewer performing a SINGLE-FILE COMPLIANCE REVIEW.
You will receive ONE file's code, its plan specification, and the code of its dependencies.

YOUR JOB: Verify this ONE file correctly implements its plan specification.

CHECK:
1. Does the file export what the plan specifies (correct class/function names)?
2. Does it implement ALL requirements from the plan spec?
3. Do method signatures match the plan (correct args, return types)?
4. Are imports correct (references to dependencies that exist)?
5. Is there real implementation (no stubs, no placeholders, no pass-only methods)?
6. Will this code actually work at runtime (correct attribute access, type usage)?
7. Does it use its dependencies correctly (right method names, right arg types)?

OUTPUT FORMAT (you MUST follow this exactly):

FILE: {filename}

REQUIREMENTS CHECK:
- Requirement 1: [MET/NOT MET] — [brief reason]
- Requirement 2: [MET/NOT MET] — [brief reason]

ISSUES:
1. [Specific bug or missing implementation — be precise with line references]
(or "None" if no issues)

STATUS: PASS
(or STATUS: FAIL if any requirement is NOT MET or any issue is critical)"""

    def process(self, input_data: dict) -> dict:
        """Process review request."""

        mode = input_data.get("mode", "default")

        if mode == "compliance":
            return self._process_compliance(input_data)
        elif mode == "compliance_file":
            return self._process_compliance_file(input_data)
        else:
            return self._process_default(input_data)

    def _process_compliance(self, input_data: dict) -> dict:
        """Compliance review — verify code implements the plan."""
        plan_yaml = input_data.get("plan_yaml", "")
        code = input_data.get("code", "")

        if not plan_yaml or not code:
            return {"status": "error", "error": "Compliance review requires both plan_yaml and code"}

        user_message = f"""ARCHITECTURE PLAN:
```yaml
{plan_yaml}
```

GENERATED CODE:
{code}

Perform a compliance review. Check every file against the plan."""

        messages = [
            {"role": "system", "content": self.COMPLIANCE_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.call_llm(messages, temperature=0.3, max_tokens=4000)
        cleaned = self.clean_response(response)

        return {
            "status": "success",
            "result": cleaned
        }

    def _process_compliance_file(self, input_data: dict) -> dict:
        """Single-file compliance review — verify one file against its plan spec."""
        file_name = input_data.get("file_name", "unknown.py")
        file_code = input_data.get("file_code", "")
        plan_spec = input_data.get("plan_spec", "")
        dep_code = input_data.get("dependency_code", "")
        job_spec = input_data.get("job_spec", "")

        if not file_code:
            return {"status": "error", "error": f"No code provided for {file_name}"}

        user_parts = [f"FILE TO REVIEW: {file_name}\n"]

        if plan_spec:
            user_parts.append(f"PLAN SPECIFICATION:\n{plan_spec}\n")

        user_parts.append(f"CODE:\n```python\n{file_code}\n```\n")

        if dep_code:
            user_parts.append(f"DEPENDENCY CODE (for reference):\n{dep_code}\n")

        if job_spec:
            user_parts.append(f"PROJECT CONTEXT:\n{job_spec[:2000]}\n")

        user_parts.append(f"Review {file_name} against its plan specification. End with STATUS: PASS or STATUS: FAIL.")

        user_message = "\n".join(user_parts)

        system_prompt = self.FILE_COMPLIANCE_PROMPT.replace("{filename}", file_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        max_tokens = input_data.get("config", {}).get("max_tokens", 4000)
        response = self.call_llm(messages, temperature=0.3, max_tokens=max_tokens)
        cleaned = self.clean_response(response)

        return {
            "status": "success",
            "result": cleaned
        }

    def _process_default(self, input_data: dict) -> dict:
        """Default review — system_prompt + user_message passthrough."""
        system_prompt = input_data.get("system_prompt", "You are a code reviewer.")
        user_message = input_data.get("user_message", "")

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


def main():
    """Shim for backward compatibility."""
    ReviewerAgent().run()


if __name__ == "__main__":
    main()
