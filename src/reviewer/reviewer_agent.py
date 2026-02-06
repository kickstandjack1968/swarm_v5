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
You have the architecture plan, the original job specification, AND the generated code.

YOUR JOB: Verify the code correctly implements the plan AND satisfies the job specification.

CHECK EACH FILE:
1. Does each planned file exist in the output?
2. Does each file export what the plan specifies?
3. Does each file implement ALL requirements from the plan?
4. Do imports between files resolve correctly?
5. Does the entry point wire all components together?
6. Is there real implementation (no stubs, no placeholders, no mock/fake data)?
7. Will this code actually run without errors?
8. Does the code meet the acceptance criteria from the job specification?
9. Are all functional requirements from the job spec addressed?

OUTPUT FORMAT:
COMPLIANCE REVIEW:

FILE-BY-FILE:
- filename.py: [PASS/FAIL] — [details]

IMPORT CHECK:
- [Any import issues found]

JOB SPEC COMPLIANCE:
- [Check each acceptance criterion and requirement from the job spec]

ISSUES REQUIRING REVISION:
1. [Specific issue in specific file with what needs to change]

OVERALL: STATUS: APPROVED | STATUS: NEEDS_REVISION"""

    def process(self, input_data: dict) -> dict:
        """Process review request."""

        mode = input_data.get("mode", "default")

        if mode == "compliance":
            return self._process_compliance(input_data)
        else:
            return self._process_default(input_data)

    def _process_compliance(self, input_data: dict) -> dict:
        """Compliance review — verify code implements the plan and job spec."""
        plan_yaml = input_data.get("plan_yaml", "")
        code = input_data.get("code", "")
        job_spec = input_data.get("job_spec", "")

        if not plan_yaml or not code:
            return {"status": "error", "error": "Compliance review requires both plan_yaml and code"}

        user_message = f"""ARCHITECTURE PLAN:
```yaml
{plan_yaml}
```

GENERATED CODE:
{code}"""

        if job_spec:
            user_message += f"""

ORIGINAL JOB SPECIFICATION:
{job_spec}"""

        user_message += """

Perform a compliance review. Check every file against the plan and the job specification."""

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
