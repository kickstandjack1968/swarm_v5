#!/usr/bin/env python3
"""
Clarifier Agent for SwarmCoordinator
====================================

Handles requirements clarification and synthesis.
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
                    'timeout': config.get("timeout", 300),
                    'temperature': config.get("temperature", 0.7),
                    'max_tokens': config.get("max_tokens", 2000)
                })()
                result = self.process(self.input_data)
                print(json.dumps(result))
            except Exception as e:
                print(json.dumps({"status": "error", "error": str(e)}))
                sys.exit(1)
        
        def call_llm(self, messages, temperature=None, max_tokens=None, retries=2):
            temp = temperature if temperature is not None else self.config.temperature
            tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            
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


class ClarifierAgent(AgentBase):
    """
    Requirements clarification agent.
    
    Two modes:
    - 'clarify': Ask questions about requirements
    - 'synthesize': Combine Q&A into job scope
    """
    
    CLARIFY_PROMPT = """You are a Requirements Clarification Agent. Your ONLY job is to ask questions.

CRITICAL INSTRUCTIONS:
1. Ask clarifying questions to fully understand the requirements
2. Simple requests: 1-2 questions
3. Complex systems: 5-10 questions
4. Do NOT provide solutions or code
5. Do NOT say requirements are clear unless they truly are

RESPONSE FORMAT:
CLARIFYING QUESTIONS:
1. [Question about missing requirement]
2. [Question about unclear specification]

ONLY if ALL requirements are crystal clear:
STATUS: CLEAR - All requirements are well-defined."""

    SYNTHESIZE_PROMPT = """You are a Technical Project Manager.
Synthesize a comprehensive Job Scope from the original request and Q&A session.

OUTPUT RULES:
1. Return ONLY the synthesized Job Scope
2. No introductory text
3. Integrate answers into a unified specification
4. Prioritize user answers over original if contradictions"""

    SYNTHESIZE_STRUCTURED_PROMPT = """You are a Technical Project Manager producing a STRUCTURED JOB SPECIFICATION.

From the original request and Q&A session, produce a document with EXACTLY these sections:

## PROJECT OVERVIEW
One paragraph describing what this software does.

## FUNCTIONAL REQUIREMENTS
Numbered list of specific behaviors the software must exhibit.

## INPUT/OUTPUT SPECIFICATION
- What inputs does the program accept (files, CLI args, stdin)?
- What outputs does it produce (files, stdout, return values)?
- What formats are used?

## ERROR HANDLING
- What errors should be handled?
- What should happen when they occur?

## CONSTRAINTS
- External libraries allowed/needed
- Environment requirements
- Performance expectations (if any)

## ACCEPTANCE CRITERIA
Numbered list of testable criteria that define "done".

Do NOT include implementation details (no class names, no file structure, no import paths).
Output ONLY the structured job specification."""
    
    def process(self, input_data: dict) -> dict:
        """Process clarification or synthesis request."""
        
        mode = input_data.get("mode", "clarify")
        user_request = input_data.get("user_message") or input_data.get("user_request", "")
        
        if not user_request:
            return {"status": "error", "error": "No user request provided"}
        
        if mode == "synthesize":
            return self._synthesize(input_data, user_request)
        else:
            return self._clarify(input_data, user_request)
    
    def _clarify(self, input_data: dict, user_request: str) -> dict:
        """Ask clarifying questions."""
        
        system_prompt = input_data.get("system_prompt") or self.CLARIFY_PROMPT
        
        user_message = f"""The user wants to BUILD the following:

{user_request}

Analyze these requirements and ask clarifying questions to fully understand what needs to be built."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.call_llm(messages)
        cleaned = self.clean_response(response)
        
        # Check if requirements are clear
        if "STATUS: CLEAR" in cleaned.upper():
            return {
                "status": "clear",
                "result": cleaned,
                "questions": []
            }
        
        # Extract questions
        questions = []
        for line in cleaned.split('\n'):
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
            if match:
                questions.append(match.group(1).strip())
        
        return {
            "status": "needs_clarification",
            "result": cleaned,
            "questions": questions,
            "questions_count": len(questions)
        }
    
    def _synthesize(self, input_data: dict, user_request: str) -> dict:
        """Synthesize Q&A into job scope."""

        questions = input_data.get("questions", "")
        answers = input_data.get("answers", "")
        output_format = input_data.get("output_format", "")

        # Choose prompt based on output format
        if output_format == "structured":
            system_prompt = self.SYNTHESIZE_STRUCTURED_PROMPT
        else:
            system_prompt = self.SYNTHESIZE_PROMPT

        user_message = f"""ORIGINAL REQUEST:
{user_request}

CLARIFICATION QUESTIONS:
{questions}

USER ANSWERS:
{answers}

Synthesize these into a complete, unified Job Scope."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.call_llm(messages)
        cleaned = self.clean_response(response)

        result = {
            "status": "success",
            "job_scope": cleaned
        }

        # Also return as job_spec key for collaborative workflow
        if output_format == "structured":
            result["job_spec"] = cleaned

        return result


if __name__ == "__main__":
    ClarifierAgent().run()
