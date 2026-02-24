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
    
    CLARIFY_PROMPT = """You are the Clarifier agent — the first stage in a multi-agent software development pipeline.
Your job is to analyze a raw user request and identify every ambiguity that would block implementation.

You do NOT design solutions. You do NOT write code. You do NOT make assumptions silently.
You identify gaps and ask concrete questions.

FIRST: Assess the request complexity.
- If the request is a trivial program (e.g., "Hello World", "fizzbuzz", "print 1 to 10"),
  respond with STATUS: CLEAR — the requirements are self-evident.
  Do NOT invent frameworks, web services, or infrastructure for simple scripts.
- Only ask questions when there are genuine ambiguities that would cause two developers
  to build meaningfully different things.

ANALYZE THE REQUEST FOR GAPS IN:
- INPUT: What data comes in? Format, source, volume, variability?
- OUTPUT: What is produced? Format, destination, schema?
- ERROR CASES: What can go wrong? What should happen when it does?
- SCALE: How much data, how often, how many users?
- EDGE CASES: Empty inputs, malformed data, boundary values, concurrent access?
- CONSTRAINTS: Language version, libraries, environment, performance requirements?

Do NOT ask about things already specified in the request.
Do NOT ask philosophical or open-ended questions.
Every question must be about a concrete implementation decision.

RESPONSE FORMAT:

QUESTIONS:
1. [BLOCKING] <specific, implementation-focused question>
   → Default if skipped: <what you will assume>
2. [IMPORTANT] <specific, implementation-focused question>
   → Default if skipped: <what you will assume>
3. [OPTIONAL] <specific, implementation-focused question>
   → Default if skipped: <what you will assume>

Priority definitions:
- BLOCKING: Cannot produce a correct spec without this answer.
- IMPORTANT: Answer significantly changes the design. Will proceed with stated default if skipped.
- OPTIONAL: Minor implementation detail. Will always use stated default if skipped.

ONLY if ALL requirements are crystal clear with no gaps:
STATUS: CLEAR - All requirements are well-defined."""

    SYNTHESIZE_PROMPT = """You are a Technical Project Manager.
Synthesize a comprehensive Job Scope from the original request and Q&A session.

OUTPUT RULES:
1. Return ONLY the synthesized Job Scope
2. No introductory text
3. Integrate answers into a unified specification
4. Prioritize user answers over original if contradictions"""

    SYNTHESIZE_STRUCTURED_PROMPT = """You are a Technical Project Manager producing a STRUCTURED JOB SPECIFICATION.

From the original request and Q&A session, synthesize a complete, unambiguous spec.

CRITICAL RULES:
- Match the spec complexity to the request complexity. A "Hello World" request gets a
  single-file script spec, not a web service. Do NOT add frameworks, servers, or
  infrastructure the user didn't ask for.
- ONLY preserve technology choices that appear in the ORIGINAL REQUEST. Do NOT invent
  or assume frameworks, libraries, or architectures that the user never mentioned.
- When answers say "no answers provided" or "proceeding with defaults", use the MINIMAL
  viable interpretation — the simplest program that satisfies the literal request.
- PRESERVE all technology choices EXPLICITLY STATED in the original request (libraries, frameworks, languages, protocols).
  If the request says "FastAPI", the spec says "FastAPI" — do NOT substitute or generalize.
- PRESERVE all specific schemas, data formats, and API designs from the original request.
- Every assumption you make (where the user didn't specify) must be documented explicitly.
- Acceptance criteria must be testable — each one is objectively pass or fail.
- out_of_scope must be populated — explicitly bounding scope prevents downstream drift.

Output ONLY the JSON below, no other text.

JOB_SPEC_JSON:
```json
{
  "title": "<short snake_case project title>",
  "description": "<complete description — specific enough that two developers would build the same thing>",
  "requirements": [
    "<functional requirement — preserve exact tech/library names from the original request>"
  ],
  "constraints": [
    "<hard constraint: language, libraries, environment, performance>",
    "<ASSUMPTION: what you assumed and why>"
  ],
  "input_format": "<all inputs: type, format, source, expected volume>",
  "output_format": "<all outputs: type, format, destination, schema>",
  "error_handling": "<how each error class should be handled>",
  "edge_cases": [
    "<edge case and required behavior>"
  ],
  "out_of_scope": [
    "<feature or behavior explicitly excluded>"
  ],
  "acceptance_criteria": [
    "<testable criterion — specific enough to pass or fail>"
  ],
  "open_questions": [
    {
      "question": "<unresolved ambiguity>",
      "impact": "<what it affects>",
      "assumed_resolution": "<what you assumed>"
    }
  ]
}
```"""
    
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

Synthesize these into a complete, unified Job Specification."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self.call_llm(messages)
        cleaned = self.clean_response(response)

        # Try to extract JSON from structured output
        job_spec_text = cleaned
        if output_format == "structured":
            job_spec_json = self._extract_job_spec_json(cleaned)
            if job_spec_json:
                # Store both the raw JSON and a readable version for the architect
                job_spec_text = self._format_job_spec_for_architect(job_spec_json)

        result = {
            "status": "success",
            "job_scope": job_spec_text
        }

        # Also return as job_spec key for collaborative workflow
        if output_format == "structured":
            result["job_spec"] = job_spec_text

        return result

    def _extract_job_spec_json(self, text: str) -> dict:
        """Try to extract JOB_SPEC_JSON from clarifier output."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*"title"[\s\S]*"requirements"[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _format_job_spec_for_architect(self, spec: dict) -> str:
        """Format JOB_SPEC_JSON into readable text the architect can use."""
        parts = []

        parts.append(f"## {spec.get('title', 'Untitled Project')}")
        parts.append(f"\n{spec.get('description', '')}")

        if spec.get('requirements'):
            parts.append("\n## REQUIREMENTS")
            for i, req in enumerate(spec['requirements'], 1):
                parts.append(f"{i}. {req}")

        if spec.get('constraints'):
            parts.append("\n## CONSTRAINTS")
            for c in spec['constraints']:
                parts.append(f"- {c}")

        if spec.get('input_format'):
            parts.append(f"\n## INPUT FORMAT\n{spec['input_format']}")

        if spec.get('output_format'):
            parts.append(f"\n## OUTPUT FORMAT\n{spec['output_format']}")

        if spec.get('error_handling'):
            parts.append(f"\n## ERROR HANDLING\n{spec['error_handling']}")

        if spec.get('edge_cases'):
            parts.append("\n## EDGE CASES")
            for ec in spec['edge_cases']:
                parts.append(f"- {ec}")

        if spec.get('out_of_scope'):
            parts.append("\n## OUT OF SCOPE")
            for oos in spec['out_of_scope']:
                parts.append(f"- {oos}")

        if spec.get('acceptance_criteria'):
            parts.append("\n## ACCEPTANCE CRITERIA")
            for i, ac in enumerate(spec['acceptance_criteria'], 1):
                parts.append(f"{i}. {ac}")

        if spec.get('open_questions'):
            parts.append("\n## OPEN QUESTIONS")
            for oq in spec['open_questions']:
                q = oq if isinstance(oq, str) else oq.get('question', '')
                assumed = oq.get('assumed_resolution', '') if isinstance(oq, dict) else ''
                parts.append(f"- {q}")
                if assumed:
                    parts.append(f"  → Assumed: {assumed}")

        return '\n'.join(parts)


if __name__ == "__main__":
    ClarifierAgent().run()
