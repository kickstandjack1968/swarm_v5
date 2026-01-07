#!/usr/bin/env python3
"""
Clarifier Agent for SwarmCoordinator v2 - FIXED
===============================================

External agent that handles requirements clarification AND synthesis.
Called by coordinator via subprocess with JSON payload on stdin.

Location: swarm_v4/src/clarifier/clarifier_agent.py
"""

import sys
import json
import requests
import re


def call_llm(config: dict, messages: list) -> str:
    """Make API call to LLM endpoint."""
    url = config.get("model_url", "http://localhost:1234/v1")
    model = config.get("model_name", "local-model")
    api_type = config.get("api_type", "openai")
    timeout = config.get("timeout", 300)
    
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 2000)
        }
        
        # OLLAMA HANDLING
        if api_type == 'ollama':
            ollama_url = url.replace('/v1', '').rstrip('/')
            payload["options"] = {
                "temperature": payload["temperature"],
                "num_predict": payload["max_tokens"]
            }
            del payload["temperature"]
            del payload["max_tokens"]
            
            response = requests.post(
                f"{ollama_url}/api/chat",
                json=payload,
                timeout=timeout
            )
        # OPENAI/LM STUDIO HANDLING
        else:
            response = requests.post(
                f"{url}/chat/completions",
                json=payload,
                timeout=timeout
            )
            
        response.raise_for_status()
        
        # Extract content
        data = response.json()
        if api_type == 'ollama':
            return data.get('message', {}).get('content', '')
        return data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
    except requests.exceptions.Timeout:
        raise Exception(f"LLM request timed out after {timeout}s")
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM request failed: {str(e)}")


def parse_clarifier_response(response: str) -> dict:
    """
    Parse the clarifier response to extract metadata.
    """
    metadata = {
        "questions_count": 0,
        "status": "UNKNOWN"
    }
    
    # Check if requirements are clear
    if "STATUS: CLEAR" in response.upper():
        metadata["status"] = "CLEAR"
        metadata["questions_count"] = 0
        return metadata
    
    # Count numbered questions (1. 2. 3. etc)
    question_pattern = r'^\s*\d+\.\s+'
    questions = re.findall(question_pattern, response, re.MULTILINE)
    metadata["questions_count"] = len(questions)
    
    if metadata["questions_count"] > 0:
        metadata["status"] = "NEEDS_CLARIFICATION"
    else:
        # Check for bullet points
        bullet_pattern = r'^\s*[-•\*]\s+'
        bullets = re.findall(bullet_pattern, response, re.MULTILINE)
        if bullets:
            metadata["questions_count"] = len(bullets)
            metadata["status"] = "NEEDS_CLARIFICATION"
        else:
            metadata["status"] = "UNCLEAR"
    
    return metadata


def clean_response(response: str) -> str:
    """Clean up the LLM response."""
    # Remove <think>...</think> tags if present
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove markdown code blocks
    if response.strip().startswith('```'):
        response = re.sub(r'^```\w*\n?', '', response.strip())
        response = re.sub(r'\n?```$', '', response.strip())
    
    return response.strip()


def handle_synthesis(input_data, config, user_request):
    """
    Handle the 'synthesize' mode: Turn Q&A into a final Job Scope.
    """
    questions = input_data.get("questions", "No questions provided.")
    answers = input_data.get("answers", "No answers provided.")
    
    system_prompt = """You are a Technical Project Manager.
Your job is to synthesize a comprehensive "Job Scope" based on an Original Request and a Q&A clarification session.

OUTPUT RULES:
1. Return ONLY the synthesized Job Scope description.
2. Do not include introductory text like "Here is the scope".
3. Integrate the user's answers into the original request to create a single, unified specification.
4. If there are contradictions, prioritize the User Answers.
"""

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

    raw_response = call_llm(config, messages)
    cleaned_response = clean_response(raw_response)

    # Output specific format expected by Coordinator for synthesis
    print(json.dumps({
        "status": "success",
        "job_scope": cleaned_response
    }))


def main():
    """Main entry point."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        config = input_data.get("config", {})
        
        # Accept either user_message or user_request
        user_request = input_data.get("user_message", "") or input_data.get("user_request", "")
        if not user_request:
            raise ValueError("No user_message or user_request provided")

        # --- MODE SWITCHING ---
        mode = input_data.get("mode", "clarify")
        
        if mode == "synthesize":
            handle_synthesis(input_data, config, user_request)
            return
        # ----------------------

        # Default: Clarification Mode
        system_prompt = input_data.get("system_prompt", "")
        if not system_prompt:
            system_prompt = """You are a Requirements Clarification Agent. Your ONLY job is to ask questions.

CRITICAL INSTRUCTIONS:
1. Ask clarifying questions to fully understand the requirements
   - Simple requests may need only 1-2 questions
   - Complex multi-component systems may need 8-10 questions
2. You MUST NOT provide solutions, code, or implementation details
3. You MUST NOT say the requirements are clear unless they truly are
4. Format your response EXACTLY as shown below

RESPONSE FORMAT (use this exact format):
CLARIFYING QUESTIONS:
1. [Specific question about missing requirement]
2. [Specific question about unclear specification]

ONLY if ALL requirements are crystal clear and complete, respond with:
STATUS: CLEAR - All requirements are well-defined.
"""
        
        user_message = f"""The user wants to BUILD the following:

{user_request}

Analyze these requirements and ask clarifying questions to fully understand what needs to be built."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        raw_response = call_llm(config, messages)
        cleaned_response = clean_response(raw_response)
        metadata = parse_clarifier_response(cleaned_response)
        
        if metadata["status"] == "CLEAR":
            print(json.dumps({
                "status": "clear",
                "result": cleaned_response,
                "questions": []
            }))
        else:
            # Extract individual questions
            questions = []
            for line in cleaned_response.split('\n'):
                match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
                if match:
                    questions.append(match.group(1).strip())
            
            print(json.dumps({
                "status": "needs_clarification",
                "result": cleaned_response,
                "questions": questions,
                "metadata": metadata
            }))
        
    except json.JSONDecodeError as e:
        print(json.dumps({
            "status": "error",
            "error": f"Invalid JSON input: {str(e)}"
        }))
        sys.exit(1)
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()