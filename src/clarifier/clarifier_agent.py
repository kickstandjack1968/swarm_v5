#!/usr/bin/env python3
"""
Clarifier Agent for SwarmCoordinator v2
=======================================

External agent that handles requirements clarification.
Called by coordinator via subprocess with JSON payload on stdin.

Location: swarm_v4/src/clarifier/clarifier_agent.py

Input (JSON on stdin):
{
    "config": {
        "model_url": "http://localhost:1234/v1",
        "model_name": "model-name",
        "api_type": "openai",
        "temperature": 0.7,
        "max_tokens": 2000,
        "timeout": 300
    },
    "system_prompt": "You are a Requirements Clarification Agent...",
    "user_message": "Analyze these requirements: ..."
}

Output (JSON on stdout):
{
    "status": "success",
    "result": "CLARIFYING QUESTIONS:\n1. ...\n2. ...",
    "metadata": {
        "questions_count": 5,
        "status": "NEEDS_CLARIFICATION" | "CLEAR"
    }
}
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
    
    Returns:
        dict with questions_count and status
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
        # Check for bullet points or other question indicators
        bullet_pattern = r'^\s*[-•\*]\s+'
        bullets = re.findall(bullet_pattern, response, re.MULTILINE)
        if bullets:
            metadata["questions_count"] = len(bullets)
            metadata["status"] = "NEEDS_CLARIFICATION"
        else:
            metadata["status"] = "UNCLEAR"
    
    return metadata


def clean_response(response: str) -> str:
    """Clean up the LLM response, removing thinking tags and extra whitespace."""
    # Remove <think>...</think> tags if present (some models use this)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove markdown code blocks if the response is wrapped in them
    if response.strip().startswith('```'):
        response = re.sub(r'^```\w*\n?', '', response.strip())
        response = re.sub(r'\n?```$', '', response.strip())
    
    return response.strip()


def main():
    """Main entry point - reads JSON from stdin, outputs JSON to stdout."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        config = input_data.get("config", {})
        
        # Get prompts
        system_prompt = input_data.get("system_prompt", "")
        if not system_prompt:
            system_prompt = """You are a Requirements Clarification Agent. Your ONLY job is to ask questions.

CRITICAL INSTRUCTIONS:
1. Ask clarifying questions to fully understand the requirements
   - Simple requests may need only 1-2 questions
   - Complex multi-component systems may need 8-10 questions
   - Don't pad with unnecessary questions, don't skip important ones
2. You MUST NOT provide solutions, code, or implementation details
3. You MUST NOT say the requirements are clear unless they truly are
4. Format your response EXACTLY as shown below

FOCUS YOUR QUESTIONS ON:
- Input/output formats and data types
- Error handling expectations (what happens when X fails?)
- Edge cases the user cares about
- External dependencies or environment constraints
- Performance or scale requirements (if relevant)

AVOID ASKING ABOUT:
- Future features not mentioned
- Multiple language support unless relevant
- Hypothetical scenarios unlikely to occur

RESPONSE FORMAT (use this exact format):
CLARIFYING QUESTIONS:
1. [Specific question about missing requirement]
2. [Specific question about unclear specification]
3. [Additional questions as needed]

ONLY if ALL requirements are crystal clear and complete, respond with:
STATUS: CLEAR - All requirements are well-defined.

Remember: Your job is to ASK QUESTIONS, not solve problems."""
        
        # Accept either user_message or user_request (coordinator sends user_request)
        user_request = input_data.get("user_message", "") or input_data.get("user_request", "")
        
        if not user_request:
            raise ValueError("No user_message or user_request provided")
        
        # Wrap the request with context so LLM knows this is a BUILD request
        user_message = f"""The user wants to BUILD the following:

{user_request}

Analyze these requirements and ask clarifying questions to fully understand what needs to be built."""
        
        # Build messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call LLM
        raw_response = call_llm(config, messages)
        
        # Clean and parse response
        cleaned_response = clean_response(raw_response)
        metadata = parse_clarifier_response(cleaned_response)
        
        # Build output in format coordinator expects
        if metadata["status"] == "CLEAR":
            # Coordinator checks for status == "clear"
            print(json.dumps({
                "status": "clear",
                "result": cleaned_response,
                "questions": []
            }))
        else:
            # Extract individual questions from response
            questions = []
            for line in cleaned_response.split('\n'):
                # Match numbered questions: "1. Question text" or "1) Question text"
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
