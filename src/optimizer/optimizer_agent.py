### FILE: src/optimizer/optimizer_agent.py ###
#!/usr/bin/env python3
import sys
import json
import requests

def call_llm(config, messages):
    url = config.get("model_url", "http://localhost:1234/v1")
    model = config.get("model_name", "local-model")
    api_type = config.get("api_type", "openai")
    timeout = config.get("timeout", 600)
    
    try:
        if api_type == 'ollama':
            ollama_url = url.replace('/v1', '').rstrip('/')
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": config.get("temperature", 0.6),
                        "num_predict": config.get("max_tokens", 4000),
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')
        else:
            response = requests.post(
                f"{url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": config.get("temperature", 0.6),
                    "max_tokens": config.get("max_tokens", 4000),
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def main():
    try:
        input_data = json.load(sys.stdin)
        config = input_data.get("config", {})
        system_prompt = input_data.get("system_prompt", "You are a performance optimization expert.")
        user_message = input_data.get("user_message", "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = call_llm(config, messages)
        print(json.dumps({"status": "success", "result": response}))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()