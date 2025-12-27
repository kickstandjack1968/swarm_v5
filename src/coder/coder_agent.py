#!/usr/bin/env python3
import sys
import json
import requests
import re
import os

def call_llm(config, messages):
    """Make API call to LLM"""
    url = config.get("model_url", "http://localhost:1234/v1")
    model = config.get("model_name", "local-model")
    api_type = config.get("api_type", "openai")
    timeout = config.get("timeout", 1200)  # High timeout for coding tasks
    
    try:
        # Lower temp for code generation to avoid hallucinations
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": config.get("temperature", 0.2), # Enforce low temp for code
            "max_tokens": config.get("max_tokens", 6000)
        }
        
        # OLLAMA HANDLING
        if api_type == 'ollama':
            ollama_url = url.replace('/v1', '').rstrip('/')
            # Ollama uses 'options' for params
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
        
    except Exception as e:
        return f"# Error generating code: {str(e)}"

def extract_code(text):
    """
    Extracts the main code block from markdown responses.
    """
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no markdown blocks, return the whole text (fallback)
    return text.strip()

def update_requirements(code, project_root="."):
    """
    Scans code for common non-standard imports and updates requirements.txt
    """
    # Common libraries map: "Import Name" -> "Pip Package Name"
    common_libs = {
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
        "yaml": "PyYAML"
    }
    
    found_packages = set()
    
    # Scan code line by line
    for line in code.split('\n'):
        # Check 'import module'
        match_import = re.match(r'^\s*import\s+(\w+)', line)
        if match_import:
            lib = match_import.group(1)
            if lib in common_libs:
                found_packages.add(common_libs[lib])
                
        # Check 'from module import'
        match_from = re.match(r'^\s*from\s+(\w+)', line)
        if match_from:
            lib = match_from.group(1)
            if lib in common_libs:
                found_packages.add(common_libs[lib])

    if not found_packages:
        return

    # Update requirements.txt
    req_path = os.path.join(project_root, "requirements.txt")
    existing_reqs = set()
    
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            existing_reqs = set(line.strip() for line in f if line.strip())
            
    # Add new ones
    new_reqs = existing_reqs.union(found_packages)
    
    if new_reqs != existing_reqs:
        # We print to stderr so it doesn't break the JSON output
        print(f"ðŸ“¦ Auto-adding dependencies to requirements.txt: {found_packages}", file=sys.stderr)
        with open(req_path, 'w') as f:
            f.write('\n'.join(sorted(new_reqs)))

def main():
    try:
        input_data = json.load(sys.stdin)
        config = input_data.get("config", {})
        
        # 1. Inject "Production Standards" into the prompt
        system_prompt = input_data.get("system_prompt", "You are an expert Python coder.")
        system_prompt += (
            "\nSTANDARDS:\n"
            "1. Use standard libraries whenever possible.\n"
            "2. If you MUST use external tools, prefer 'pypdf' for PDFs and 'Pillow' for images.\n"
            "3. Write robust, error-handling code."
        )
        
        user_message = input_data.get("user_message", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 2. Generate Code
        raw_response = call_llm(config, messages)
        final_code = extract_code(raw_response)
        
        # 3. AUTO-MANAGE DEPENDENCIES
        # We assume the script is running in src/coder/ or similar.
        # We try to find the project root (one level up from src) to save requirements.txt
        # Adjust logic: if we are in 'src', go up one level.
        current_dir = os.getcwd()
        project_root = current_dir
        # If the current dir ends in 'src', the project root is likely the parent
        if os.path.basename(current_dir) == "src":
             project_root = os.path.dirname(current_dir)
        # If the script is executed from the project root (usual case), we just use current_dir.
             
        update_requirements(final_code, project_root)
        
        # 4. Return Clean Result
        print(json.dumps({
            "status": "success",
            "result": final_code
        }))
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()