import sys
import json
import time
import re
import logging

# library imports (standard dependencies for this project)
try:
    import requests
    import yaml
except ImportError as e:
    # Fallback error if environment is not set up, though prompt implies availability
    json.dump({
        "ok": False,
        "error_code": "INTERNAL_ERROR",
        "error": f"Missing dependency: {str(e)}",
        "details": {"hint": "pip install requests pyyaml"}
    }, sys.stdout)
    sys.exit(1)

# Configure logging to STDERR only
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_INPUT_SIZE = 1024 * 1024  # 1MB
FORBIDDEN_FILENAME = "config_v2.json"

# Error Codes
ERR_INVALID_JSON_INPUT = "INVALID_JSON_INPUT"
ERR_INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
ERR_MISSING_FIELD = "MISSING_REQUIRED_FIELD"
ERR_INVALID_MODEL = "INVALID_MODEL_CONFIG"
ERR_MODEL_UNREACHABLE = "MODEL_UNREACHABLE"
ERR_MODEL_TIMEOUT = "MODEL_TIMEOUT"
ERR_MODEL_HTTP = "MODEL_HTTP_ERROR"
ERR_MODEL_BAD_RESP = "MODEL_BAD_RESPONSE"
ERR_EMPTY_RESP = "EMPTY_LLM_RESPONSE"
ERR_YAML_MISSING = "YAML_MISSING"
ERR_YAML_PARSE = "YAML_PARSE_FAILED"
ERR_INTERNAL = "INTERNAL_ERROR"

def fail(code, message, details=None):
    """Writes failure JSON to stdout and exits."""
    output = {
        "ok": False,
        "error_code": code,
        "error": message
    }
    if details:
        output["details"] = details
    print(json.dumps(output, indent=2))
    sys.exit(1)

def success(raw_response, plan_yaml):
    """Writes success JSON to stdout and exits."""
    output = {
        "ok": True,
        "status": "success",
        "result": plan_yaml,  # Coordinator expects 'result'
        "raw_response": raw_response,
        "plan_yaml": plan_yaml
    }
    print(json.dumps(output, indent=2))
    sys.exit(0)

def validate_model_config(cfg):
    """Validates the model configuration block."""
    required = ["url", "model", "api_type", "timeout", "temperature", "max_tokens", "top_p"]
    for field in required:
        if field not in cfg:
            fail(ERR_MISSING_FIELD, f"Model config missing field: {field}")
    
    if cfg["api_type"] != "openai":
        fail(ERR_INVALID_MODEL, "Only 'api_type': 'openai' is supported.")
    
    if not isinstance(cfg["timeout"], int) or cfg["timeout"] <= 0:
        fail(ERR_INVALID_MODEL, "Timeout must be a positive integer.")
    
    if not (0.0 <= cfg["temperature"] <= 2.0):
        fail(ERR_INVALID_MODEL, "Temperature must be between 0.0 and 2.0.")
        
    if not (0.0 < cfg["top_p"] <= 1.0):
        fail(ERR_INVALID_MODEL, "Top_p must be between 0.0 (exclusive) and 1.0.")

def generate_system_prompt():
    """Generates the system prompt enforcing the specific schema."""
    return f"""You are a Software Architect. You must output a software architecture plan in strict YAML format.

RULES:
1. Return ONLY YAML. No markdown formatting, no introductory text, no explanations.
2. The YAML must strictly adhere to the schema below.
3. Do NOT include any file named "{FORBIDDEN_FILENAME}" in the plan.

REQUIRED YAML SCHEMA:
```yaml
program:
  name: "project_name"
  description: "What the program does"
  type: "cli|subprocess_tool|library|service"
architecture:
  pattern: "simple|layered|modular"
  entry_point: "filename.py or null for libraries"

files:
  - name: "config.py"
    purpose: "What this file does"
    dependencies: []
    exports:
      - name: "Settings"
        type: "class"
    requirements:
      - "Specific requirement 1"
  
  - name: "main.py"
    purpose: "Entry point"
    dependencies: ["config.py"]
    imports_from:
      config.py: ["Settings"]
    exports: []
    requirements:
      - "Run the application"

execution_order: ["config.py", "main.py"]
```

CONSTRAINTS:
- 'program' must be a dict with 'name', 'description', and 'type' keys.
- 'architecture' must be a dict with 'pattern' and 'entry_point' keys.
- 'files' must be a non-empty list of file specifications.
- Each file must have 'name', 'purpose', and 'requirements' (list).
- 'dependencies', 'exports', 'imports_from' are optional but recommended.
- 'execution_order' must be a non-empty list of filenames.
- Every filename in 'execution_order' MUST exist in the 'files' list.
- Every dependency listed must exist in the 'files' list.
"""

def generate_user_prompt(request, clarification):
    prompt = f"USER REQUEST:\n{request}\n"
    if clarification:
        prompt += f"\nCLARIFICATION:\n{clarification}\n"
    prompt += "\nOUTPUT REQUIREMENT:\nProvide the YAML plan now. Ensure valid YAML syntax."
    return prompt

def call_llm(model_cfg, system_prompt, user_prompt):
    """Calls the LLM with retries."""
    url = f"{model_cfg['url'].rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": model_cfg["temperature"],
        "max_tokens": model_cfg["max_tokens"],
        "top_p": model_cfg["top_p"]
    }

    retries = 2
    for attempt in range(retries + 1):
        try:
            logger.info(f"Sending request to LLM (Attempt {attempt+1}/{retries+1})...")
            response = requests.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=model_cfg["timeout"]
            )
            
            if response.status_code != 200:
                # If it's a server error or rate limit, we might retry
                if response.status_code in [429, 503, 500, 502, 504]:
                    raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
                # Otherwise, fail immediately on client error
                fail(ERR_MODEL_HTTP, f"Model returned HTTP {response.status_code}", {"body": response.text})

            try:
                data = response.json()
            except json.JSONDecodeError:
                fail(ERR_MODEL_BAD_RESP, "Model returned invalid JSON")

            if "choices" not in data or not data["choices"]:
                fail(ERR_MODEL_BAD_RESP, "Model response missing 'choices'")
            
            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                fail(ERR_EMPTY_RESP, "Model returned empty content")
            
            return content

        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            if attempt < retries:
                sleep_time = 1 if attempt == 0 else 3
                logger.warning(f"Connection error: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                fail(ERR_MODEL_UNREACHABLE, f"Could not connect to model: {str(e)}")
        
        except requests.exceptions.Timeout as e:
            if attempt < retries:
                sleep_time = 1 if attempt == 0 else 3
                logger.warning(f"Timeout: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                fail(ERR_MODEL_TIMEOUT, "Model request timed out")
        
        except requests.exceptions.RequestException as e:
            # Catch-all for HTTP errors raised above for retry logic
            if attempt < retries:
                sleep_time = 1 if attempt == 0 else 3
                logger.warning(f"HTTP Error: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                fail(ERR_MODEL_HTTP, f"Model HTTP error after retries: {str(e)}")
        except Exception as e:
            fail(ERR_INTERNAL, f"Unexpected error during model call: {str(e)}")

def extract_yaml(text):
    """Extracts YAML block from text."""
    # 1. Try markdown block (yaml or yml)
    match = re.search(r"```ya?ml\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. Try finding raw YAML start with "program:" (nested structure)
    match = re.search(r"(^|\n)(program:\s*\n\s+name:.*)", text, re.DOTALL)
    if match:
        return match.group(2).strip()
    
    # 3. Fallback: look for "program:" at start of a line (flat structure - legacy)
    match = re.search(r"(^|\n)(program:.*)", text, re.DOTALL)
    if match:
        return match.group(2).strip()
    
    return None

def normalize_filename(name: str) -> str:
    """
    Normalize a filename to a canonical form for comparison.
    Handles various formats LLMs might output:
    - parsers.base_parser.py -> parsers/base_parser.py
    - parsers/base_parser -> parsers/base_parser.py
    - ./parsers/base_parser.py -> parsers/base_parser.py
    """
    # Replace dots with slashes (but not the file extension dot)
    # e.g., "parsers.base_parser.py" -> "parsers/base_parser.py"
    if name.count('.') > 1:  # Has dots beyond just extension
        parts = name.rsplit('.', 1)  # Split off extension
        if len(parts) == 2 and parts[1] in ('py', 'json', 'yaml', 'yml', 'txt', 'md'):
            base = parts[0].replace('.', '/')
            name = f"{base}.{parts[1]}"
    
    # Remove leading ./
    if name.startswith('./'):
        name = name[2:]
    
    # Ensure .py extension for Python files without extension
    if '/' in name and not name.endswith(('.py', '.json', '.yaml', '.yml', '.txt', '.md', '.toml')):
        # Check if it looks like a Python module path
        if not any(name.endswith(ext) for ext in ['.py', '.json', '.yaml', '.yml', '.txt', '.md', '.toml']):
            name = name + '.py'
    
    return name


def build_filename_lookup(files: list) -> dict:
    """
    Build a lookup dict that maps various filename formats to the canonical name.
    This allows flexible matching when LLM uses different formats.
    """
    lookup = {}
    for f in files:
        canonical = f["name"]
        normalized = normalize_filename(canonical)
        
        # Add the canonical name
        lookup[canonical] = canonical
        lookup[normalized] = canonical
        
        # Add dotted version (parsers/base.py -> parsers.base.py)
        dotted = canonical.replace('/', '.')
        lookup[dotted] = canonical
        
        # Add without extension
        if canonical.endswith('.py'):
            no_ext = canonical[:-3]
            lookup[no_ext] = canonical
            lookup[no_ext.replace('/', '.')] = canonical
        
        # Add with ./ prefix
        lookup['./' + canonical] = canonical
        
    return lookup


def validate_plan_schema(plan):
    """
    Validates the parsed YAML object against the PlanExecutor schema.
    
    This schema matches what plan_executor.py expects:
    - program: dict with name, description, type
    - architecture: dict with pattern, entry_point
    - files: list of file specs
    - execution_order: list of filenames
    """
    if not isinstance(plan, dict):
        raise ValueError("Root YAML must be a dictionary.")

    # =========================================================================
    # VALIDATE TOP-LEVEL KEYS
    # =========================================================================
    required_keys = ["program", "architecture", "files", "execution_order"]
    for k in required_keys:
        if k not in plan:
            raise ValueError(f"Missing top-level key: {k}")

    # =========================================================================
    # VALIDATE 'program' SECTION
    # =========================================================================
    program = plan["program"]
    if isinstance(program, str):
        # Legacy flat format - convert to dict for compatibility
        plan["program"] = {
            "name": program,
            "description": program,
            "type": "cli"
        }
    elif isinstance(program, dict):
        # Modern nested format - validate required fields
        if "name" not in program:
            raise ValueError("'program' dict missing 'name' field.")
        # Optional fields with defaults
        program.setdefault("description", program["name"])
        program.setdefault("type", "cli")
    else:
        raise ValueError("'program' must be a string or dict.")

    # =========================================================================
    # VALIDATE 'architecture' SECTION
    # =========================================================================
    architecture = plan["architecture"]
    if isinstance(architecture, str):
        # Legacy flat format - convert to dict for compatibility
        plan["architecture"] = {
            "pattern": architecture,
            "entry_point": "main.py"
        }
    elif isinstance(architecture, dict):
        # Modern nested format - validate/default fields
        architecture.setdefault("pattern", "modular")
        architecture.setdefault("entry_point", "main.py")
    else:
        raise ValueError("'architecture' must be a string or dict.")

    # =========================================================================
    # VALIDATE 'files' SECTION
    # =========================================================================
    if not isinstance(plan["files"], list) or not plan["files"]:
        raise ValueError("'files' must be a non-empty list.")
    
    known_files = set()
    
    for idx, f in enumerate(plan["files"]):
        if not isinstance(f, dict):
            raise ValueError(f"Item {idx} in 'files' is not a dictionary.")
        if "name" not in f or not isinstance(f["name"], str):
            raise ValueError(f"File at index {idx} missing valid 'name'.")
        if "purpose" not in f:
            raise ValueError(f"File '{f['name']}' missing 'purpose'.")
        
        # Normalize optional fields to expected types
        if "dependencies" not in f:
            f["dependencies"] = []
        if "requirements" not in f:
            f["requirements"] = []
        if "exports" not in f:
            f["exports"] = []
        if "imports_from" not in f:
            f["imports_from"] = {}
            
        if not isinstance(f["dependencies"], list):
            raise ValueError(f"File '{f['name']}' dependencies must be a list.")
        if not isinstance(f["requirements"], list):
            raise ValueError(f"File '{f['name']}' requirements must be a list.")
        if not isinstance(f["exports"], list):
            raise ValueError(f"File '{f['name']}' exports must be a list.")
        if not isinstance(f["imports_from"], dict):
            raise ValueError(f"File '{f['name']}' imports_from must be a dict.")
        
        known_files.add(f["name"])

    # =========================================================================
    # VALIDATE 'execution_order' SECTION
    # =========================================================================
    if not isinstance(plan["execution_order"], list) or not plan["execution_order"]:
        raise ValueError("'execution_order' must be a non-empty list.")

    # =========================================================================
    # BUILD FILENAME LOOKUP FOR FLEXIBLE MATCHING
    # =========================================================================
    filename_lookup = build_filename_lookup(plan["files"])
    
    # =========================================================================
    # NORMALIZE AND VALIDATE REFERENTIAL INTEGRITY
    # =========================================================================
    # Check and normalize dependencies
    for f in plan["files"]:
        normalized_deps = []
        for dep in f["dependencies"]:
            if dep in filename_lookup:
                normalized_deps.append(filename_lookup[dep])
            elif normalize_filename(dep) in filename_lookup:
                normalized_deps.append(filename_lookup[normalize_filename(dep)])
            else:
                raise ValueError(f"File '{f['name']}' depends on unknown file '{dep}'.")
        f["dependencies"] = normalized_deps
        
        # Check and normalize imports_from
        normalized_imports = {}
        for import_file, imports in f["imports_from"].items():
            if import_file in filename_lookup:
                canonical = filename_lookup[import_file]
            elif normalize_filename(import_file) in filename_lookup:
                canonical = filename_lookup[normalize_filename(import_file)]
            else:
                raise ValueError(f"File '{f['name']}' imports from unknown file '{import_file}'.")
            normalized_imports[canonical] = imports
        f["imports_from"] = normalized_imports
    
    # Normalize and check execution_order
    normalized_order = []
    for item in plan["execution_order"]:
        if item in filename_lookup:
            normalized_order.append(filename_lookup[item])
        elif normalize_filename(item) in filename_lookup:
            normalized_order.append(filename_lookup[normalize_filename(item)])
        else:
            raise ValueError(f"execution_order contains unknown file '{item}'.")
    plan["execution_order"] = normalized_order

def main():
    # 1. Read STDIN
    try:
        raw_input = sys.stdin.read()
    except Exception as e:
        fail(ERR_INTERNAL, f"Failed to read STDIN: {e}")

    if len(raw_input) > MAX_INPUT_SIZE:
        fail(ERR_INPUT_TOO_LARGE, f"Input size {len(raw_input)} exceeds limit {MAX_INPUT_SIZE}")

    if not raw_input.strip():
        fail(ERR_INVALID_JSON_INPUT, "Empty input")

    # 2. Parse Input
    try:
        input_data = json.loads(raw_input)
    except json.JSONDecodeError as e:
        fail(ERR_INVALID_JSON_INPUT, f"Invalid JSON: {e}")

    # 3. Validate Input Data
    # Accept either user_request or user_message (coordinator sends user_message)
    user_request = input_data.get("user_request", "") or input_data.get("user_message", "")
    if not user_request:
        fail(ERR_MISSING_FIELD, "user_request or user_message is required and cannot be empty")
    
    # Accept either model or config (coordinator sends config)
    model_config = input_data.get("model") or input_data.get("config")
    if not model_config:
        fail(ERR_MISSING_FIELD, "model or config configuration is required")

    # Map config fields to expected names if needed
    if "url" not in model_config and "model_url" in model_config:
        model_config["url"] = model_config["model_url"]
    if "model" not in model_config and "model_name" in model_config:
        model_config["model"] = model_config["model_name"]
    # Set defaults for optional fields
    model_config.setdefault("api_type", "openai")
    model_config.setdefault("timeout", 600)
    model_config.setdefault("temperature", 0.6)
    model_config.setdefault("max_tokens", 25000)
    model_config.setdefault("top_p", 0.9)

    validate_model_config(model_config)

    clarification = input_data.get("clarification", "")
    
    # 4. Prepare Prompts
    # Use system_prompt from input if provided, otherwise generate default
    system_prompt = input_data.get("system_prompt") or generate_system_prompt()
    user_prompt = generate_user_prompt(user_request, clarification)

    # 5. Call LLM
    raw_response = call_llm(model_config, system_prompt, user_prompt)

    # 6. Extract YAML
    yaml_str = extract_yaml(raw_response)
    if not yaml_str:
        # Fallback: if the whole response looks like YAML (starts with program:), try it
        if "program:" in raw_response:
            yaml_str = raw_response
        else:
            fail(ERR_YAML_MISSING, "Could not extract YAML block from response", {"raw": raw_response})

    # 7. Check for Forbidden Config
    if FORBIDDEN_FILENAME in yaml_str:
        fail(ERR_YAML_PARSE, f"Plan contains forbidden file reference: {FORBIDDEN_FILENAME}")

    # 8. Parse and Validate YAML
    try:
        plan = yaml.safe_load(yaml_str)
    except yaml.YAMLError as e:
        fail(ERR_YAML_PARSE, f"YAML syntax error: {e}", {"extracted_yaml": yaml_str})

    try:
        validate_plan_schema(plan)
    except ValueError as e:
        fail(ERR_YAML_PARSE, f"Schema validation failed: {e}", {"extracted_yaml": yaml_str})

    # 9. Re-serialize to ensure normalized format
    try:
        normalized_yaml = yaml.dump(plan, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception:
        normalized_yaml = yaml_str  # Fallback to original if dump fails

    # 10. Success
    success(raw_response, normalized_yaml)

if __name__ == "__main__":
    main()
