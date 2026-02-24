#!/usr/bin/env python3
"""
Architect Agent for SwarmCoordinator
====================================

Creates YAML execution plans with strict export/import specifications.
Uses shared AgentBase infrastructure.
"""

import sys
import os
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agent_base import AgentBase, AgentError
    import yaml
except ImportError:
    import yaml
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
                    'url': config.get("model_url") or config.get("url", "http://localhost:1234/v1"),
                    'model': config.get("model_name") or config.get("model", "local-model"),
                    'api_type': config.get("api_type", "openai"),
                    'timeout': config.get("timeout", 600),
                    'temperature': config.get("temperature", 0.6),
                    'max_tokens': config.get("max_tokens", 8000),
                    'top_p': config.get("top_p", 0.9)
                })()
                result = self.process(self.input_data)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(json.dumps({"status": "error", "ok": False, "error": str(e)}))
                sys.exit(1)
        
        def call_llm(self, messages, temperature=None, max_tokens=None, retries=2):
            temp = temperature if temperature is not None else self.config.temperature
            tokens = max_tokens if max_tokens is not None else self.config.max_tokens
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": self.config.top_p
            }
            response = requests.post(
                f"{self.config.url.rstrip('/')}/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')


# Forbidden filename
FORBIDDEN_FILENAME = "config_v2.json"


class ArchitectAgent(AgentBase):
    """
    Architecture planning agent.
    
    Creates YAML plans with:
    - Explicit export specifications (names and types)
    - Import mappings between files
    - Dependency graphs
    - Execution order
    """
    
    SYSTEM_PROMPT = """You are an expert software architect creating a YAML execution plan.

Your plan will be STRICTLY ENFORCED - the coder MUST export exactly what you specify.
Be precise about exports and imports.

OUTPUT: Valid YAML with this structure:

```yaml
program:
  name: "project_name"
  description: "What it does"
  type: "cli|library|service"

architecture:
  pattern: "simple|modular"
  entry_point: "main.py"

files:
  - name: "config.py"
    purpose: "Configuration management"
    dependencies: []
    exports:
      - name: "Settings"
        type: "class"
    imports_from: {}
    requirements:
      - "Load config from file or environment"

  - name: "core.py"
    purpose: "Core business logic"
    dependencies: ["config.py"]
    exports:
      - name: "Processor"
        type: "class"
    imports_from:
      config.py: ["Settings"]
    requirements:
      - "Main processing logic"

  - name: "main.py"
    purpose: "Entry point"
    dependencies: ["config.py", "core.py"]
    exports: []
    imports_from:
      config.py: ["Settings"]
      core.py: ["Processor"]
    requirements:
      - "Parse CLI arguments"
      - "Initialize and run"

execution_order: ["config.py", "core.py", "main.py"]
```

CRITICAL RULES:
1. exports MUST list what each file defines (exact names!)
2. imports_from MUST only reference names in source file's exports
3. dependencies MUST include every file used in imports_from
4. execution_order MUST have dependencies before dependents

Keep it SIMPLE. Output ONLY the YAML."""

    DRAFT_PLAN_PROMPT = """You are an expert software architect creating a DRAFT YAML execution plan.

A coder will review this plan before it's finalized, so focus on WHAT needs to be built, not HOW to import it.

OUTPUT: Valid YAML with this structure:

```yaml
program:
  name: "project_name"
  description: "What it does"
  type: "cli|library|service"

architecture:
  pattern: "simple|modular"
  entry_point: "main.py"

files:
  - name: "config.py"
    purpose: "Configuration management"
    dependencies: []
    exports:
      - name: "Settings"
        type: "class"
        description: "Holds application configuration"
    needs_from: {}
    requirements:
      - "Load config from file or environment using dataclass with defaults"
      - "Validate that required paths exist, raise ValueError if not"

  - name: "core/processor.py"
    purpose: "Data processing pipeline"
    dependencies: ["config.py"]
    exports:
      - name: "Processor"
        type: "class"
        description: "Processes input data files"
        methods:
          __init__:
            args: ["self", "settings: Settings"]
            returns: "None"
          run:
            args: ["self", "input_path: Path"]
            returns: "dict"
          process_batch:
            args: ["self", "paths: list[Path]"]
            returns: "list[dict]"
    needs_from:
      config.py: ["Settings"]
    requirements:
      - "Read CSV files using csv.DictReader, extract rows matching filter criteria"
      - "run(input_path: Path) -> dict: process one file, return dict with keys: rows (list[dict]), stats (dict)"
      - "process_batch(paths: list[Path]) -> list[dict]: process multiple files, return list of result dicts"
      - "Compute statistics (mean, median, std) using statistics module"
      - "Write results to JSON using json.dump with indent=2"

  - name: "main.py"
    purpose: "Entry point"
    dependencies: ["config.py", "core/processor.py"]
    exports: []
    needs_from:
      config.py: ["Settings"]
      "core/processor.py": ["Processor"]
    requirements:
      - "Parse CLI arguments with argparse (--input, --output, --config)"
      - "Initialize Settings(), then Processor(settings), then call processor.run(input_path) -> dict"
      - "Print or save the dict returned by processor.run()"

execution_order: ["config.py", "core/processor.py", "main.py"]
```

CRITICAL RULES:
1. exports lists WHAT each file defines — name, type, and a short description
2. needs_from declares WHAT names a file needs from which other file (the coder decides HOW to import)
3. dependencies MUST include every file referenced in needs_from
4. execution_order MUST have dependencies before dependents
5. Keep it SIMPLE. If it's a simple program, use few files. Don't over-engineer.
6. Use forward slashes for subdirectory files: "core/processor.py" not "core.processor"
7. Quote filenames containing slashes in YAML: "core/processor.py"

COMPLETENESS RULES (prevents missing features):
7a. The entry_point file MUST exist in the files list. If entry_point is "main.py", there MUST be a file named "main.py".
7b. EVERY feature in the job spec MUST map to a requirement in at least one file. Before outputting, mentally check each
    requirement from the spec and verify it appears somewhere in your plan. If the spec mentions periodic tasks, background
    jobs, pruning, cleanup, caching, or scheduled operations — they MUST appear as requirements on a file.
7c. Non-Python files (HTML, CSS, JS, config files) are allowed in the plan. Mark them with no exports and no needs_from.
    The entry point or API file MUST have a requirement to serve/load these files.
7d. requirements.txt should only list installable packages (not stdlib modules like sqlite3, json, os, etc.)

REQUIREMENTS MUST BE IMPLEMENTATION-SPECIFIC:
8. Each requirement MUST specify HOW to implement, not just WHAT.
   BAD:  "Implement document classification"
   GOOD: "Classify documents by file extension and keyword matching using regex patterns"
   BAD:  "Use vision-language models for images"
   GOOD: "Read image files with PIL, extract text with pytesseract OCR, classify by keywords in extracted text"
   BAD:  "Store data in database"
   GOOD: "Store records in SQLite using sqlite3 module with tables for documents(id, path, hash, status)"
9. Requirements must reference specific Python stdlib or common libraries (requests, sqlite3, re, json, csv, pathlib, etc.)
10. If a feature requires an external API or ML model that may not be available, the requirement MUST specify a working fallback using stdlib.
    Example: "Classify documents using LLM if available, otherwise use keyword-matching rules with regex"

INTERFACE CONTRACTS (CRITICAL — prevents cross-file mismatches):
11. Config/settings/dataclass files MUST list ALL their fields in the requirements.
    BAD:  "Load config from environment using dataclass with defaults"
    GOOD: "Dataclass with fields: log_directories (list[str]), max_file_size (int, default 10485760), encoding (str, default 'utf-8'), output_dir (Path, default ~/.eve/logs)"
12. If downstream files need specific attributes from a class, those attributes MUST be declared in that class's requirements.
    Think: what fields/methods will other files actually call on this class? List them ALL.
13. Class exports MUST include a `methods:` block for every public method other files will call.
    The `methods:` block is MACHINE-READ — it drives mandatory export checks and coder prompts.
    DO NOT put method signatures in the description string — they are ignored by tooling.
    BAD (description string — IGNORED):
      exports:
        - name: "Processor"
          type: "class"
          description: "Processes files. Methods: run(path) -> dict"  # ← tooling never reads this
    GOOD (structured methods: block — ENFORCED):
      exports:
        - name: "Processor"
          type: "class"
          description: "Processes input data files"
          methods:
            __init__:
              args: ["self", "settings: Settings"]
              returns: "None"
            run:
              args: ["self", "input_path: Path"]
              returns: "dict"
            process_batch:
              args: ["self", "paths: list[Path]"]
              returns: "list[dict]"
    Every method that another file will call MUST appear in this block.
    The entry point file (main.py) depends entirely on these signatures — if they are wrong or missing,
    the entry point coder WILL invent wrong method names and the project will not run.
14. RETURN TYPES ARE CRITICAL. Every method mentioned in requirements or exports MUST specify what it returns.
    BAD:  "Implement get_summary method for generating text summary"
    GOOD: "get_summary() -> str: returns plain English summary of findings"
    BAD:  "Return categorized list of log files"
    GOOD: "discover_logs() -> list[dict]: returns list of dicts with keys: path (str), size (int), type (str)"
15. Methods that mutate state in-place MUST say so explicitly. Do NOT let downstream files guess.
    BAD:  "detect_patterns method to find anomalies"
    GOOD: "detect_patterns() -> list[dict]: returns list of detected anomaly dicts with keys: type, description, severity"
    (NOT: "detect_patterns() modifies self.anomalies in-place, returns None" — prefer returning data)

Output ONLY the YAML."""

    FINALIZE_PROMPT = """You are an expert software architect FINALIZING a plan based on a coder's review.

The coder reviewed your draft plan and provided feedback. Adjust accordingly.
The ORIGINAL REQUIREMENTS will be provided — your final plan MUST cover all of them.

RULES:
1. Verify the final plan covers ALL original requirements — do not drop features when addressing feedback
2. Address every concern the coder raised
3. If the coder suggests structural changes, adopt them if reasonable
4. If you disagree with the coder, explain why in a YAML comment
5. The plan must still use the same YAML format (with needs_from, NOT imports_from)
6. Output the COMPLETE revised YAML plan
7. Config/settings/dataclass files MUST list ALL their fields in requirements
   Example: "Dataclass with fields: log_dirs (list[str]), max_file_size (int, default 10MB), encoding (str, default 'utf-8')"
8. Every class export MUST have a `methods:` block — NOT a description string with method names.
   The `methods:` block is machine-read and drives coder prompts. Description strings are ignored.
   BAD:  description: "Parses logs. Methods: parse_file(path) -> list[dict]"   ← IGNORED
   GOOD:
     methods:
       parse_file:
         args: ["self", "path: Path"]
         returns: "list[dict]"
       discover:
         args: ["self", "dirs: list[str]"]
         returns: "list[Path]"
9. Cross-check: for every method a file calls on a dependency, verify that method exists in the
   dependency's `methods:` block with matching argument names and return type.
10. main.py and other entry points MUST list exact method calls in requirements:
    BAD:  "Initialize and run the processor"
    GOOD: "Initialize Processor(settings), call processor.run(input_path) -> dict, print result" """
    
    def process(self, input_data: dict) -> dict:
        """Generate YAML execution plan."""

        mode = input_data.get("mode", "default")

        if mode == "draft":
            return self._process_draft(input_data)
        elif mode == "finalize":
            return self._process_finalize(input_data)
        else:
            return self._process_default(input_data)

    def _process_draft(self, input_data: dict) -> dict:
        """Draft plan mode — receives job_spec, produces draft YAML plan."""
        job_spec = input_data.get("job_spec", "")
        if not job_spec:
            return {"status": "error", "ok": False, "error": "No job_spec provided for draft mode"}

        validation_error = input_data.get("validation_error", "")
        fix_section = ""
        if validation_error:
            fix_section = (
                f"\n\nPREVIOUS PLAN REJECTED — FIX THESE ERRORS:\n{validation_error}\n\n"
                "Common fixes:\n"
                "- Python builtins (str, int, Exception, Path, etc.) do NOT need needs_from entries — just use them\n"
                "- If a type IS defined in your plan (e.g. a class you export), the file using it MUST list it in needs_from\n"
                "- If a parameter has a default value, use the format 'param: type = default' with spaces around '='\n"
                "- Use path-based filenames (core/module.py not core.module)\n"
                "- needs_from must ONLY reference internal project files, not external packages\n"
            )

        user_prompt = f"Create a YAML execution plan for this project:\n\n{job_spec}{fix_section}\n\nOutput the YAML plan:"

        messages = [
            {"role": "system", "content": self.DRAFT_PLAN_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.call_llm(messages, temperature=0.6)
        return self._parse_and_validate(response)

    def _process_finalize(self, input_data: dict) -> dict:
        """Finalize plan mode — receives draft_plan + coder_feedback + job_spec, produces final YAML plan."""
        draft_plan = input_data.get("draft_plan", "")
        coder_feedback = input_data.get("coder_feedback", "")
        job_spec = input_data.get("job_spec", "")
        validation_error = input_data.get("validation_error", "")

        if not draft_plan:
            return {"status": "error", "ok": False, "error": "No draft_plan provided for finalize mode"}

        job_spec_section = f"\nORIGINAL REQUIREMENTS (verify your final plan covers ALL of these):\n{job_spec}\n" if job_spec else ""
        fix_section = ""
        if validation_error:
            fix_section = (
                f"\nPREVIOUS PLAN REJECTED — FIX THESE ERRORS:\n{validation_error}\n\n"
                "Common fixes:\n"
                "- Python builtins (str, int, Exception, Path, etc.) do NOT need needs_from entries — just use them\n"
                "- If a type IS defined in your plan (e.g. a class you export), the file using it MUST list it in needs_from\n"
                "- If a parameter has a default value, use the format 'param: type = default' with spaces around '='\n"
                "- File names in needs_from must be exact path-based names (core/module.py not core.module)\n"
                "- needs_from must ONLY reference internal project files, not external packages\n"
            )

        user_prompt = f"""{job_spec_section}{fix_section}
DRAFT PLAN:
```yaml
{draft_plan}
```

CODER'S REVIEW:
{coder_feedback}

Revise the plan based on the coder's feedback. Ensure the final plan still covers ALL original requirements.
Output the COMPLETE revised YAML plan:"""

        messages = [
            {"role": "system", "content": self.FINALIZE_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.call_llm(messages, temperature=0.6)
        return self._parse_and_validate(response)

    def _process_default(self, input_data: dict) -> dict:
        """Default mode — original behavior for backward compatibility."""
        # Get user request
        user_request = input_data.get("user_request") or input_data.get("user_message", "")
        if not user_request:
            return {"status": "error", "ok": False, "error": "No user_request provided"}

        clarification = input_data.get("clarification", "")

        # Use custom system prompt if provided
        system_prompt = input_data.get("system_prompt") or self.SYSTEM_PROMPT

        # Build user prompt
        user_prompt = f"Create a YAML execution plan for:\n\n{user_request}"
        if clarification:
            user_prompt += f"\n\nCLARIFICATION:\n{clarification}"
        user_prompt += "\n\nOutput the YAML plan:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM
        response = self.call_llm(messages, temperature=0.6)
        return self._parse_and_validate(response)
    
    def _parse_and_validate(self, response: str) -> dict:
        """Shared logic: extract YAML from LLM response, validate, and return result dict."""
        yaml_str = self._extract_yaml(response)
        if not yaml_str:
            return {
                "status": "error",
                "ok": False,
                "error": "Could not extract YAML from response",
                "raw_response": response
            }

        if FORBIDDEN_FILENAME in yaml_str:
            return {
                "status": "error",
                "ok": False,
                "error": f"Plan contains forbidden file: {FORBIDDEN_FILENAME}"
            }

        try:
            plan = yaml.safe_load(yaml_str)
        except yaml.YAMLError:
            yaml_str = self._repair_yaml(yaml_str)
            try:
                plan = yaml.safe_load(yaml_str)
            except yaml.YAMLError as e:
                return {
                    "status": "error",
                    "ok": False,
                    "error": f"YAML parse error: {e}",
                    "extracted_yaml": yaml_str
                }

        try:
            self._validate_plan(plan)
            normalized = yaml.dump(plan, default_flow_style=False, sort_keys=False)

            return {
                "status": "success",
                "ok": True,
                "result": normalized,
                "plan_yaml": normalized,
                "raw_response": response
            }
        except ValueError as e:
            return {
                "status": "error",
                "ok": False,
                "error": f"Validation error: {e}",
                "extracted_yaml": yaml_str
            }

    def _extract_yaml(self, response: str) -> str:
        """Extract YAML from response."""
        # Try markdown block
        match = re.search(r'```ya?ml\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try raw YAML
        match = re.search(r'(program:\s*\n.*)', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return whole thing if it starts with program:
        if response.strip().startswith('program:'):
            return response.strip()
        
        return ""

    @staticmethod
    def _repair_yaml(yaml_str: str) -> str:
        """Fix common YAML issues from LLM output.

        Handles:
        - Unquoted values containing colons (Python type hints, method sigs)
        - Unquoted values containing ``->``
        - Bare list items with embedded colons
        - Orphaned aliases (*idNNN) without anchor definitions
        """
        import re as _re

        # Resolve anchors/aliases: collect defined anchors, then replace any
        # orphaned *idNNN aliases (no matching &idNNN) with {} so YAML parses.
        defined_anchors = set(_re.findall(r'&(id\d+)', yaml_str))
        used_aliases = set(_re.findall(r'\*(id\d+)', yaml_str))
        orphaned = used_aliases - defined_anchors
        for alias in orphaned:
            yaml_str = _re.sub(r'\*' + alias + r'\b', '{}', yaml_str)

        repaired_lines = []
        for line in yaml_str.split("\n"):
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]

            # Skip comments, blank lines, block scalars
            if not stripped or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith(">"):
                repaired_lines.append(line)
                continue

            # Handle list items: "  - some value with: colon"
            list_prefix = ""
            working = stripped
            if stripped.startswith("- "):
                list_prefix = "- "
                working = stripped[2:]

            # Try to match a YAML key: value pair
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:\s?(.*)', working)
            if m:
                key, value = m.group(1), m.group(2)

                # If value is empty or a YAML token, skip
                if not value or value.startswith(("&", "*")):
                    repaired_lines.append(line)
                    continue

                # Check quoted values — only safe if quote is closed with no trailing content
                if value.startswith(("'", '"')):
                    q = value[0]
                    close_idx = value.find(q, 1)
                    while close_idx != -1 and close_idx > 0 and value[close_idx - 1] == '\\':
                        close_idx = value.find(q, close_idx + 1)
                    if close_idx != -1 and close_idx == len(value) - 1:
                        repaired_lines.append(line)
                        continue
                    # Trailing content after close quote — re-quote the whole value
                    safe_value = value.replace("'", "''")
                    repaired_lines.append(f"{indent}{list_prefix}{key}: '{safe_value}'")
                    continue

                # Inline flow sequence: [item1, item2, ...] — quote unquoted items with colons
                if value.startswith("[") and value.endswith("]"):
                    inner = value[1:-1]
                    fixed_items = []
                    changed = False
                    for item in inner.split(","):
                        s = item.strip()
                        if s and not s.startswith(("'", '"')) and (":" in s or "->" in s):
                            fixed_items.append(f"'{s.replace(chr(39), chr(39)*2)}'")
                            changed = True
                        else:
                            fixed_items.append(item)
                    if changed:
                        repaired_lines.append(f"{indent}{list_prefix}{key}: [{', '.join(i.strip() for i in fixed_items)}]")
                    else:
                        repaired_lines.append(line)
                    continue

                # Skip other block/mapping starts
                if value.startswith(("{",)):
                    repaired_lines.append(line)
                    continue

                # Value contains colon or arrow — needs quoting
                if ":" in value or "->" in value:
                    safe_value = value.replace("'", "''")
                    repaired_lines.append(f"{indent}{list_prefix}{key}: '{safe_value}'")
                else:
                    repaired_lines.append(line)

            elif list_prefix and working:
                # Bare list item (no key: value structure)
                # Check if it looks quoted but has trailing content after the
                # closing quote, e.g.: "some value" extra stuff
                # YAML will parse the quoted part as a scalar and choke on the rest.
                if working.startswith(("'", '"')):
                    q = working[0]
                    # Find the matching close quote (skip escaped quotes)
                    close_idx = working.find(q, 1)
                    while close_idx != -1 and working[close_idx - 1] == '\\':
                        close_idx = working.find(q, close_idx + 1)
                    # If properly quoted with nothing after → safe
                    if close_idx != -1 and close_idx == len(working) - 1:
                        repaired_lines.append(line)
                        continue
                    # Otherwise fall through to re-quote the whole value

                # Contains ": " or "->" or problematic quoting — quote the whole value
                if ": " in working or "->" in working or working.startswith(("'", '"')):
                    safe_value = working.replace("'", "''")
                    repaired_lines.append(f"{indent}- '{safe_value}'")
                else:
                    repaired_lines.append(line)
            else:
                repaired_lines.append(line)

        return "\n".join(repaired_lines)

    def _validate_plan(self, plan: dict):
        """Validate plan structure and consistency."""
        
        # Required sections
        if 'program' not in plan:
            raise ValueError("Missing 'program' section")
        if 'files' not in plan or not plan['files']:
            raise ValueError("Missing or empty 'files' section")
        if 'execution_order' not in plan or not plan['execution_order']:
            raise ValueError("Missing or empty 'execution_order' section")
        
        # Normalize program
        if isinstance(plan['program'], str):
            plan['program'] = {'name': plan['program'], 'description': plan['program'], 'type': 'cli'}
        
        # Normalize architecture
        if 'architecture' not in plan:
            plan['architecture'] = {'pattern': 'simple', 'entry_point': 'main.py'}
        elif isinstance(plan['architecture'], str):
            plan['architecture'] = {'pattern': plan['architecture'], 'entry_point': 'main.py'}
        
        # Build file lookup
        file_names = set()
        file_exports = {}  # filename -> set of export names
        
        for f in plan['files']:
            if not isinstance(f, dict) or 'name' not in f:
                raise ValueError("Invalid file entry")
            
            name = f['name']
            file_names.add(name)
            
            # Extract exports
            exports = set()
            for exp in f.get('exports', []):
                if isinstance(exp, dict):
                    exports.add(exp.get('name', ''))
                elif isinstance(exp, str):
                    exports.add(exp)
            file_exports[name] = exports
            
            # Ensure required fields with defaults
            f.setdefault('purpose', '')
            f.setdefault('dependencies', [])
            f.setdefault('exports', [])
            f.setdefault('requirements', [])

            # Accept needs_from as equivalent to imports_from
            if 'needs_from' in f and 'imports_from' not in f:
                f['imports_from'] = f['needs_from']
            f.setdefault('imports_from', {})
            f.setdefault('needs_from', f.get('imports_from', {}))

        # Validate cross-references
        for f in plan['files']:
            name = f['name']

            # Check dependencies exist
            for dep in f['dependencies']:
                if dep not in file_names:
                    raise ValueError(f"File '{name}' depends on unknown file '{dep}'")

            # Check imports_from/needs_from references valid files and exports
            import_map = f.get('imports_from', {}) or f.get('needs_from', {})
            if not isinstance(import_map, dict):
                import_map = {}
            for source, imports in import_map.items():
                if source not in file_names:
                    raise ValueError(f"File '{name}' imports from unknown file '{source}'")
                
                # Check that imports exist in source exports
                source_exports = file_exports.get(source, set())
                for imp in imports:
                    if imp not in source_exports:
                        raise ValueError(
                            f"File '{name}' imports '{imp}' from '{source}', "
                            f"but '{source}' only exports: {sorted(source_exports) if source_exports else '(nothing)'}"
                        )
                
                # Ensure source is in dependencies
                if source not in f['dependencies']:
                    f['dependencies'].append(source)
        
        # Validate execution order
        for item in plan['execution_order']:
            if item not in file_names:
                raise ValueError(f"execution_order contains unknown file '{item}'")

        # Validate that non-primitive types in method signatures are declared in needs_from
        # This ensures coders receive dependency code for every type they need to use.
        PRIMITIVES = {
            # Python builtins
            'str', 'int', 'float', 'bool', 'dict', 'list', 'tuple', 'set', 'bytes',
            'bytearray', 'memoryview', 'complex', 'frozenset', 'None', 'object',
            # Builtin exceptions
            'Exception', 'BaseException', 'ValueError', 'TypeError', 'KeyError',
            'IndexError', 'AttributeError', 'RuntimeError', 'OSError', 'IOError',
            'FileNotFoundError', 'ConnectionError', 'TimeoutError', 'StopIteration',
            'NotImplementedError', 'PermissionError', 'ImportError', 'ArithmeticError',
            'LookupError', 'StopAsyncIteration', 'SystemExit', 'KeyboardInterrupt',
            # typing module
            'Any', 'Optional', 'Union', 'List', 'Dict', 'Tuple', 'Set', 'FrozenSet',
            'Callable', 'Iterator', 'Generator', 'Type', 'ClassVar', 'Final',
            'Sequence', 'Mapping', 'MutableMapping', 'Iterable', 'IO',
            'Coroutine', 'Awaitable', 'AsyncIterator', 'AsyncGenerator',
            'NamedTuple', 'TypedDict', 'Protocol', 'Literal',
            # stdlib types commonly used in signatures
            'Path', 'PurePath', 'PosixPath', 'WindowsPath',  # pathlib
            'datetime', 'date', 'time', 'timedelta', 'timezone',  # datetime
            'Enum', 'IntEnum', 'StrEnum',  # enum
            'dataclass',  # dataclasses
            'Pattern', 'Match',  # re
            'Logger',  # logging
            'TextIO', 'BinaryIO',  # io
            'Queue',  # queue
            'Thread', 'Lock', 'Event',  # threading
            'Decimal',  # decimal
            'UUID',  # uuid
            'deque', 'defaultdict', 'Counter', 'OrderedDict',  # collections
            'Connection', 'Cursor',  # sqlite3
            'Process',  # multiprocessing
            # generic type vars
            'self', 'cls', 'T', 'KT', 'VT', 'S',
        }
        import re as _re
        for f in plan['files']:
            name = f['name']
            import_map = f.get('imports_from', {}) or f.get('needs_from', {})
            if not isinstance(import_map, dict):
                import_map = {}
            importable_types = set()
            for src_imports in import_map.values():
                if isinstance(src_imports, list):
                    importable_types.update(src_imports)
            own_exports = file_exports.get(name, set())

            for exp in f.get('exports', []):
                if not isinstance(exp, dict):
                    continue
                methods = exp.get('methods', {})
                if not isinstance(methods, dict):
                    continue
                for method_name, method_info in methods.items():
                    if not isinstance(method_info, dict):
                        continue
                    for arg_str in method_info.get('args', []):
                        if not isinstance(arg_str, str) or ':' not in arg_str:
                            continue
                        type_part = arg_str.split(':', 1)[1].strip()
                        # Strip default values: str = '~/.eve/watchdog' → str
                        # Only split on '=' that is outside quotes
                        if '=' in type_part:
                            eq_idx = type_part.find('=')
                            before_eq = type_part[:eq_idx]
                            if before_eq.count("'") % 2 == 0 and before_eq.count('"') % 2 == 0:
                                type_part = before_eq.strip()
                        # Strip generics: list[T] → list, Optional[T] → Optional
                        base_type = _re.sub(r'\[.*', '', type_part).strip()
                        base_type = base_type.lstrip('*').strip()
                        # Handle union types: str|bytes → check each part separately
                        union_parts = [p.strip() for p in base_type.replace('|', ',').split(',')]
                        for upart in union_parts:
                            # Strip dotted module prefix: pathlib.Path → Path
                            leaf_type = upart.rsplit('.', 1)[-1] if '.' in upart else upart
                            if (upart and leaf_type not in PRIMITIVES
                                    and upart not in own_exports and leaf_type not in own_exports
                                    and upart not in importable_types and leaf_type not in importable_types):
                                raise ValueError(
                                    f"File '{name}' method '{exp.get('name','?')}.{method_name}' "
                                    f"parameter '{arg_str}' uses type '{upart}', but '{upart}' "
                                    f"is not listed in needs_from. Add the file that exports '{upart}' "
                                    f"to needs_from so coders can see its interface."
                                )


if __name__ == "__main__":
    ArchitectAgent().run()
