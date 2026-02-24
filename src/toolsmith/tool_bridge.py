"""
Tool Bridge - Connector between SwarmCoordinator (host) and Tool Forge (Docker).

Executes Tool Forge operations inside the Docker container via `docker exec`.
All Tool Forge state (SQLite DB, tool files) lives inside the container.

Usage:
    bridge = ToolBridge()
    tools = bridge.search("csv parsing")
    tool = bridge.get("csv_parser")
    result = bridge.build(name="csv_parser", purpose="Parse CSV files", ...)
    output = bridge.run("csv_parser", {"file": "/data/test.csv"})
"""

import json
import subprocess
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CONTAINER_NAME = "swarm_sandbox"
WORKSPACE = "/workspace"


@dataclass
class ToolInfo:
    """Lightweight representation of a Tool Forge tool."""
    name: str
    slug: str
    purpose: str
    category: str
    version: str
    code: Optional[str] = None
    inputs: Optional[Dict] = None
    outputs: Optional[Dict] = None
    usage_count: int = 0


class ToolBridge:
    """Bridge between swarm host and Tool Forge running inside Docker.

    All operations are executed via `docker exec` running Python snippets
    inside the swarm_sandbox container where Tool Forge is deployed.
    """

    def __init__(self, container: str = CONTAINER_NAME,
                 db_path: str = f"{WORKSPACE}/eve_tools.db",
                 tools_dir: str = f"{WORKSPACE}/eve_tools",
                 docker_mode: bool = True):
        self.container = container
        self.db_path = db_path
        self.tools_dir = tools_dir
        self.docker_mode = docker_mode
        self._verified = False

    def _exec(self, python_code: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute a Python snippet inside the Docker container and return parsed JSON."""
        # Wrap the code to output JSON
        wrapper = f"""
import sys, json
sys.path.insert(0, '{WORKSPACE}')
try:
{self._indent(python_code, 4)}
except Exception as e:
    print(json.dumps({{"status": "error", "message": str(e)}}))
"""
        try:
            result = subprocess.run(
                ["docker", "exec", self.container, "python", "-c", wrapper],
                capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.error(f"Docker exec failed: {stderr}")
                return {"status": "error", "message": stderr}

            stdout = result.stdout.strip()
            if not stdout:
                return {"status": "error", "message": "Empty output from container"}

            # Parse last line as JSON (skip any logging output)
            lines = stdout.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{'):
                    return json.loads(line)

            return {"status": "error", "message": f"No JSON in output: {stdout[:200]}"}

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Container exec timed out after {timeout}s"}
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Invalid JSON: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _indent(self, code: str, spaces: int) -> str:
        """Indent each line of code."""
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n"))

    def _get_forge_init(self) -> str:
        """Return Python code to initialize a ToolForge instance."""
        return f"""
from src.config import Settings
from src.core.forge import ToolForge
settings = Settings(db_path="{self.db_path}", tools_dir="{self.tools_dir}", docker_mode={self.docker_mode})
forge = ToolForge(settings)
"""

    # ── Verify ──────────────────────────────────────────────────────

    def verify(self) -> bool:
        """Check that the container is running and Tool Forge is accessible."""
        if self._verified:
            return True

        result = self._exec(f"""
{self._get_forge_init()}
tools = forge.list()
print(json.dumps({{"status": "success", "tool_count": len(tools)}}))
""")
        if result.get("status") == "success":
            self._verified = True
            logger.info(f"Tool Forge verified: {result.get('tool_count', 0)} tools in inventory")
            return True

        logger.warning(f"Tool Forge verification failed: {result.get('message', 'unknown')}")
        return False

    # ── Search ──────────────────────────────────────────────────────

    def search(self, query: str) -> List[ToolInfo]:
        """Search Tool Forge for tools matching a keyword query."""
        result = self._exec(f"""
{self._get_forge_init()}
results = forge.search("{self._escape(query)}")
tools = []
for t in results:
    tools.append({{
        "name": t.name,
        "slug": t.slug,
        "purpose": t.purpose,
        "category": t.category or "custom",
        "version": t.version
    }})
print(json.dumps({{"status": "success", "tools": tools}}))
""")
        if result.get("status") != "success":
            logger.warning(f"Search failed: {result.get('message')}")
            return []

        return [ToolInfo(**t) for t in result.get("tools", [])]

    # ── List ────────────────────────────────────────────────────────

    def list_tools(self, category: str = None) -> List[ToolInfo]:
        """List all tools, optionally filtered by category."""
        cat_filter = f', category="{category}"' if category else ""
        result = self._exec(f"""
{self._get_forge_init()}
results = forge.list({cat_filter})
tools = []
for t in results:
    tools.append({{
        "name": t.name,
        "slug": t.slug,
        "purpose": t.purpose,
        "category": t.category or "custom",
        "version": t.version
    }})
print(json.dumps({{"status": "success", "tools": tools}}))
""")
        if result.get("status") != "success":
            return []
        return [ToolInfo(**t) for t in result.get("tools", [])]

    # ── Get ─────────────────────────────────────────────────────────

    def get(self, slug: str) -> Optional[ToolInfo]:
        """Get full tool details including code."""
        result = self._exec(f"""
{self._get_forge_init()}
t = forge.get("{self._escape(slug)}")
if t is None:
    print(json.dumps({{"status": "not_found"}}))
else:
    print(json.dumps({{
        "status": "success",
        "tool": {{
            "name": t.name,
            "slug": t.slug,
            "purpose": t.purpose,
            "category": t.category or "custom",
            "version": t.version,
            "code": t.code,
            "inputs": t.inputs,
            "outputs": t.outputs
        }}
    }}))
""")
        if result.get("status") != "success":
            return None
        tool_data = result.get("tool", {})
        return ToolInfo(**tool_data) if tool_data else None

    # ── Build ───────────────────────────────────────────────────────

    def build(self, name: str, purpose: str,
              inputs: Dict = None, outputs: Dict = None,
              dependencies: List[str] = None) -> Dict[str, Any]:
        """Build a new tool via Tool Forge (LLM generates code, validates, stores).

        Returns dict with 'status', 'slug', 'category', 'source_path'.
        """
        inputs_json = json.dumps(inputs or {})
        outputs_json = json.dumps(outputs or {})
        deps_json = json.dumps(dependencies or [])

        result = self._exec(f"""
{self._get_forge_init()}
from src.core.tool_spec import ToolSpec
spec = ToolSpec(
    name="{self._escape(name)}",
    purpose="{self._escape(purpose)}",
    inputs={inputs_json},
    outputs={outputs_json},
    dependencies={deps_json}
)
result = forge.build(spec)
print(json.dumps(result))
""", timeout=300)

        return result

    # ── Run ─────────────────────────────────────────────────────────

    def run(self, slug: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an existing tool with given inputs."""
        inputs_json = json.dumps(inputs)
        result = self._exec(f"""
{self._get_forge_init()}
result = forge.run("{self._escape(slug)}", {inputs_json})
print(json.dumps(result))
""")
        return result

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> List[Dict[str, Any]]:
        """Get usage statistics for all tools."""
        result = self._exec(f"""
{self._get_forge_init()}
stats = forge.stats()
print(json.dumps({{"status": "success", "stats": stats}}))
""")
        if result.get("status") != "success":
            return []
        return result.get("stats", [])

    # ── Helpers ─────────────────────────────────────────────────────

    def _escape(self, s: str) -> str:
        """Escape a string for safe embedding in Python code."""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def get_tool_inventory_summary(self) -> str:
        """Return a formatted summary of all available tools for LLM context."""
        tools = self.list_tools()
        if not tools:
            return "No tools currently available in Tool Forge."

        lines = [f"AVAILABLE TOOLS ({len(tools)} total):"]
        by_category = {}
        for t in tools:
            by_category.setdefault(t.category, []).append(t)

        for cat, cat_tools in sorted(by_category.items()):
            lines.append(f"\n  [{cat}]")
            for t in cat_tools:
                lines.append(f"    - {t.slug} (v{t.version}): {t.purpose}")

        return "\n".join(lines)

    def find_relevant_tools(self, task_description: str) -> List[ToolInfo]:
        """Search for tools relevant to a task description.

        Extracts keywords from the description and searches Tool Forge.
        Returns matching tools sorted by relevance.
        """
        # Extract meaningful keywords (skip common words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'and',
            'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
            'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
            'than', 'too', 'very', 'just', 'because', 'if', 'when', 'where',
            'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
            'those', 'it', 'its', 'my', 'your', 'his', 'her', 'their',
            'our', 'we', 'they', 'them', 'us', 'i', 'me', 'he', 'she',
            'create', 'build', 'make', 'write', 'implement', 'generate',
            'system', 'tool', 'program', 'application', 'code', 'function',
        }

        words = task_description.lower().split()
        keywords = [w.strip('.,!?:;()[]{}"\'-') for w in words
                     if w.strip('.,!?:;()[]{}"\'-') not in stop_words and len(w) > 2]

        # Search with top keywords
        seen_slugs = set()
        results = []
        for keyword in keywords[:8]:
            matches = self.search(keyword)
            for tool in matches:
                if tool.slug not in seen_slugs:
                    seen_slugs.add(tool.slug)
                    results.append(tool)

        return results
