#!/usr/bin/env python3
"""
Toolsmith Agent - Decision layer between SwarmCoordinator and Tool Forge.

This agent sits in the workflow before coding tasks. It:
1. Analyzes the task requirements
2. Searches Tool Forge for existing reusable tools
3. Decides: reuse existing tools, build new ones, or skip (write inline)
4. Returns tool context that gets injected into the coder's prompt

Runs as a subprocess agent following the AgentBase pattern.
"""

import sys
import os
import json
import logging

# Add parent dir for agent_base import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_base import AgentBase
from toolsmith.tool_bridge import ToolBridge

logger = logging.getLogger(__name__)


# Decisions the toolsmith can make
DECISION_REUSE = "reuse"        # Existing tool(s) found, inject into coder context
DECISION_BUILD = "build"        # No existing tool, build one via Tool Forge
DECISION_SKIP = "skip"          # Task doesn't benefit from tool creation
DECISION_BUILD_AFTER = "build_after"  # Code first, then extract reusable tool


class ToolsmithAgent(AgentBase):
    """Agent that decides when and how to use Tool Forge during code generation."""

    SYSTEM_PROMPT = """You are a Toolsmith - an intelligent decision layer that determines whether
a coding task should leverage pre-built tools from a Tool Forge (a persistent library of reusable
Python tools) or be written from scratch.

You will receive:
1. A task description (what needs to be built)
2. An architecture plan (if available)
3. A list of existing tools in the Tool Forge inventory

Your job is to analyze the task and return a JSON decision.

DECISION CRITERIA:

Choose "reuse" when:
- An existing tool directly matches a component needed by the task
- The tool's inputs/outputs align with what the task requires
- Using the tool would save significant development effort

Choose "build" when:
- The task requires a utility that would be reusable across future projects
- No existing tool matches, but the component is generic enough to be a tool
- Examples: CSV parsers, API clients, file processors, data validators
- The component is self-contained (single function, clear inputs/outputs)

Choose "build_after" when:
- The task is complex and needs to be coded first
- After coding, specific components could be extracted as reusable tools
- The extraction would benefit future projects

Choose "skip" when:
- The task is highly specific to this project (not reusable)
- The task is simple enough that a tool would be over-engineering
- The task is mostly glue code, configuration, or UI-specific logic
- The architecture is too intertwined for tool extraction

OUTPUT FORMAT - Return ONLY valid JSON:
{
    "decision": "reuse|build|build_after|skip",
    "reasoning": "Brief explanation of why",
    "reuse_tools": [
        {"slug": "tool_slug", "purpose": "why this tool is relevant", "usage_hint": "how to use it in this context"}
    ],
    "build_specs": [
        {"name": "tool_name", "purpose": "what it does", "inputs": {...}, "outputs": {...}}
    ],
    "extract_candidates": [
        {"name": "potential_tool_name", "purpose": "what to extract after coding"}
    ],
    "coder_context": "Additional context or instructions for the coder about using these tools"
}

For "reuse": populate reuse_tools
For "build": populate build_specs
For "build_after": populate extract_candidates
For "skip": leave arrays empty

Be practical. Not every task needs a tool. But when a task involves file I/O, data parsing,
API calls, system operations, or any utility that future projects might need - that's a tool."""

    def process(self, input_data: dict) -> dict:
        """Analyze task and make tool decision."""

        task_description = input_data.get("task_description", "")
        architecture_plan = input_data.get("architecture_plan", "")
        user_request = input_data.get("user_request", "")
        file_specs = input_data.get("file_specs", [])

        # Initialize bridge to Tool Forge
        bridge = ToolBridge()

        # Step 1: Verify Tool Forge is accessible
        if not bridge.verify():
            logger.warning("Tool Forge not accessible, skipping tool check")
            return {
                "status": "success",
                "decision": DECISION_SKIP,
                "reasoning": "Tool Forge not accessible",
                "tool_context": "",
                "reuse_tools": [],
                "build_specs": [],
                "extract_candidates": []
            }

        # Step 2: Get current tool inventory
        inventory = bridge.get_tool_inventory_summary()

        # Step 3: Search for relevant tools
        search_text = f"{user_request} {task_description}"
        relevant_tools = bridge.find_relevant_tools(search_text)

        relevant_summary = ""
        if relevant_tools:
            relevant_summary = "\n\nPOTENTIALLY RELEVANT TOOLS:"
            for t in relevant_tools:
                relevant_summary += f"\n  - {t.slug}: {t.purpose} [{t.category}]"

        # Step 4: Ask LLM to make the decision
        user_message = f"""Analyze this coding task and decide on tool usage.

TASK/USER REQUEST:
{user_request}

TASK DESCRIPTION:
{task_description}

ARCHITECTURE PLAN (if available):
{architecture_plan[:3000] if architecture_plan else 'Not yet created'}

FILE SPECS:
{json.dumps(file_specs[:10], indent=2) if file_specs else 'Not yet defined'}

CURRENT TOOL FORGE INVENTORY:
{inventory}
{relevant_summary}

Make your decision and return JSON."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.call_llm(messages, temperature=0.3)
        cleaned = self.clean_response(response)

        # Step 5: Parse the LLM's decision
        decision_data = self._parse_decision(cleaned)

        # Step 6: If decision is "build", actually build the tools now
        built_tools = []
        if decision_data.get("decision") == DECISION_BUILD:
            for spec in decision_data.get("build_specs", []):
                build_result = bridge.build(
                    name=spec.get("name", "unnamed_tool"),
                    purpose=spec.get("purpose", ""),
                    inputs=spec.get("inputs", {}),
                    outputs=spec.get("outputs", {})
                )
                if build_result.get("status") == "success":
                    built_tools.append({
                        "slug": build_result.get("slug"),
                        "category": build_result.get("category"),
                        "status": "built"
                    })
                    logger.info(f"Built tool: {build_result.get('slug')}")
                else:
                    logger.warning(f"Failed to build tool {spec.get('name')}: {build_result.get('message')}")
                    built_tools.append({
                        "name": spec.get("name"),
                        "status": "failed",
                        "error": build_result.get("message", "unknown")
                    })

        # Step 7: If decision is "reuse", fetch the actual tool code
        reuse_code = {}
        if decision_data.get("decision") == DECISION_REUSE:
            for tool_ref in decision_data.get("reuse_tools", []):
                slug = tool_ref.get("slug", "")
                if slug:
                    tool_info = bridge.get(slug)
                    if tool_info and tool_info.code:
                        reuse_code[slug] = {
                            "code": tool_info.code,
                            "purpose": tool_info.purpose,
                            "inputs": tool_info.inputs,
                            "outputs": tool_info.outputs
                        }

        # Step 8: Build coder context string
        coder_context = self._build_coder_context(decision_data, reuse_code, built_tools)

        return {
            "status": "success",
            "decision": decision_data.get("decision", DECISION_SKIP),
            "reasoning": decision_data.get("reasoning", ""),
            "tool_context": coder_context,
            "reuse_tools": decision_data.get("reuse_tools", []),
            "build_specs": decision_data.get("build_specs", []),
            "built_tools": built_tools,
            "extract_candidates": decision_data.get("extract_candidates", []),
            "result": coder_context  # Standard AgentBase output field
        }

    def _parse_decision(self, response: str) -> dict:
        """Parse the LLM's JSON decision, handling markdown and bad formatting."""
        text = response.strip()

        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "decision" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        brace_start = text.find('{')
        if brace_start >= 0:
            # Find matching closing brace
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[brace_start:i + 1])
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            break

        # Fallback: skip if we can't parse
        logger.warning(f"Could not parse toolsmith decision, defaulting to skip")
        return {
            "decision": DECISION_SKIP,
            "reasoning": "Could not parse LLM response",
            "reuse_tools": [],
            "build_specs": [],
            "extract_candidates": []
        }

    def _build_coder_context(self, decision: dict, reuse_code: dict,
                              built_tools: list) -> str:
        """Build context string that gets injected into the coder's prompt."""
        parts = []

        if decision.get("decision") == DECISION_REUSE and reuse_code:
            parts.append("=== REUSABLE TOOLS AVAILABLE ===")
            parts.append("The following pre-built tools are available from Tool Forge.")
            parts.append("You can import and use them in your code, or adapt their logic.\n")

            for slug, info in reuse_code.items():
                parts.append(f"--- Tool: {slug} ---")
                parts.append(f"Purpose: {info['purpose']}")
                parts.append(f"Inputs: {json.dumps(info.get('inputs', {}))}")
                parts.append(f"Outputs: {json.dumps(info.get('outputs', {}))}")
                parts.append(f"Code:\n{info['code']}")
                parts.append("")

        elif decision.get("decision") == DECISION_BUILD and built_tools:
            successful = [t for t in built_tools if t.get("status") == "built"]
            if successful:
                parts.append("=== NEWLY BUILT TOOLS ===")
                parts.append("These tools were just built by Tool Forge for this project:\n")
                for t in successful:
                    parts.append(f"  - {t['slug']} [{t['category']}]")
                parts.append("\nYou can use forge.run(slug, inputs) to call them.")

        elif decision.get("decision") == DECISION_BUILD_AFTER:
            candidates = decision.get("extract_candidates", [])
            if candidates:
                parts.append("=== POST-CODING TOOL EXTRACTION ===")
                parts.append("After implementing, consider extracting these as reusable tools:\n")
                for c in candidates:
                    parts.append(f"  - {c.get('name', '?')}: {c.get('purpose', '?')}")

        # Add any custom coder instructions from the LLM
        coder_hint = decision.get("coder_context", "")
        if coder_hint:
            parts.append(f"\n{coder_hint}")

        return "\n".join(parts) if parts else ""


if __name__ == "__main__":
    ToolsmithAgent().run()
