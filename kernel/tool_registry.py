"""
Agent OS — Tool Registry
Agents can call registered Python functions as tools.
The registry validates calls, executes them safely, and returns results.
"""

from __future__ import annotations
import asyncio
import inspect
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("agent_os.tool_registry")


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict        # JSON schema style
    fn: Callable
    is_async: bool = False


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolRegistry:
    """
    Register Python functions as tools agents can call.

    Usage:
        registry = ToolRegistry()

        @registry.register("calculator", "Evaluate a math expression")
        def calculator(expression: str) -> str:
            return str(eval(expression))

        result = await registry.call("calculator", {"expression": "2 + 2"})
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: dict | None = None):
        """Decorator to register a function as a tool."""
        def decorator(fn: Callable):
            is_async = asyncio.iscoroutinefunction(fn)
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or self._infer_parameters(fn),
                fn=fn,
                is_async=is_async,
            )
            logger.debug("TOOL REGISTERED: %s", name)
            return fn
        return decorator

    async def call(self, tool_name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(tool_name=tool_name, success=False,
                              error=f"Unknown tool: {tool_name}")
        try:
            if tool.is_async:
                output = await tool.fn(**arguments)
            else:
                output = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool.fn(**arguments)
                )
            logger.info("TOOL CALL %s(%s) → %s", tool_name, arguments, str(output)[:80])
            return ToolResult(tool_name=tool_name, success=True, output=output)
        except Exception as e:
            logger.error("TOOL ERROR %s: %s", tool_name, e)
            return ToolResult(tool_name=tool_name, success=False, error=str(e))

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        """Format all tools as a prompt string for the LLM."""
        if not self._tools:
            return "No tools available."
        lines = ["Available tools (call using TOOL: tool_name {\"arg\": \"value\"}):"]
        for t in self._tools.values():
            lines.append(f"\n- {t.name}: {t.description}")
            if t.parameters:
                for param, info in t.parameters.items():
                    desc = info.get("description", "") if isinstance(info, dict) else info
                    lines.append(f"    param '{param}': {desc}")
        return "\n".join(lines)

    def _infer_parameters(self, fn: Callable) -> dict:
        sig = inspect.signature(fn)
        params = {}
        for name, param in sig.parameters.items():
            params[name] = {"description": f"Parameter: {name}"}
        return params


# ------------------------------------------------------------------
# Built-in tools
# ------------------------------------------------------------------

def make_default_tools() -> ToolRegistry:
    """Create a registry with useful built-in tools."""
    registry = ToolRegistry()

    @registry.register(
        "calculator",
        "Evaluate a mathematical expression. Input must be a safe math expression.",
        {"expression": {"description": "Math expression e.g. '2 + 2' or '10 * 5'"}}
    )
    def calculator(expression: str) -> str:
        # Safe eval — only allow math
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            raise ValueError(f"Unsafe expression: {expression}")
        return str(eval(expression))

    @registry.register(
        "word_count",
        "Count the number of words in a text.",
        {"text": {"description": "The text to count words in"}}
    )
    def word_count(text: str) -> str:
        count = len(text.split())
        return f"{count} words"

    @registry.register(
        "reverse_text",
        "Reverse a string of text.",
        {"text": {"description": "The text to reverse"}}
    )
    def reverse_text(text: str) -> str:
        return text[::-1]

    @registry.register(
        "list_files",
        "List files in a directory.",
        {"path": {"description": "Directory path to list, e.g. '.' for current"}}
    )
    def list_files(path: str = ".") -> str:
        import os
        try:
            files = os.listdir(path)
            return "\n".join(files)
        except Exception as e:
            return f"Error: {e}"

    @registry.register(
        "get_time",
        "Get the current date and time.",
        {}
    )
    def get_time() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return registry
