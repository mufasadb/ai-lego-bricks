#!/usr/bin/env python3
"""
Comprehensive end-to-end test of the restored tool calling functionality.

This test demonstrates:
1. Tool registration in the global registry
2. Tool preparation for different LLM providers
3. LLM tool calling with auto-execution
4. Multi-turn conversations with tools
5. Full workflow integration through Agent Orchestrator

Created to validate the tool calling restoration work.
"""

import sys
import asyncio

sys.path.append(".")

from tools.tool_types import (
    ToolSchema,
    ToolParameter,
    ParameterType,
    ToolExecutor,
    ToolCall,
    ToolResult,
    Tool,
)
from tools.tool_registry import get_global_registry
from tools.tool_service import ToolService
from agent_orchestration import AgentOrchestrator


class CalculatorExecutor(ToolExecutor):
    """Simple calculator tool that performs basic math operations."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        try:
            operation = tool_call.parameters.get("operation", "")
            print(f"🧮 Calculator executing: {operation}")

            # Safe evaluation of basic math operations
            allowed_chars = "0123456789+-*/.() "
            if all(c in allowed_chars for c in operation):
                result = eval(operation)
                print(f"🧮 Calculator result: {result}")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={
                        "calculation": operation,
                        "result": result,
                        "message": f"Calculated {operation} = {result}",
                    },
                )
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=None,
                    error="Invalid characters in operation",
                )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e),
            )


class WeatherExecutor(ToolExecutor):
    """Mock weather tool for testing."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        try:
            location = tool_call.parameters.get("location", "Unknown")
            print(f"🌤️ Weather tool getting weather for: {location}")

            # Mock weather data
            weather_data = {
                "London": {"temp": "15°C", "condition": "Cloudy", "humidity": "78%"},
                "New York": {"temp": "22°C", "condition": "Sunny", "humidity": "65%"},
                "Tokyo": {"temp": "18°C", "condition": "Rainy", "humidity": "85%"},
            }

            weather = weather_data.get(
                location, {"temp": "20°C", "condition": "Pleasant", "humidity": "70%"}
            )
            weather["location"] = location

            print(f"🌤️ Weather result: {weather}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result={
                    "weather": weather,
                    "message": f"Weather in {location}: {weather['temp']}, {weather['condition']}",
                },
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=str(e),
            )


def create_calculator_tool():
    """Create calculator tool with proper schema."""
    schema = ToolSchema(
        name="calculator",
        description="Perform mathematical calculations. Use this for any arithmetic operations.",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "operation": ToolParameter(
                    type=ParameterType.STRING,
                    description='Mathematical operation to perform (e.g., "25 * 8", "100 + 47")',
                )
            },
            required=["operation"],
        ),
    )
    return Tool(schema=schema, executor=CalculatorExecutor())


def create_weather_tool():
    """Create weather tool with proper schema."""
    schema = ToolSchema(
        name="get_weather",
        description="Get current weather information for a city.",
        parameters=ToolParameter(
            type=ParameterType.OBJECT,
            properties={
                "location": ToolParameter(
                    type=ParameterType.STRING,
                    description="City name to get weather for",
                )
            },
            required=["location"],
        ),
    )
    return Tool(schema=schema, executor=WeatherExecutor())


async def register_test_tools():
    """Register test tools in the global registry."""
    try:
        registry = await get_global_registry()

        # Register calculator tool
        calculator_tool = create_calculator_tool()
        await registry.register_tool(calculator_tool, "math")
        print("✅ Registered calculator tool")

        # Register weather tool
        weather_tool = create_weather_tool()
        await registry.register_tool(weather_tool, "utilities")
        print("✅ Registered weather tool")

        # Verify registration
        all_tools = await registry.get_all_schemas()
        print(f"📋 Total tools registered: {len(all_tools)}")
        for tool in all_tools:
            print(f"  - {tool.name}: {tool.description}")

        return True

    except Exception as e:
        print(f"❌ Error registering tools: {e}")
        return False


async def test_tool_preparation():
    """Test tool preparation for different providers."""
    try:
        tool_service = ToolService()

        # Test different providers
        providers = ["gemini", "ollama"]
        tool_names = ["calculator", "get_weather"]

        print("\n🔧 Testing tool preparation:")
        for provider in providers:
            tool_config = await tool_service.prepare_tools_for_provider(
                provider, tool_names
            )
            print(
                f"  {provider}: {len(tool_config.get('tools', tool_config.get('functionDeclarations', [])))} tools prepared"
            )

        return True

    except Exception as e:
        print(f"❌ Error testing tool preparation: {e}")
        return False


def run_orchestrator_test():
    """Run tool calling test through Agent Orchestrator."""
    try:
        # Create test workflow
        workflow_dict = {
            "name": "comprehensive_tool_test",
            "description": "Test restored tool calling functionality end-to-end",
            "config": {
                "default_llm_provider": "gemini",
                "default_model": "gemini-1.5-flash",
                "max_iterations": 5,
            },
            "steps": [
                {
                    "id": "tool_calling_test",
                    "type": "tool_call",
                    "description": "Test multi-tool execution",
                    "config": {
                        "provider": "gemini",
                        "model": "gemini-1.5-flash",
                        "temperature": 0.1,
                        "max_tokens": 800,
                        "tools": ["calculator", "get_weather"],
                        "tool_choice": "auto",
                        "max_iterations": 3,
                        "auto_execute": True,
                        "system_prompt": "You are a helpful assistant with access to calculator and weather tools. Always use these tools when appropriate for calculations or weather requests.",
                    },
                    "inputs": {
                        "message": "Calculate 147 * 63 and then get the weather for London. Use the appropriate tools for each task."
                    },
                    "outputs": ["test_result"],
                }
            ],
        }

        print("\n🚀 Running Agent Orchestrator test...")
        orchestrator = AgentOrchestrator()
        workflow = orchestrator.load_workflow_from_dict(workflow_dict)
        result = orchestrator.execute_workflow(workflow, {})

        print("\n📊 Test Results:")
        print(f"  Success: {result.success}")
        print(f"  Execution time: {result.execution_time:.2f}s")

        if result.step_outputs:
            for key, value in result.step_outputs.items():
                if isinstance(value, dict):
                    tools_used = value.get("tools_used", 0)
                    tool_calls = value.get("tool_calls", [])
                    tool_results = value.get("tool_results", [])

                    print(f"  Tools Used: {tools_used}")
                    print(f"  Tool Calls Made: {len(tool_calls)}")
                    print(f"  Tool Results: {len(tool_results)}")
                    print(f"  Iterations: {value.get('iterations', 'N/A')}")

                    if tool_calls:
                        print("  🔧 Tool Calls:")
                        for i, call in enumerate(tool_calls):
                            print(f"    {i+1}. {call}")

                    if tool_results:
                        print("  ✅ Tool Results:")
                        for i, result_item in enumerate(tool_results):
                            print(f"    {i+1}. {result_item}")

                    response = value.get("response", "")
                    if response:
                        print(
                            f"  Final Response: {response[:200]}{'...' if len(response) > 200 else ''}"
                        )

        # Determine success
        step_result = (
            list(result.step_outputs.values())[0] if result.step_outputs else {}
        )
        tools_used = step_result.get("tools_used", 0)

        if tools_used > 0:
            print(
                f"\n🎉 SUCCESS: Tool calling is fully functional! {tools_used} tools were executed."
            )
            return True
        else:
            print("\n⚠️ ISSUE: No tools were called despite being available.")
            return False

    except Exception as e:
        print(f"❌ Error running orchestrator test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🔬 Starting comprehensive tool calling test...")
    print("=" * 60)

    # Step 1: Register tools
    print("\n1. Registering test tools...")
    if not await register_test_tools():
        return False

    # Step 2: Test tool preparation
    print("\n2. Testing tool preparation...")
    if not await test_tool_preparation():
        return False

    # Step 3: Run orchestrator test
    print("\n3. Running Agent Orchestrator test...")
    success = run_orchestrator_test()

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Tool calling is fully restored!")
    else:
        print("❌ Tests completed with issues - debugging needed")

    return success


if __name__ == "__main__":
    # Run the comprehensive test
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
