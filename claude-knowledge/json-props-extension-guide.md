# JSON Props Extension

## Overview

The JSON Props extension allows natural definition of JSON structures in agent configurations and prompt templates, making it easier to work with structured data without cluttering prompts with complex inline JSON.

## Core Components

### JsonStructure Class (`prompt/prompt_models.py`)

Handles JSON structure definitions with Jinja2 template support:

```python
class JsonStructure(BaseModel):
    structure: Dict[str, Any]
    description: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    required_variables: List[str] = Field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render JSON structure as formatted string with variable substitution"""
    
    def render_as_dict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Render JSON structure as dictionary with variable substitution"""
```

### Enhanced PromptTemplate

Added `json_props` field to `PromptTemplate` class:

```python
class PromptTemplate(BaseModel):
    # ... existing fields ...
    json_props: Dict[str, JsonStructure] = Field(default_factory=dict)
    
    def get_json_prop(self, prop_name: str, context: Dict[str, Any] = None) -> str:
        """Get a specific JSON prop rendered as a string"""
    
    def get_json_prop_dict(self, prop_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get a specific JSON prop rendered as a dictionary"""
```

### Enhanced StepConfig

Added `json_props` field to `StepConfig` class:

```python
class StepConfig(BaseModel):
    # ... existing fields ...
    json_props: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
```

### Step Handler Integration

Modified LLM step handlers to automatically process JSON props:

- `_handle_llm_chat()` - Processes JSON props before handling message
- `_handle_llm_structured()` - Processes JSON props before structured response
- `_process_json_props()` - Core processing function that creates template variables

## Usage Patterns

### In Agent Configurations

```json
{
  "id": "example_step",
  "type": "llm_chat",
  "json_props": {
    "user_profile": {
      "description": "Structure for user profile data",
      "structure": {
        "name": "{{user_name}}",
        "age": "{{user_age}}",
        "interests": ["{{interest1}}", "{{interest2}}"]
      },
      "variables": {
        "interest1": "reading"
      },
      "required_variables": ["user_name", "user_age"]
    }
  },
  "inputs": {
    "message": "Create a user profile using this format: {{json_user_profile}}"
  }
}
```

### Template Variables Created

For a JSON prop named `example_format`:
- `{{json_example_format}}` - Rendered JSON as formatted string
- `{{json_example_format_dict}}` - Rendered JSON as dictionary (for structured steps)

### For Structured LLM Steps

```json
{
  "id": "structured_step",
  "type": "llm_structured",
  "json_props": {
    "response_schema": {
      "structure": {
        "analysis": "{{analysis_text}}",
        "confidence": "{{confidence_score}}"
      }
    }
  },
  "config": {
    "response_schema": "{{json_response_schema_dict}}"
  }
}
```

## Benefits

1. **Clean Separation**: JSON structures defined separately from prompt text
2. **Variable Substitution**: Full Jinja2 template support within structures
3. **Reusability**: Define once, reference multiple times
4. **Maintainability**: Update structures without touching prompts
5. **Readability**: Prompts remain clean and focused
6. **Type Safety**: JSON validation ensures valid structures

## Migration Pattern

**Before** (inline JSON):
```json
{
  "inputs": {
    "message": "Respond using: {\"name\": \"{{user_name}}\", \"score\": 0.95}"
  }
}
```

**After** (JSON props):
```json
{
  "json_props": {
    "response_format": {
      "structure": {
        "name": "{{user_name}}",
        "score": 0.95
      }
    }
  },
  "inputs": {
    "message": "Respond using: {{json_response_format}}"
  }
}
```

## Implementation Details

### Variable Resolution Order

1. JSON prop's default variables
2. Step inputs
3. Global context variables
4. Runtime context

### Template Processing

1. JSON structure converted to JSON string
2. Jinja2 template applied to JSON string
3. Result parsed back to JSON for validation
4. Both string and dict versions made available

### Error Handling

- Missing required variables raise `ValueError`
- Invalid JSON after templating raises `ValueError`
- Graceful fallback for missing JSON props

## Example Agents

- `json_props_demo_agent.json` - Comprehensive demonstration
- Shows integration with both `llm_chat` and `llm_structured` steps
- Demonstrates variable substitution and nested structures

This extension maintains full backwards compatibility while providing a much more natural way to work with JSON structures in agent configurations.