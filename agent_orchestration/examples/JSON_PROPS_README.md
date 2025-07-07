# JSON Props Extension for AI Lego Bricks

## Overview

The JSON Props extension allows you to define reusable JSON structures in a more natural way within agent configurations and prompt templates. Instead of writing complex JSON structures inline within prompts, you can define them separately and reference them with variables.

## Features

- **Clean Structure Definition**: Define JSON structures separately from prompt text
- **Variable Substitution**: Use Jinja2 templating within JSON structures
- **Reusable Templates**: Define once, reference multiple times
- **Natural Prompt Writing**: Keep prompts readable by referencing JSON props instead of inlining complex structures

## Usage in Agent Configurations

### Basic JSON Props in Steps

Add a `json_props` section to any step configuration:

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

### Available Template Variables

When you define a JSON prop named `example_format`, these variables become available:

- `{{json_example_format}}` - The rendered JSON as a formatted string
- `{{json_example_format_dict}}` - The rendered JSON as a dictionary (for structured LLM steps)

### For Structured LLM Steps

For `llm_structured` steps, you can use JSON props to define the response schema:

```json
{
  "id": "structured_step",
  "type": "llm_structured",
  "json_props": {
    "response_schema": {
      "structure": {
        "analysis": "{{analysis_text}}",
        "confidence": "{{confidence_score}}",
        "recommendations": ["{{rec1}}", "{{rec2}}"]
      }
    }
  },
  "config": {
    "response_schema": "{{json_response_schema_dict}}"
  }
}
```

## Usage in Prompt Templates

You can also use JSON props directly in prompt template definitions:

```python
from prompt.prompt_models import PromptTemplate, JsonStructure

template = PromptTemplate(
    template="Analyze the data and respond using: {{json_analysis_format}}",
    json_props={
        "analysis_format": JsonStructure(
            structure={
                "summary": "{{summary_text}}",
                "score": "{{confidence_score}}"
            }
        )
    }
)
```

## Example Agent

See `json_props_demo_agent.json` for a complete example that demonstrates:

- User profile analysis with structured JSON responses
- Multiple JSON props in a single step
- Variable substitution within JSON structures
- Integration with both `llm_chat` and `llm_structured` steps

## Variable Substitution

JSON props support Jinja2 templating with:

- **Default Variables**: Set in the `variables` field
- **Required Variables**: Listed in `required_variables` field  
- **Context Variables**: Passed from step inputs and global context
- **Nested Substitution**: Variables can reference other variables

## Benefits

1. **Cleaner Prompts**: Keep complex JSON structures separate from prompt text
2. **Reusability**: Define structures once, use multiple times
3. **Maintainability**: Update JSON structures without touching prompt text
4. **Type Safety**: JSON validation ensures structures are valid
5. **Variable Support**: Dynamic content through template variables

## Migration from Inline JSON

**Before** (inline JSON in prompts):
```json
{
  "inputs": {
    "message": "Respond using this format: {\"name\": \"value\", \"score\": 0.95, \"items\": [\"item1\", \"item2\"]}"
  }
}
```

**After** (using JSON props):
```json
{
  "json_props": {
    "response_format": {
      "structure": {
        "name": "{{user_name}}",
        "score": 0.95,
        "items": ["item1", "item2"]
      }
    }
  },
  "inputs": {
    "message": "Respond using this format: {{json_response_format}}"
  }
}
```

This extension makes agent configurations more maintainable and prompts more readable while preserving all existing functionality.