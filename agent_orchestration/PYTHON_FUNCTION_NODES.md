# Python Function Nodes

Python function nodes allow you to execute custom Python functions as part of your agent workflow. This provides maximum flexibility for data processing, calculations, and custom business logic.

## Overview

Python function nodes are defined with the step type `python_function` and can execute Python code in three different ways:

1. **Inline functions** - Python code defined directly in the workflow configuration
2. **Module functions** - Functions imported from existing Python modules
3. **Built-in functions** - Python built-in functions like `len()`, `str()`, etc.

## Configuration

### Basic Structure

```json
{
  "id": "my_function_step",
  "type": "python_function",
  "description": "Description of what this function does",
  "config": {
    "function": {
      // Function configuration goes here
    }
  },
  "inputs": {
    // Input parameters for the function
  },
  "outputs": ["result"]
}
```

## Function Configuration Methods

### 1. Inline Functions

Define Python code directly in the workflow:

```json
{
  "config": {
    "function": {
      "code": "def calculate_sum(numbers):\n    return {'total': sum(numbers), 'count': len(numbers)}",
      "name": "calculate_sum"
    }
  },
  "inputs": {
    "numbers": [1, 2, 3, 4, 5]
  }
}
```

**Features:**
- The function name is optional - if not provided, any single function in the code will be used
- Common imports are automatically available: `json`, `os`, `sys`, `time`, `datetime`, `re`, `math`, `random`
- Functions can import additional modules as needed

### 2. Module Functions

Reference functions from existing Python modules:

```json
{
  "config": {
    "function": {
      "module": "statistics",
      "name": "mean"
    }
  },
  "inputs": {
    "data": [1, 2, 3, 4, 5]
  }
}
```

Or using string notation:

```json
{
  "config": {
    "function": "statistics.median"
  },
  "inputs": {
    "data": [1, 2, 3, 4, 5]
  }
}
```

### 3. Built-in Functions

Use Python's built-in functions:

```json
{
  "config": {
    "function": "builtins.len"
  },
  "inputs": {
    "obj": "Hello, World!"
  }
}
```

## Input/Output Handling

### Input Parameters

Function parameters are automatically matched with the inputs:

```json
{
  "inputs": {
    "param1": "value1",
    "param2": {"from_step": "previous_step", "field": "output"}
  }
}
```

### Parameter Mapping

If input names don't match function parameters, use parameter mapping:

```json
{
  "config": {
    "function": {
      "code": "def process_text(text_content):\n    return len(text_content)",
      "name": "process_text"
    },
    "parameter_mapping": {
      "text_content": "input_text"
    }
  },
  "inputs": {
    "input_text": "Hello, World!"
  }
}
```

### Return Values

Functions can return:

1. **Simple values** - wrapped in `{"success": True, "result": value}`
2. **Dictionaries** - merged with success metadata
3. **None** - returns `{"success": True, "result": None}`

Example function returning structured data:

```python
def analyze_data(data):
    return {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "count": len(data)
    }
```

## Examples

### Example 1: Data Processing Pipeline

```json
{
  "steps": [
    {
      "id": "load_data",
      "type": "input",
      "config": {"value": {"data": [1, 5, 3, 9, 2, 8, 4, 7, 6]}},
      "outputs": ["data"]
    },
    {
      "id": "sort_data",
      "type": "python_function",
      "config": {
        "function": {
          "code": "def sort_numbers(numbers):\n    return {'sorted': sorted(numbers), 'original': numbers}",
          "name": "sort_numbers"
        }
      },
      "inputs": {
        "numbers": {"from_step": "load_data", "field": "data"}
      },
      "outputs": ["result"]
    },
    {
      "id": "calculate_stats",
      "type": "python_function",
      "config": {
        "function": {
          "code": "import statistics\n\ndef calculate_statistics(data):\n    return {\n        'mean': statistics.mean(data),\n        'median': statistics.median(data),\n        'std_dev': statistics.stdev(data) if len(data) > 1 else 0\n    }",
          "name": "calculate_statistics"
        }
      },
      "inputs": {
        "data": {"from_step": "sort_data", "field": "sorted"}
      },
      "outputs": ["stats"]
    }
  ]
}
```

### Example 2: Text Analysis

```json
{
  "id": "text_analysis",
  "type": "python_function",
  "config": {
    "function": {
      "code": "import re\n\ndef analyze_text(text):\n    words = re.findall(r'\\b\\w+\\b', text)\n    sentences = re.split(r'[.!?]+', text)\n    \n    return {\n        'word_count': len(words),\n        'sentence_count': len([s for s in sentences if s.strip()]),\n        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,\n        'unique_words': len(set(word.lower() for word in words))\n    }",
      "name": "analyze_text"
    }
  },
  "inputs": {
    "text": {"from_step": "get_input", "field": "content"}
  },
  "outputs": ["analysis"]
}
```

### Example 3: Using Built-in Functions

```json
{
  "steps": [
    {
      "id": "get_length",
      "type": "python_function",
      "config": {"function": "builtins.len"},
      "inputs": {"obj": {"from_step": "input_step", "field": "text"}},
      "outputs": ["length"]
    },
    {
      "id": "convert_to_upper",
      "type": "python_function", 
      "config": {"function": "builtins.str.upper"},
      "inputs": {"self": {"from_step": "input_step", "field": "text"}},
      "outputs": ["upper_text"]
    }
  ]
}
```

## Error Handling

When functions fail, the step returns error information:

```json
{
  "success": false,
  "error": "Error message",
  "error_type": "ExceptionType",
  "traceback": "Full traceback...",
  "execution_time_ms": 123,
  "function_config": "...",
  "inputs": "...",
  "step_id": "step_name"
}
```

## Best Practices

### 1. Function Design
- Keep functions focused and single-purpose
- Return structured data when possible
- Include error handling within functions
- Use type hints and docstrings

### 2. Performance
- Avoid computationally expensive operations in inline functions
- Consider using module functions for complex algorithms
- Cache results when appropriate

### 3. Security
- Be cautious with `exec()` and `eval()` in inline functions
- Validate inputs before processing
- Avoid exposing sensitive information in error messages

### 4. Reusability
- Create module functions for commonly used operations
- Use parameter mapping for flexible input handling
- Document function requirements and return formats

## Integration with Other Step Types

Python function nodes work seamlessly with other step types:

```json
{
  "steps": [
    {
      "id": "llm_analysis",
      "type": "llm_chat",
      "config": {"provider": "gemini"},
      "inputs": {"message": "Analyze this data: {data}"}
    },
    {
      "id": "process_llm_response",
      "type": "python_function",
      "config": {
        "function": {
          "code": "import json\n\ndef extract_insights(llm_response):\n    # Process LLM response and extract structured insights\n    return {'insights': llm_response, 'processed': True}",
          "name": "extract_insights"
        }
      },
      "inputs": {
        "llm_response": {"from_step": "llm_analysis", "field": "response"}
      }
    },
    {
      "id": "store_results",
      "type": "memory_store",
      "inputs": {
        "content": {"from_step": "process_llm_response", "field": "insights"}
      }
    }
  ]
}
```

This enables powerful workflows that combine AI capabilities with custom business logic and data processing.