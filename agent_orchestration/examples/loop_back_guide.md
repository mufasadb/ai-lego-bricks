# Enhanced Loop-Back System Guide

The agent orchestration system now supports sophisticated loop-back capabilities that allow workflows to conditionally return to previous steps with iteration tracking and context preservation.

## Key Features

### 1. Configurable Iteration Limits
- **Global limit**: Set `max_iterations` in workflow config (default: 10)
- **Per-step limit**: Set `max_iterations` on individual steps
- **Protection**: Prevents infinite loops while allowing legitimate iteration

### 2. Iteration Context Variables
Access iteration information in your workflow inputs:
- `$iteration_context.iteration_count` - Current attempt number for this step
- `$iteration_context.previous_result` - Result from the previous attempt
- `$iteration_context.iteration_history` - List of all previous results
- `$iteration_context.previous_result.field_name` - Specific field from previous result

### 3. Result Preservation
Set `preserve_previous_results: true` on steps to maintain history of all attempts.

## Basic Loop-Back Example

```json
{
  "name": "retry_until_success",
  "config": {
    "max_iterations": 5
  },
  "steps": [
    {
      "id": "attempt_task",
      "type": "llm_chat",
      "config": {
        "provider": "gemini"
      },
      "inputs": {
        "message": "Try to solve this problem. Attempt #$iteration_context.iteration_count. Previous result: $iteration_context.previous_result"
      },
      "max_iterations": 3,
      "preserve_previous_results": true,
      "outputs": ["solution"]
    },
    {
      "id": "validate_solution",
      "type": "condition",
      "config": {
        "condition_type": "llm_decision",
        "condition_prompt": "Is this solution correct and complete?",
        "route_options": ["valid", "invalid"]
      },
      "inputs": {
        "solution": {"from_step": "attempt_task", "field": "solution"}
      },
      "routes": {
        "valid": "success",
        "invalid": "attempt_task"  // Loop back for another attempt
      }
    },
    {
      "id": "success",
      "type": "output",
      "inputs": {
        "final_solution": {"from_step": "attempt_task", "field": "solution"},
        "attempts_needed": "$iteration_context.iteration_count"
      }
    }
  ]
}
```

## Advanced Example: Topic Analysis with Improvement

```json
{
  "name": "topic_analysis_with_refinement",
  "config": {
    "default_llm_provider": "gemini",
    "max_iterations": 4
  },
  "steps": [
    {
      "id": "extract_topics",
      "type": "llm_structured",
      "config": {
        "response_schema": {
          "name": "TopicsResponse",
          "fields": {
            "topics": {"type": "list"},
            "confidence": {"type": "float"},
            "reasoning": {"type": "string"}
          }
        }
      },
      "inputs": {
        "message": "Extract interesting topics from: $document_text. Attempt: $iteration_context.iteration_count. Previous topics to improve on: $iteration_context.previous_result.topics"
      },
      "max_iterations": 3,
      "preserve_previous_results": true,
      "outputs": ["topics", "confidence", "reasoning"]
    },
    {
      "id": "evaluate_quality",
      "type": "condition",
      "config": {
        "condition_type": "simple_comparison",
        "left_value": {"from_step": "extract_topics", "field": "confidence"},
        "operator": ">=",
        "right_value": 0.8
      },
      "routes": {
        "true": "final_output",
        "false": "extract_topics"
      }
    },
    {
      "id": "final_output",
      "type": "output",
      "inputs": {
        "topics": {"from_step": "extract_topics", "field": "topics"},
        "final_confidence": {"from_step": "extract_topics", "field": "confidence"},
        "iterations_required": "$iteration_context.iteration_count"
      }
    }
  ]
}
```

## Benefits of This Approach

### 1. Simplicity
- Uses existing conditional routing system
- No complex loop constructs needed
- Easy to understand and debug

### 2. Flexibility
- Can loop back to any step, not just the previous one
- Multiple conditional paths possible
- Can combine with other workflow patterns

### 3. Safety
- Built-in infinite loop protection
- Configurable iteration limits
- Clear error messages when limits exceeded

### 4. Context Preservation
- Access to previous attempt results
- Iteration counting for progress tracking
- Optional result history for analysis

## Best Practices

1. **Set Reasonable Limits**: Always configure `max_iterations` to prevent runaway workflows
2. **Use Context Wisely**: Reference previous results to improve subsequent attempts
3. **Clear Exit Conditions**: Ensure conditional steps have clear success criteria
4. **Preserve Important Results**: Use `preserve_previous_results` when you need iteration history
5. **Error Handling**: Plan for max iteration scenarios in your workflow design

## Error Handling

When max iterations are reached, the workflow will fail with a clear error message:
```
RuntimeError: Maximum iterations (3) exceeded for step: extract_topics
```

This allows you to:
- Adjust iteration limits if needed
- Implement fallback logic
- Analyze why convergence wasn't achieved

## Migration from Traditional Loops

Instead of complex loop constructs, simply:
1. Add conditional steps that evaluate success criteria
2. Use routes to point back to earlier steps
3. Configure max_iterations to prevent infinite loops
4. Access iteration context for continuous improvement

This approach is more intuitive and leverages the existing workflow structure while providing powerful iterative capabilities.