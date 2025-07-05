# Conditional Workflows Implementation

## Overview

We've successfully implemented conditional flow control for the agent orchestration system. This enables workflows to branch and route to different steps based on intelligent decisions or simple comparisons, making the orchestration much more dynamic and powerful.

## Key Features Implemented

### 1. Conditional Step Type (`condition`)

#### LLM-Based Conditional Evaluation
- **Intelligent routing**: Uses LLM to make decisions based on content analysis
- **Structured responses**: Returns decision, reasoning, and confidence score
- **Multiple route options**: Support for any number of routing paths
- **Fallback handling**: Graceful error handling with default routes

#### Simple Comparison Evaluation  
- **Direct comparisons**: `==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`
- **Variable references**: Support for step outputs and global variables
- **Boolean results**: Returns `true`/`false` for simple routing decisions

### 2. Enhanced Workflow Execution

#### Non-Linear Execution
- **Step jumping**: Can jump to any step by ID based on conditions
- **Route mapping**: Maps condition results to target step IDs
- **Loop protection**: Prevents infinite loops with step tracking
- **Flexible termination**: Support for early workflow termination

#### Route Configuration
```json
{
  "routes": {
    "condition_result_1": "target_step_id_1",
    "condition_result_2": "target_step_id_2", 
    "default": "fallback_step_id"
  }
}
```

### 3. Configuration Options

#### LLM Conditional Step
```json
{
  "id": "decision_step",
  "type": "condition",
  "config": {
    "condition_type": "llm_decision",
    "provider": "gemini",
    "condition_prompt": "Analyze this content and decide...",
    "route_options": ["option1", "option2", "option3"],
    "default_route": "option1",
    "temperature": 0.3
  },
  "routes": {
    "option1": "step_for_option1",
    "option2": "step_for_option2", 
    "option3": "step_for_option3"
  }
}
```

#### Simple Comparison Step
```json
{
  "id": "threshold_check",
  "type": "condition", 
  "config": {
    "condition_type": "simple_comparison",
    "left_value": {"from_step": "previous_step", "field": "score"},
    "operator": ">=",
    "right_value": 7.5
  },
  "routes": {
    "true": "high_score_processing",
    "false": "low_score_processing"
  }
}
```

## Implementation Details

### 1. Step Handler (`_handle_condition`)

**LLM Decision Making**:
- Creates dynamic schema for available route options
- Uses structured LLM responses for consistent decision format
- Includes contextual information from previous steps
- Validates decisions against allowed options

**Simple Comparisons**:
- Supports variable references with `$variable_name` syntax
- Resolves values from step outputs and global variables  
- Comprehensive operator support for different data types

### 2. Workflow Executor Updates

**Conditional Routing Logic**:
- Modified from sequential execution to conditional branching
- `_get_next_step()` method determines routing based on results
- Step lookup by ID for jumping to target steps
- Support for special keywords (`end`, `exit`, `terminate`)

**Execution Flow**:
1. Execute current step
2. Check for conditional routing in step configuration
3. Evaluate step result against route mapping
4. Jump to target step or continue sequentially
5. Track executed steps to prevent infinite loops

### 3. Enhanced Models

**StepConfig Extensions**:
- Added `routes` field for conditional routing configuration
- Maps condition results to target step IDs
- Supports both exact matches and default fallbacks

## Usage Examples

### 1. Content Classification Routing
```json
{
  "id": "classify_content",
  "type": "condition",
  "config": {
    "condition_type": "llm_decision",
    "condition_prompt": "Classify this content as: technical, business, or creative",
    "route_options": ["technical", "business", "creative"]
  },
  "routes": {
    "technical": "technical_analysis",
    "business": "business_analysis", 
    "creative": "creative_analysis"
  }
}
```

### 2. Quality Score Routing
```json
{
  "id": "quality_check",
  "type": "condition",
  "config": {
    "condition_type": "simple_comparison",
    "left_value": {"from_step": "evaluate", "field": "quality_score"},
    "operator": ">=", 
    "right_value": 8.0
  },
  "routes": {
    "true": "publish_content",
    "false": "request_improvements"
  }
}
```

### 3. Multi-Level Decision Tree
```json
{
  "steps": [
    {
      "id": "security_classification",
      "type": "condition",
      "config": {
        "condition_type": "llm_decision",
        "route_options": ["public", "internal", "confidential"]
      },
      "routes": {
        "public": "public_review",
        "internal": "internal_processing",
        "confidential": "security_check"
      }
    },
    {
      "id": "security_check", 
      "type": "condition",
      "config": {
        "condition_type": "llm_decision",
        "route_options": ["restricted", "secret", "top_secret"]
      },
      "routes": {
        "restricted": "restricted_processing",
        "secret": "secret_processing",
        "top_secret": "top_secret_processing"
      }
    }
  ]
}
```

## Benefits

### 1. Intelligent Workflow Control
- **Smart routing**: LLM makes context-aware decisions
- **Flexible branching**: Support for complex decision trees
- **Dynamic execution**: Workflows adapt based on content and conditions

### 2. Enhanced Automation
- **Reduced manual intervention**: Automated decision making
- **Context awareness**: Decisions based on previous step outputs
- **Scalable logic**: Easy to add new routes and conditions

### 3. Robust Error Handling
- **Graceful fallbacks**: Default routes when conditions fail
- **Loop protection**: Prevents infinite execution loops
- **Validation**: Ensures decisions match allowed options

## Integration with Structured Responses

The conditional system leverages the structured response implementation:

- **Consistent decision format**: All LLM decisions return structured responses
- **Validation**: Decision results are validated against allowed options
- **Reasoning**: Each decision includes explanation and confidence score
- **Error recovery**: Fallback to default routes on parsing errors

## Testing & Examples

### Example Files Created
- `examples/conditional_workflow_example.py`: Comprehensive examples
- Demonstrates LLM routing, simple comparisons, and multi-conditional workflows

### Test Scenarios Covered
- ✅ LLM-based content classification routing
- ✅ Simple numeric threshold comparisons  
- ✅ Multi-level decision trees
- ✅ Error handling and fallback routing
- ✅ Complex branching workflows
- ✅ Route validation and loop protection

## Architecture Notes

### Design Principles
- **Backward compatibility**: Existing sequential workflows continue to work
- **Flexible routing**: Multiple ways to define conditional logic
- **Provider agnostic**: Works with any LLM provider that supports structured responses
- **Extensible**: Easy to add new condition types and operators

### Key Classes
- `StepHandlerRegistry`: Extended with `_handle_condition()` method
- `WorkflowExecutor`: Modified for non-linear execution with `_get_next_step()`
- `StepConfig`: Enhanced with `routes` field for routing configuration

This implementation provides a powerful foundation for creating intelligent, adaptive agent workflows that can make decisions and route dynamically based on content analysis and business logic.