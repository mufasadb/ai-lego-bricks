# Agent Orchestration System - Complete Overview

## Latest Updates & Capabilities

### Recent Major Enhancements (Latest Implementation)

We've significantly enhanced the agent orchestration system with three major capabilities:

1. **Clean LLM Architecture** - Separated Generation vs Conversation services for optimal performance
2. **Structured LLM Responses** - Guaranteed, validated responses using Pydantic schemas  
3. **Conditional Workflows** - Intelligent branching and routing based on LLM decisions or comparisons

These features work together to create a powerful, reliable agent orchestration platform.

---

## 🔧 Clean LLM Architecture: Generation vs Conversation

### What It Solves
- **Mixed responsibilities**: Previous chat service handled both one-shot and multi-turn use cases
- **Performance overhead**: Conversation management even for simple generation tasks  
- **Limited conversation access**: No way for agents to reference conversation state
- **Unclear service selection**: Developers had to manually manage conversation history

### New Architecture

#### 1. Generation Service (Stateless)
**Purpose**: One-shot prompt → response interactions optimized for speed
- **Use cases**: Document analysis, classification, transformation tasks
- **Performance**: No conversation overhead, optimized for single interactions
- **Methods**: `generate()`, `generate_with_system_prompt()`, `batch_generate()`

```python
from llm.generation_service import GenerationService, quick_generate_gemini

# Direct usage
gen_service = GenerationService(LLMProvider.GEMINI, temperature=0.7)
response = gen_service.generate("Analyze this document: [content]")

# Quick usage  
response = quick_generate_gemini("What is 2+2?")
```

#### 2. Conversation Service (Stateful)
**Purpose**: Rich multi-turn conversations with full state management
- **Use cases**: Interactive agents, context-aware dialogues, complex conversations
- **Features**: Full conversation tracking, search, export, statistics
- **Rich Access**: Agent orchestrator can reference any conversation element

```python
from chat.conversation_service import ConversationService, create_gemini_conversation

# Create conversation
conv = create_gemini_conversation(temperature=0.7)
conv.add_system_message("You are a helpful assistant")

# Multi-turn conversation
response1 = conv.send_message("What is Python?") 
response2 = conv.send_message("How do I create a list?")

# Rich conversation access
first_prompt = conv.get_first_prompt()
last_response = conv.get_last_response()  
total_messages = conv.get_conversation_length()
conversation_summary = conv.get_conversation_summary()
```

### Agent Orchestrator Integration

The orchestrator automatically selects the appropriate service based on workflow configuration:

```json
{
  "id": "one_shot_analysis", 
  "type": "llm_chat",
  "config": {
    "provider": "gemini",
    "use_conversation": false,  // Uses generation service
    "system_message": "You are an analyst"
  },
  "inputs": {"message": "Analyze this data"}
}
```

```json
{
  "id": "interactive_chat",
  "type": "llm_chat", 
  "config": {
    "provider": "gemini", 
    "use_conversation": true,   // Uses conversation service
    "conversation_id": "user_session_123",
    "system_message": "You are a helpful assistant"
  },
  "inputs": {"message": "How can I help you?"}
}
```

### Rich Conversation Access for Agents

Agents can now reference any part of conversation state:

```json
{
  "id": "summarize_conversation",
  "type": "llm_chat",
  "config": {"use_conversation": false},
  "inputs": {
    "message": "Summarize this conversation: {conversation_summary}"
  }
}
```

**Available conversation references:**
- `first_prompt` - Initial user message
- `last_response` - Most recent assistant response
- `conversation_summary` - Full conversation as single string  
- `total_messages` - Message count
- `conversation_id` - Unique identifier

### Service Selection Guidelines

**Use Generation Service when:**
- Processing documents or data
- Performing analysis or classification
- Single prompt-response needed
- Performance is critical

**Use Conversation Service when:**
- Building interactive agents
- Context from previous messages matters
- Need to track conversation state
- Multi-turn dialogue required

---

## 🎯 Structured LLM Responses

### What It Solves
- **Unreliable parsing**: No more extracting data from free-text LLM responses
- **Type safety**: Guaranteed data types and validation using Pydantic
- **Consistency**: Structured responses across different LLM providers

### How It Works

#### 1. Provider-Specific Implementation
- **Gemini**: Uses native function calling API for true structured responses
- **Ollama/Anthropic**: Uses enhanced JSON schema prompting with validation
- **All providers**: Fallback mechanisms and retry logic

#### 2. Multiple Schema Definition Methods

**Predefined Schemas:**
```json
{
  "type": "llm_structured",
  "config": {
    "response_schema": "classification"  // Built-in schema
  }
}
```

**Dictionary-Based Schemas:**
```json
{
  "response_schema": {
    "name": "ProductAnalysis",
    "fields": {
      "price": {"type": "float", "description": "Product price"},
      "quality": {"type": "string", "description": "Quality rating"},
      "recommended": {"type": "boolean", "description": "Recommendation"}
    }
  }
}
```

**Custom Pydantic Classes:**
```python
class MovieInfo(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating 1-10", ge=1, le=10)
    genre: str = Field(description="Primary genre")

# Reference in workflow: "response_schema": "mymodule.MovieInfo"
```

#### 3. Available Predefined Schemas
- `simple_response`: Basic response with confidence
- `classification`: Category classification with confidence and reasoning
- `extraction`: Data extraction with confidence and source
- `decision`: Decision making with reasoning and alternatives
- `summary`: Text summarization with key points

### Usage Examples

#### Basic Structured Response
```json
{
  "id": "analyze_content",
  "type": "llm_structured",
  "config": {
    "provider": "gemini",
    "response_schema": "classification"
  },
  "inputs": {
    "message": "Classify this content as technical, business, or creative"
  },
  "outputs": ["category", "confidence", "reasoning"]
}
```

#### Multi-Step Structured Workflow
```json
{
  "steps": [
    {
      "id": "extract_data",
      "type": "llm_structured",
      "config": {
        "response_schema": "extraction"
      }
    },
    {
      "id": "make_decision",
      "type": "llm_structured", 
      "config": {
        "response_schema": "decision"
      },
      "inputs": {
        "message": {
          "from_step": "extract_data",
          "field": "extracted_data"
        }
      }
    }
  ]
}
```

---

## 🔀 Conditional Workflows

### What It Enables
- **Intelligent routing**: LLM makes context-aware decisions about workflow paths
- **Dynamic execution**: Workflows adapt based on content and conditions
- **Complex logic**: Multi-level decision trees and branching workflows

### How It Works

#### 1. Conditional Step Types

**LLM-Based Decisions:**
```json
{
  "id": "routing_decision",
  "type": "condition",
  "config": {
    "condition_type": "llm_decision",
    "provider": "gemini",
    "condition_prompt": "Should this be processed as urgent, normal, or low priority?",
    "route_options": ["urgent", "normal", "low_priority"]
  },
  "routes": {
    "urgent": "urgent_processing",
    "normal": "standard_processing", 
    "low_priority": "batch_processing"
  }
}
```

**Simple Comparisons:**
```json
{
  "id": "threshold_check",
  "type": "condition",
  "config": {
    "condition_type": "simple_comparison",
    "left_value": {"from_step": "evaluation", "field": "score"},
    "operator": ">=",
    "right_value": 8.0
  },
  "routes": {
    "true": "high_quality_processing",
    "false": "improvement_needed"
  }
}
```

#### 2. Supported Operators
- `==`, `!=`: Equality comparisons
- `>`, `<`, `>=`, `<=`: Numeric comparisons  
- `contains`: String/list containment

#### 3. Reference Resolution
- **Step outputs**: `{"from_step": "step_id", "field": "field_name"}`
- **Global variables**: `"$variable_name"`
- **Direct values**: Any literal value

### Advanced Examples

#### Content Classification Router
```json
{
  "name": "intelligent_content_processor",
  "steps": [
    {
      "id": "classify_content",
      "type": "condition",
      "config": {
        "condition_type": "llm_decision",
        "condition_prompt": "Classify this content: technical_document, creative_writing, business_report, or customer_feedback",
        "route_options": ["technical_document", "creative_writing", "business_report", "customer_feedback"]
      },
      "routes": {
        "technical_document": "technical_analysis",
        "creative_writing": "creative_analysis",
        "business_report": "business_analysis", 
        "customer_feedback": "feedback_analysis"
      }
    },
    {
      "id": "technical_analysis",
      "type": "llm_structured",
      "config": {
        "response_schema": {
          "name": "TechnicalAnalysis",
          "fields": {
            "complexity_level": {"type": "string", "description": "beginner, intermediate, advanced"},
            "key_technologies": {"type": "list", "description": "Technologies mentioned"},
            "accuracy_score": {"type": "float", "description": "Technical accuracy 0-10"}
          }
        }
      }
    }
  ]
}
```

#### Multi-Level Decision Tree
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
        "public": "public_processing",
        "internal": "internal_check",
        "confidential": "security_evaluation"
      }
    },
    {
      "id": "security_evaluation", 
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

---

## 🔄 Combined Usage Patterns

### Structured Decision Making
Combine both features for powerful decision workflows:

```json
{
  "id": "intelligent_quality_gate",
  "type": "llm_structured",
  "config": {
    "provider": "gemini",
    "response_schema": {
      "name": "QualityEvaluation", 
      "fields": {
        "quality_score": {"type": "float", "description": "Score 0-10"},
        "meets_standards": {"type": "boolean", "description": "Meets quality standards"},
        "improvement_areas": {"type": "list", "description": "Areas needing improvement"},
        "recommendation": {"type": "string", "description": "approve, revise, or reject"}
      }
    }
  },
  "routes": {
    "approve": "approval_processing",
    "revise": "revision_workflow", 
    "reject": "rejection_processing"
  }
}
```

### Adaptive Processing Pipeline
```json
{
  "steps": [
    {
      "id": "content_analysis",
      "type": "llm_structured",
      "config": {
        "response_schema": "classification"
      }
    },
    {
      "id": "processing_decision",
      "type": "condition", 
      "config": {
        "condition_type": "simple_comparison",
        "left_value": {"from_step": "content_analysis", "field": "confidence"},
        "operator": ">=",
        "right_value": 0.8
      },
      "routes": {
        "true": "high_confidence_processing",
        "false": "manual_review_required"
      }
    },
    {
      "id": "high_confidence_processing",
      "type": "llm_structured",
      "config": {
        "response_schema": "summary"
      },
      "inputs": {
        "message": "Process this high-confidence classification result"
      }
    }
  ]
}
```

---

## 🛠️ Implementation Architecture

### Core Components

#### 1. Structured Response Infrastructure (`llm/`)
- **StructuredLLMWrapper**: Main wrapper ensuring structured responses
- **Enhanced LLM Clients**: All clients support `chat_structured()` methods
- **LLMClientFactory**: Creates structured clients with schema validation
- **Schema Utilities**: Convert Pydantic to JSON schema and function calling format

#### 2. Conditional Flow System (`agent_orchestration/`)
- **Condition Step Handler**: Handles both LLM and simple comparison decisions
- **Enhanced Executor**: Non-linear execution with conditional routing
- **Route Mapping**: Maps condition results to target step IDs
- **Reference Resolution**: Resolves step outputs and variables in conditions

#### 3. Error Handling & Reliability
- **Graceful Fallbacks**: Default routes when conditions fail
- **Retry Logic**: Automatic retries for failed LLM calls
- **Validation**: Pydantic validation with detailed error messages
- **Loop Protection**: Prevents infinite execution loops

### Key Benefits

#### For Developers
- **Type Safety**: Guaranteed data types from LLM responses
- **Intelligent Logic**: LLM-powered decision making in workflows
- **Easy Integration**: Works with existing orchestration system
- **Flexible Configuration**: Multiple ways to define schemas and conditions

#### For Production Use
- **Reliability**: Robust error handling and fallback mechanisms
- **Performance**: Optimized for different LLM providers
- **Scalability**: Easy to add new schemas, conditions, and routes
- **Maintainability**: Clear separation of concerns and modular design

---

## 📁 File Structure & Examples

### Key Implementation Files
```
agent_orchestration/
├── step_handlers.py        # Conditional logic & structured response handling
├── orchestrator.py         # Enhanced executor with routing
├── models.py              # Schema definitions with routing support

llm/
├── llm_types.py           # Structured response protocols
├── text_clients.py        # Enhanced clients with structured support
├── llm_factory.py         # Structured client factory methods

examples/
├── structured_response_example.py     # Structured response demos
├── conditional_workflow_example.py    # Conditional routing demos
├── test_structured_llm_clients.py    # Client-level tests
└── test_conditional_workflows.py     # Workflow-level tests
```

### Example Workflows
```
examples/
├── document_analysis_agent.json      # Real-world document processing
├── content_classification.json       # Multi-path content routing
└── quality_control_pipeline.json     # Structured decision making
```

---

## 🚀 Getting Started

### 1. Simple Structured Response
```python
from agent_orchestration import AgentOrchestrator

workflow = {
  "name": "my_first_structured_workflow",
  "steps": [{
    "id": "analyze",
    "type": "llm_structured", 
    "config": {
      "provider": "gemini",
      "response_schema": "classification"
    },
    "inputs": {
      "message": "Classify this text: 'Machine learning revolutionizes data analysis'"
    }
  }]
}

orchestrator = AgentOrchestrator()
result = orchestrator.execute_workflow(orchestrator.load_workflow_from_dict(workflow))
print(result.step_outputs["analyze"]["category"])  # "technical"
```

### 2. Simple Conditional Routing
```python
workflow = {
  "name": "conditional_example",
  "steps": [
    {
      "id": "evaluate",
      "type": "llm_structured",
      "config": {"response_schema": "simple_response"}
    },
    {
      "id": "route_decision", 
      "type": "condition",
      "config": {
        "condition_type": "llm_decision",
        "condition_prompt": "Route this to: process_further or complete",
        "route_options": ["process_further", "complete"]
      },
      "routes": {
        "process_further": "additional_processing",
        "complete": "final_output"
      }
    }
  ]
}
```

This enhanced agent orchestration system provides a robust foundation for building intelligent, adaptive AI workflows that can make decisions, validate responses, and route dynamically based on content analysis and business logic.