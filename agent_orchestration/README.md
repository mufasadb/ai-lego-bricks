# Agent Orchestration System

A JSON-driven system for orchestrating AI agents using existing building blocks like LLM clients, memory services, document processing, and chunking services.

## Overview

This system allows you to create sophisticated AI agents by combining different services in a declarative, configuration-driven manner. Instead of writing code for each agent, you define workflows in JSON that specify how to chain together different operations.

## Key Features

- **JSON-driven workflows**: Define agent behavior through configuration
- **Modular architecture**: Uses existing services as building blocks
- **Streaming support**: Real-time LLM response streaming
- **Text-to-speech integration**: Convert text responses to audio
- **Flexible orchestration**: Support for conditional logic and loops
- **Memory integration**: Persistent storage with semantic search
- **Multi-modal support**: Text, vision, and document processing
- **Step-by-step execution**: Clear data flow between operations
- **Prompt Management**: Externalized, versioned prompts with evaluation and A/B testing
- **Human-in-the-Loop**: Interactive approval and feedback collection

## Building Blocks

The system orchestrates these existing services:

### LLM Services
- **Generation Service**: Stateless one-shot LLM interactions (optimized for speed)
- **Conversation Service**: Rich multi-turn conversations with full state management
- **Streaming Support**: Real-time response streaming (Ollama, Anthropic)
- **Structured Responses**: Type-safe LLM outputs with Pydantic validation
- **Vision Analysis**: Image analysis with Gemini Vision and LLaVA
- **Embeddings**: Text embedding generation

### Audio Services
- **Text-to-Speech**: Multi-provider TTS with OpenAI, Google, and Coqui-XTTS support
- **Streaming TTS**: Real-time audio generation from streaming LLM responses

### Prompt Management
- **Versioned Prompts**: Semantic versioning with lifecycle management
- **Template System**: Dynamic prompts with Jinja2 variable substitution
- **Execution Logging**: Automatic tracking for evaluation and training
- **A/B Testing**: Performance comparison between prompt versions
- **Storage Backends**: File-based and Supabase storage options

### Memory Services
- **Vector Storage**: Supabase with pgvector
- **Graph Storage**: Neo4j for relationships
- **Semantic Search**: Similarity-based retrieval

### Document Processing
- **PDF Extraction**: Traditional and LLM-enhanced
- **Text Chunking**: Semantic boundary preservation
- **Content Enhancement**: LLM-powered formatting

### Conversation Management
- **Rich State Tracking**: Access first prompt, last response, message history
- **Search & Export**: Search messages, export as JSON/markdown/text
- **Conversation Statistics**: Message counts, duration, metadata
- **Multi-provider Support**: Gemini, Ollama, Anthropic

## Quick Start

### 1. Basic Usage

```python
from agent_orchestration import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator()

# Load workflow from JSON file
workflow = orchestrator.load_workflow_from_file("my_agent.json")

# Execute with initial inputs
result = orchestrator.execute_workflow(workflow, {
    "user_query": "What is machine learning?"
})

print(result.final_output)
```

### 2. Simple Chat Agent

```json
{
  "name": "simple_chat_agent",
  "description": "A simple chat agent",
  "steps": [
    {
      "id": "get_input",
      "type": "input",
      "config": {"prompt": "What would you like to know?"},
      "outputs": ["user_query"]
    },
    {
      "id": "generate_response",
      "type": "llm_chat",
      "config": {"provider": "gemini"},
      "inputs": {
        "message": {"from_step": "get_input", "field": "user_query"}
      },
      "outputs": ["response"]
    },
    {
      "id": "output",
      "type": "output",
      "inputs": {
        "result": {"from_step": "generate_response", "field": "response"}
      }
    }
  ]
}
```

## Step Types

### Core Operations

- **`input`**: Collect user input or external data
- **`output`**: Format and return results
- **`llm_chat`**: Generate text using LLM (auto-selects generation/conversation service)
- **`llm_structured`**: Generate structured responses with Pydantic validation
- **`llm_vision`**: Analyze images

### Document Processing

- **`document_processing`**: Extract text from PDFs
- **`chunk_text`**: Break text into semantic chunks

### Memory Operations

- **`memory_store`**: Store content in vector/graph storage
- **`memory_retrieve`**: Search and retrieve relevant memories

### Control Flow

- **`condition`**: Conditional execution
- **`loop`**: Iterate over collections

### Human Interaction

- **`human_approval`**: Human-in-the-loop approval and feedback collection

## LLM Architecture: Generation vs Conversation

The system uses two distinct services for different LLM interaction patterns:

### Generation Service (One-Shot)
- **Use Case**: Single prompt → response interactions
- **Optimized For**: Speed, simplicity, stateless operations
- **Best For**: Analysis, classification, transformation tasks
- **Configuration**: Set `use_conversation: false` in llm_chat steps

```json
{
  "id": "analyze_text",
  "type": "llm_chat",
  "config": {
    "provider": "gemini",
    "use_conversation": false,
    "system_message": "You are a text analyzer"
  },
  "inputs": {"message": "Analyze this document"}
}
```

### Conversation Service (Multi-Turn)
- **Use Case**: Interactive conversations with context
- **Features**: Rich state management, conversation history, message search
- **Best For**: Interactive agents, complex dialogues, context-aware responses
- **Configuration**: Set `use_conversation: true` in llm_chat steps

```json
{
  "id": "interactive_chat",
  "type": "llm_chat", 
  "config": {
    "provider": "gemini",
    "use_conversation": true,
    "conversation_id": "user_session_123",
    "system_message": "You are a helpful assistant"
  },
  "inputs": {"message": "How can I help you today?"}
}
```

## Prompt Management Integration

The system supports externalized prompt management for better maintainability, evaluation, and A/B testing.

### Using Managed Prompts

Instead of hardcoding prompts in workflow configurations, you can reference managed prompts:

```json
{
  "id": "analyze_document",
  "type": "llm_structured",
  "prompt_ref": {
    "prompt_id": "document_analysis",
    "version": "1.2.0",
    "context_variables": {
      "analysis_type": "comprehensive",
      "focus_areas": ["metrics", "trends", "insights"]
    }
  },
  "config": {
    "provider": "gemini",
    "response_schema": "classification"
  },
  "inputs": {
    "document_text": "$document_content"
  }
}
```

### Prompt Reference Configuration

- **`prompt_id`**: Unique identifier for the prompt
- **`version`**: Specific version (optional, uses latest if omitted)
- **`context_variables`**: Static variables for template rendering

### Template Variable Resolution

Variables are resolved in this order:
1. `context_variables` from prompt reference
2. Step `inputs` from workflow data flow
3. Global workflow variables

### Benefits of Managed Prompts

- **Version Control**: Track prompt evolution and rollback capability
- **A/B Testing**: Compare performance between prompt versions
- **Team Collaboration**: Non-developers can manage prompts
- **Evaluation**: Automatic performance tracking and analytics
- **Reusability**: Share prompts across multiple workflows

### Creating Managed Prompts

```python
from prompt import create_prompt_service, PromptStatus

prompt_service = create_prompt_service("auto")

# Create a template-based prompt
prompt = prompt_service.create_prompt(
    prompt_id="document_analysis",
    name="Document Analysis Expert",
    content=[
        {
            "role": "system",
            "content": "You are an expert document analyst."
        },
        {
            "role": "user",
            "content": {
                "template": """
                Analyze this {{ document_type }}:
                {{ document_text }}
                
                Focus on: {{ focus_areas | join(', ') }}
                """,
                "required_variables": ["document_type", "document_text"],
                "variables": {"focus_areas": ["main topics", "insights"]}
            }
        }
    ],
    version="1.0.0",
    status=PromptStatus.ACTIVE
)
```

### Execution Tracking

When using managed prompts, the system automatically logs:
- Execution time and token usage
- Success/failure rates
- Template context and rendered output
- LLM provider and model used

This data powers evaluation reports and A/B testing capabilities.

### Rich Conversation Access

When using conversation mode, the orchestrator can access conversation state:

```json
{
  "id": "conversation_analysis",
  "type": "llm_chat",
  "config": {"use_conversation": false},
  "inputs": {
    "message": "Summarize conversation: {conversation_summary}"
  }
}
```

Access methods available:
- `first_prompt`: Initial user message
- `last_response`: Most recent assistant response
- `conversation_summary`: Full conversation as text
- `total_messages`: Message count
- `conversation_id`: Unique conversation identifier

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

## Configuration Reference

### Workflow Structure

```json
{
  "name": "workflow_name",
  "description": "What this workflow does",
  "config": {
    "memory_backend": "auto|supabase|neo4j",
    "default_llm_provider": "gemini|ollama",
    "default_model": "model_name"
  },
  "steps": [...]
}
```

### Step Structure

```json
{
  "id": "unique_step_id",
  "type": "step_type",
  "description": "What this step does",
  "config": {
    // Step-specific configuration
  },
  "inputs": {
    "input_name": "literal_value",
    "another_input": {
      "from_step": "previous_step_id",
      "field": "output_field"
    }
  },
  "outputs": ["output1", "output2"]
}
```

### Input References

Reference outputs from previous steps:
```json
{
  "from_step": "step_id",
  "field": "output_field"
}
```

Reference global variables:
```json
"$variable_name"
```

## Examples

### Document Analysis Agent

Processes PDFs, stores content in memory, and answers questions:

```json
{
  "name": "document_analyzer",
  "steps": [
    {
      "id": "process_pdf",
      "type": "document_processing",
      "config": {"enhance_with_llm": true},
      "inputs": {"file_path": "$document_path"}
    },
    {
      "id": "chunk_text",
      "type": "chunk_text",
      "inputs": {"text": {"from_step": "process_pdf", "field": "text"}}
    },
    {
      "id": "store_chunks",
      "type": "memory_store",
      "inputs": {"content": {"from_step": "chunk_text", "field": "chunks"}}
    },
    {
      "id": "get_question",
      "type": "input",
      "config": {"prompt": "What's your question about the document?"}
    },
    {
      "id": "search_memory",
      "type": "memory_retrieve",
      "inputs": {"query": {"from_step": "get_question", "field": "question"}}
    },
    {
      "id": "generate_answer",
      "type": "llm_chat",
      "config": {"provider": "gemini"},
      "inputs": {
        "message": "Context: {context}\n\nQuestion: {question}"
      }
    },
    {
      "id": "output_answer",
      "type": "output",
      "inputs": {"answer": {"from_step": "generate_answer", "field": "response"}}
    }
  ]
}
```

### Research Agent

Processes multiple documents and provides comprehensive analysis:

```json
{
  "name": "research_agent",
  "steps": [
    {
      "id": "process_documents",
      "type": "document_processing",
      "config": {"enhance_with_llm": true},
      "inputs": {"file_path": "$document_paths"}
    },
    {
      "id": "extract_concepts",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Extract key concepts from: {documents}"
      }
    },
    {
      "id": "store_research",
      "type": "memory_store",
      "inputs": {
        "content": {"from_step": "process_documents", "field": "text"},
        "metadata": {"from_step": "extract_concepts", "field": "concepts"}
      }
    },
    {
      "id": "research_query",
      "type": "input",
      "config": {"prompt": "What research question should I investigate?"}
    },
    {
      "id": "search_knowledge",
      "type": "memory_retrieve",
      "config": {"limit": 5, "threshold": 0.6},
      "inputs": {"query": {"from_step": "research_query", "field": "query"}}
    },
    {
      "id": "synthesize_findings",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Synthesize research findings for: {query}\n\nFindings: {findings}"
      }
    },
    {
      "id": "format_report",
      "type": "output",
      "config": {"format": "text"},
      "inputs": {"report": {"from_step": "synthesize_findings", "field": "response"}}
    }
  ]
}
```

## Advanced Features

### Conditional Execution

```json
{
  "id": "check_condition",
  "type": "condition",
  "condition": {
    "if": "input_value > 0",
    "then": {
      "id": "positive_path",
      "type": "llm_chat",
      "inputs": {"message": "Process positive value"}
    },
    "else": {
      "id": "negative_path", 
      "type": "llm_chat",
      "inputs": {"message": "Process negative value"}
    }
  }
}
```

### Loop Processing

```json
{
  "id": "process_documents",
  "type": "loop",
  "loop": {
    "over": "document_list",
    "body": {
      "id": "process_single_doc",
      "type": "document_processing",
      "inputs": {"file_path": "$current_document"}
    }
  }
}
```

## Human-in-the-Loop Workflows

The `human_approval` step type enables human oversight and feedback collection during workflow execution. This allows for hybrid human-AI workflows where critical decisions require human input.

### Human Approval Step Types

#### 1. Approve/Reject
Simple binary decision making:

```json
{
  "id": "review_analysis",
  "type": "human_approval",
  "description": "Human review of automated analysis",
  "config": {
    "approval_type": "approve_reject",
    "prompt": "Please review the analysis below. Do you approve?",
    "timeout_seconds": 180,
    "default_action": "reject",
    "show_context": true,
    "context_fields": ["analysis", "confidence"]
  },
  "inputs": {
    "analysis": {"from_step": "ai_analysis", "field": "summary"},
    "confidence": {"from_step": "ai_analysis", "field": "confidence"}
  },
  "outputs": ["decision", "feedback"],
  "routes": {
    "approve": "next_step",
    "reject": "error_handler"
  }
}
```

#### 2. Multiple Choice
Selection from predefined options:

```json
{
  "id": "choose_action",
  "type": "human_approval",
  "config": {
    "approval_type": "multiple_choice",
    "prompt": "What action should we take?",
    "options": ["continue", "revise", "abort", "enhance"],
    "timeout_seconds": 300,
    "default_action": "continue"
  },
  "routes": {
    "continue": "continue_processing",
    "revise": "revision_step", 
    "abort": "cleanup_step",
    "enhance": "enhancement_step"
  }
}
```

#### 3. Custom Input
Free-form text input for detailed feedback:

```json
{
  "id": "get_feedback",
  "type": "human_approval",
  "config": {
    "approval_type": "custom_input",
    "prompt": "Please provide specific feedback on how to improve this analysis:",
    "timeout_seconds": 600,
    "default_action": "Use original analysis"
  },
  "outputs": ["user_input", "feedback"]
}
```

### Configuration Options

- **`approval_type`**: `"approve_reject"`, `"multiple_choice"`, or `"custom_input"`
- **`prompt`**: Message displayed to the human reviewer
- **`options`**: Available choices for multiple_choice type
- **`timeout_seconds`**: Time limit before using default action
- **`default_action`**: Fallback when timeout occurs or input is interrupted
- **`show_context`**: Whether to display relevant context information
- **`context_fields`**: Specific fields to show from inputs and previous steps

### Human Input Data Flow

Human responses become available to subsequent steps through the standard input reference system:

```json
{
  "id": "process_feedback",
  "type": "llm_chat",
  "inputs": {
    "message": "Original analysis: {original}\n\nHuman feedback: {human_feedback}\n\nPlease revise accordingly."
  }
}
```

Where `human_feedback` references:
```json
{
  "from_step": "human_review",
  "field": "user_input"
}
```

### Complete Example: Document Analysis with Human Review

```json
{
  "name": "human_reviewed_analysis",
  "description": "Document analysis with human oversight",
  "steps": [
    {
      "id": "analyze_document",
      "type": "llm_chat",
      "config": {"provider": "gemini"},
      "inputs": {
        "message": "Analyze this document: {document_text}"
      },
      "outputs": ["analysis"]
    },
    {
      "id": "human_review", 
      "type": "human_approval",
      "config": {
        "approval_type": "multiple_choice",
        "prompt": "Review the AI analysis. What should we do?",
        "options": ["approve", "enhance", "revise"],
        "show_context": true,
        "context_fields": ["analysis"]
      },
      "inputs": {
        "ai_analysis": {"from_step": "analyze_document", "field": "analysis"}
      },
      "routes": {
        "approve": "final_output",
        "enhance": "enhance_analysis", 
        "revise": "get_revision_feedback"
      }
    },
    {
      "id": "get_revision_feedback",
      "type": "human_approval",
      "config": {
        "approval_type": "custom_input",
        "prompt": "How should the analysis be revised?"
      },
      "outputs": ["user_input"]
    },
    {
      "id": "revise_analysis",
      "type": "llm_chat",
      "config": {"provider": "gemini"},
      "inputs": {
        "message": "Original: {original}\nFeedback: {feedback}\nPlease revise."
      },
      "routes": {"default": "final_output"}
    },
    {
      "id": "enhance_analysis",
      "type": "llm_chat", 
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Provide a more detailed analysis: {original_analysis}"
      },
      "routes": {"default": "final_output"}
    },
    {
      "id": "final_output",
      "type": "output",
      "inputs": {
        "result": "Analysis completed with human oversight"
      }
    }
  ]
}
```

### Best Practices

1. **Clear Prompts**: Provide specific, actionable prompts for human reviewers
2. **Reasonable Timeouts**: Set appropriate timeouts (3-10 minutes for complex decisions)
3. **Meaningful Defaults**: Choose safe default actions for timeout scenarios
4. **Context Display**: Show relevant information to help human decision-making
5. **Conditional Routing**: Use human decisions to guide workflow execution paths
6. **Error Handling**: Always provide fallback paths for rejection scenarios

### Use Cases

- **Quality Control**: Human review of AI-generated content
- **Critical Decisions**: Human oversight for high-stakes determinations
- **Feedback Collection**: Gathering human input to improve AI outputs
- **Approval Workflows**: Multi-step approval processes
- **Interactive Analysis**: Collaborative human-AI investigation

## Setup Requirements

1. **Memory Service**: Configure Supabase or Neo4j
2. **LLM Services**: Set up Gemini API keys or Ollama
3. **Dependencies**: Install required packages

```bash
pip install pydantic
# Plus existing service dependencies
```

## Best Practices

1. **Modular Design**: Break complex workflows into smaller, reusable steps
2. **Clear Naming**: Use descriptive IDs and descriptions for steps
3. **Error Handling**: Include fallback steps for robust execution
4. **Memory Management**: Use appropriate chunk sizes for vector storage
5. **Model Selection**: Choose appropriate models for task complexity

## Extending the System

### Custom Step Handlers

```python
def custom_step_handler(step: StepConfig, inputs: Dict[str, Any], 
                       context: ExecutionContext) -> Any:
    # Your custom logic here
    return result

# Register the handler
orchestrator.step_registry.register_handler(StepType.CUSTOM, custom_step_handler)
```

### Custom Services

Add new services to the orchestrator:

```python
class CustomService:
    def process(self, data):
        return processed_data

orchestrator._services["custom"] = CustomService()
```

## Examples Directory

- `simple_chat_agent.json`: Basic chat functionality
- `document_analysis_agent.json`: PDF processing and Q&A
- `research_agent.json`: Multi-document research analysis
- `human_approval_workflow.json`: Complex human-in-the-loop document analysis
- `simple_human_input_example.json`: Basic human feedback collection
- `usage_example.py`: Python usage examples

## Troubleshooting

1. **Service Not Available**: Ensure required services are properly configured
2. **Step Input Errors**: Check input references and global variables
3. **Memory Issues**: Verify memory service configuration
4. **Model Errors**: Confirm API keys and model availability

## Contributing

This system is designed to be extensible. You can:

1. Add new step types by creating handlers
2. Integrate additional services
3. Extend the JSON schema for new features
4. Create reusable workflow templates

The modular architecture makes it easy to add new capabilities while maintaining backward compatibility.