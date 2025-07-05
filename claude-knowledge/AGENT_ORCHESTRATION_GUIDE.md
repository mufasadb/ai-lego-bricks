# Agent Orchestration System - Comprehensive Guide

## Overview

The Agent Orchestration System is the primary interface for building sophisticated AI agents by combining all the project's building blocks through JSON configuration files. This allows creating complex agent workflows without writing code.

## Core Concepts

### Building Blocks Available

1. **LLM Services**
   - Text generation (Gemini, Ollama)
   - Vision analysis (Gemini Vision)
   - Embedding generation

2. **Memory Management**
   - Vector storage (Supabase pgvector)
   - Graph storage (Neo4j)
   - Semantic search and retrieval

3. **Document Processing**
   - PDF text extraction
   - LLM-enhanced content improvement
   - Semantic analysis and classification

4. **Text Chunking**
   - Intelligent text segmentation
   - Semantic boundary preservation
   - Configurable chunk sizes

5. **Chat Interface**
   - Conversation management
   - Multi-provider support

### Workflow Execution Model

1. **Sequential Processing**: Steps execute in order
2. **Data Flow**: Outputs from one step become inputs to the next
3. **Context Management**: Global variables and step outputs tracked
4. **Error Handling**: Graceful fallbacks and comprehensive error reporting

## JSON Workflow Configuration

### Basic Structure

```json
{
  "name": "workflow_name",
  "description": "What this workflow does",
  "config": {
    "memory_backend": "auto|supabase|neo4j",
    "default_llm_provider": "gemini|ollama",
    "default_model": "model_name"
  },
  "steps": [
    {
      "id": "unique_step_id",
      "type": "step_type",
      "description": "What this step does",
      "config": {},
      "inputs": {},
      "outputs": []
    }
  ]
}
```

### Available Step Types

#### Core Operations
- **`input`**: Collect user input or external data
- **`output`**: Format and return final results
- **`llm_chat`**: Generate text using LLM
- **`llm_vision`**: Analyze images with vision models

#### Document Processing
- **`document_processing`**: Extract and enhance text from PDFs
- **`chunk_text`**: Break text into semantic chunks

#### Memory Operations
- **`memory_store`**: Store content in vector/graph storage
- **`memory_retrieve`**: Search and retrieve relevant memories

#### Control Flow
- **`condition`**: Conditional execution
- **`loop`**: Iterate over collections

#### Human Interaction
- **`human_approval`**: Human-in-the-loop approval and feedback collection

### Input/Output Mapping

#### Direct Values
```json
"inputs": {
  "message": "Hello, world!"
}
```

#### Step References
```json
"inputs": {
  "text": {
    "from_step": "previous_step_id",
    "field": "output_field"
  }
}
```

#### Global Variables
```json
"inputs": {
  "file_path": "$document_path"
}
```

## Example Workflows

### 1. Simple Chat Agent

**Purpose**: Basic conversational AI
**Building Blocks**: LLM chat only

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

### 2. Document Analysis Agent

**Purpose**: Process PDFs and answer questions about them
**Building Blocks**: Document processing, chunking, memory storage, LLM chat

```json
{
  "name": "document_analyzer",
  "description": "Processes PDFs and answers questions",
  "steps": [
    {
      "id": "process_pdf",
      "type": "document_processing",
      "config": {"enhance_with_llm": true},
      "inputs": {"file_path": "$document_path"},
      "outputs": ["text", "semantic_analysis"]
    },
    {
      "id": "chunk_text",
      "type": "chunk_text",
      "config": {"target_size": 1000},
      "inputs": {"text": {"from_step": "process_pdf", "field": "text"}},
      "outputs": ["chunks"]
    },
    {
      "id": "store_chunks",
      "type": "memory_store",
      "inputs": {
        "content": {"from_step": "chunk_text", "field": "chunks"},
        "metadata": {"source": "$document_path"}
      },
      "outputs": ["memory_id"]
    },
    {
      "id": "get_question",
      "type": "input",
      "config": {"prompt": "What's your question about the document?"},
      "outputs": ["question"]
    },
    {
      "id": "search_memory",
      "type": "memory_retrieve",
      "config": {"limit": 3, "threshold": 0.7},
      "inputs": {"query": {"from_step": "get_question", "field": "question"}},
      "outputs": ["memories"]
    },
    {
      "id": "generate_answer",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Context: {context}\n\nQuestion: {question}\n\nAnswer based on the context:"
      },
      "outputs": ["response"]
    },
    {
      "id": "output_answer",
      "type": "output",
      "inputs": {"answer": {"from_step": "generate_answer", "field": "response"}}
    }
  ]
}
```

### 3. Research Agent

**Purpose**: Analyze multiple documents for comprehensive research
**Building Blocks**: All components - document processing, memory, LLM analysis

```json
{
  "name": "research_agent",
  "description": "Comprehensive research analysis",
  "steps": [
    {
      "id": "process_documents",
      "type": "document_processing",
      "config": {"enhance_with_llm": true, "semantic_analysis": true},
      "inputs": {"file_path": "$document_paths"},
      "outputs": ["processed_docs"]
    },
    {
      "id": "extract_concepts",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Extract key concepts, themes, and findings from: {documents}"
      },
      "outputs": ["key_concepts"]
    },
    {
      "id": "store_research",
      "type": "memory_store",
      "inputs": {
        "content": {"from_step": "process_documents", "field": "processed_docs"},
        "metadata": {"concepts": {"from_step": "extract_concepts", "field": "key_concepts"}}
      },
      "outputs": ["stored_research"]
    },
    {
      "id": "research_query",
      "type": "input",
      "config": {"prompt": "What research question should I investigate?"},
      "outputs": ["query"]
    },
    {
      "id": "search_knowledge",
      "type": "memory_retrieve",
      "config": {"limit": 5, "threshold": 0.6},
      "inputs": {"query": {"from_step": "research_query", "field": "query"}},
      "outputs": ["findings"]
    },
    {
      "id": "synthesize_findings",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Research Question: {query}\n\nFindings: {findings}\n\nProvide comprehensive analysis with:\n1. Direct answer\n2. Supporting evidence\n3. Limitations\n4. Further research suggestions"
      },
      "outputs": ["synthesis"]
    },
    {
      "id": "format_report",
      "type": "output",
      "config": {"format": "text"},
      "inputs": {"report": {"from_step": "synthesize_findings", "field": "synthesis"}}
    }
  ]
}
```

## Human-in-the-Loop Workflows

### Overview
The `human_approval` step type enables human oversight and feedback collection during workflow execution. This allows for hybrid human-AI workflows where critical decisions require human input, quality control, or feedback collection.

### Key Features
- **Multiple approval types**: approve/reject, multiple choice, custom text input
- **Context presentation**: Shows relevant workflow data to help human decision-making
- **Timeout handling**: Configurable timeouts with default fallback actions
- **Data flow integration**: Human responses become available to subsequent steps
- **Conditional routing**: Different workflow paths based on human decisions
- **Error handling**: Graceful handling of interruptions and edge cases

### Human Approval Step Types

#### 1. Approve/Reject Pattern
```json
{
  "id": "quality_review",
  "type": "human_approval",
  "description": "Human quality review of AI analysis",
  "config": {
    "approval_type": "approve_reject",
    "prompt": "Please review the AI analysis below. Do you approve this for publication?",
    "timeout_seconds": 180,
    "default_action": "reject",
    "show_context": true,
    "context_fields": ["analysis", "confidence_score"]
  },
  "inputs": {
    "analysis": {"from_step": "ai_analysis", "field": "summary"},
    "confidence_score": {"from_step": "ai_analysis", "field": "confidence"}
  },
  "outputs": ["decision", "feedback", "timestamp"],
  "routes": {
    "approve": "publish_content",
    "reject": "revision_needed"
  }
}
```

#### 2. Multiple Choice Selection
```json
{
  "id": "action_selection",
  "type": "human_approval",
  "config": {
    "approval_type": "multiple_choice",
    "prompt": "Based on the analysis, what action should we take?",
    "options": ["continue", "enhance", "revise", "abort"],
    "timeout_seconds": 300,
    "default_action": "continue",
    "show_context": true
  },
  "routes": {
    "continue": "standard_processing",
    "enhance": "detailed_analysis",
    "revise": "get_revision_feedback",
    "abort": "cleanup_workflow"
  }
}
```

#### 3. Custom Input Collection
```json
{
  "id": "feedback_collection",
  "type": "human_approval",
  "config": {
    "approval_type": "custom_input",
    "prompt": "Please provide specific feedback on how to improve this analysis:",
    "timeout_seconds": 600,
    "default_action": "Use original analysis without changes",
    "show_context": true,
    "context_fields": ["original_analysis", "key_findings"]
  },
  "outputs": ["user_input", "feedback", "decision"]
}
```

### Configuration Options

- **`approval_type`**: `"approve_reject"`, `"multiple_choice"`, or `"custom_input"`
- **`prompt`**: Clear, specific message displayed to the human reviewer
- **`options`**: Available choices for multiple_choice type (array of strings)
- **`timeout_seconds`**: Time limit before using default action (default: 300)
- **`default_action`**: Fallback when timeout occurs or input is interrupted
- **`show_context`**: Whether to display relevant context information (default: true)
- **`context_fields`**: Specific fields to show from inputs and previous steps (optional)

### Human Input Data Flow

Human responses seamlessly integrate into the workflow data flow. Subsequent steps can reference human input using the standard input reference system:

```json
{
  "id": "incorporate_feedback",
  "type": "llm_chat",
  "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
  "inputs": {
    "message": "Original analysis: {original_analysis}\n\nHuman feedback: {human_feedback}\n\nPlease revise the analysis incorporating the human feedback while maintaining accuracy and coherence."
  }
}
```

Where `human_feedback` references the human approval step:
```json
{
  "from_step": "feedback_collection",
  "field": "user_input"
}
```

### Complete Example: Document Analysis with Human Review

```json
{
  "name": "human_reviewed_document_analysis",
  "description": "Document analysis workflow with human oversight and feedback",
  "config": {
    "memory_backend": "supabase",
    "default_llm_provider": "gemini",
    "default_model": "gemini-1.5-pro"
  },
  "steps": [
    {
      "id": "process_document",
      "type": "document_processing",
      "config": {"enhance_with_llm": true, "semantic_analysis": true},
      "inputs": {"file_path": "$document_path"},
      "outputs": ["text", "enhanced_text", "key_points"]
    },
    {
      "id": "initial_analysis",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Analyze this document and provide:\n1. Executive summary\n2. Key findings\n3. Recommendations\n4. Confidence level\n\nDocument: {enhanced_text}"
      },
      "outputs": ["analysis"]
    },
    {
      "id": "human_review",
      "type": "human_approval",
      "config": {
        "approval_type": "multiple_choice",
        "prompt": "Review the AI analysis. What would you like to do?",
        "options": ["approve", "enhance", "revise", "reject"],
        "timeout_seconds": 300,
        "default_action": "approve",
        "show_context": true,
        "context_fields": ["analysis", "key_points"]
      },
      "inputs": {
        "ai_analysis": {"from_step": "initial_analysis", "field": "analysis"},
        "document_summary": {"from_step": "process_document", "field": "key_points"}
      },
      "routes": {
        "approve": "store_final_analysis",
        "enhance": "enhance_analysis",
        "revise": "get_revision_feedback",
        "reject": "analysis_rejected"
      }
    },
    {
      "id": "get_revision_feedback",
      "type": "human_approval",
      "config": {
        "approval_type": "custom_input",
        "prompt": "Please specify how the analysis should be revised:",
        "timeout_seconds": 600,
        "show_context": true
      },
      "outputs": ["revision_instructions"],
      "routes": {"default": "revise_analysis"}
    },
    {
      "id": "revise_analysis",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Original analysis: {original_analysis}\n\nRevision instructions: {revision_instructions}\n\nPlease revise the analysis according to the human feedback while maintaining accuracy."
      },
      "routes": {"default": "store_final_analysis"}
    },
    {
      "id": "enhance_analysis",
      "type": "llm_chat",
      "config": {"provider": "gemini", "model": "gemini-1.5-pro"},
      "inputs": {
        "message": "Original analysis: {original_analysis}\n\nDocument text: {full_document}\n\nPlease provide an enhanced, more detailed analysis with deeper insights and comprehensive conclusions."
      },
      "routes": {"default": "store_final_analysis"}
    },
    {
      "id": "store_final_analysis",
      "type": "memory_store",
      "config": {
        "metadata": {
          "analysis_type": "human_reviewed",
          "workflow_version": "1.0",
          "review_timestamp": "current"
        }
      },
      "inputs": {
        "content": {"from_step": "determine_final_content", "field": "final_analysis"},
        "metadata": {
          "human_decision": {"from_step": "human_review", "field": "decision"},
          "review_timestamp": {"from_step": "human_review", "field": "timestamp"}
        }
      },
      "routes": {"default": "final_output"}
    },
    {
      "id": "analysis_rejected",
      "type": "output",
      "config": {"format": "text"},
      "inputs": {
        "message": "Analysis was rejected by human reviewer. Workflow terminated."
      }
    },
    {
      "id": "final_output",
      "type": "output",
      "config": {"format": "json"},
      "inputs": {
        "analysis": {"from_step": "store_final_analysis", "field": "content"},
        "human_decision": {"from_step": "human_review", "field": "decision"},
        "memory_id": {"from_step": "store_final_analysis", "field": "memory_id"},
        "status": "completed_with_human_oversight"
      }
    }
  ]
}
```

### Best Practices for Human Approval Workflows

#### 1. Clear Communication
- **Specific prompts**: Provide clear, actionable prompts that explain what decision is needed
- **Context relevance**: Show only relevant context to avoid information overload
- **Decision impact**: Explain what happens based on different choices

#### 2. Robust Timeout Handling
- **Reasonable timeouts**: Set appropriate timeouts (3-10 minutes for complex decisions)
- **Safe defaults**: Choose conservative default actions for timeout scenarios
- **Escalation paths**: Consider workflows for handling prolonged delays

#### 3. Effective Routing
- **Meaningful paths**: Ensure each route leads to appropriate workflow continuation
- **Error handling**: Always provide fallback paths for rejection scenarios
- **Loop prevention**: Avoid creating infinite approval loops

#### 4. Data Integration
- **Structured outputs**: Use consistent output field names for easy reference
- **Context preservation**: Maintain relevant context through the workflow
- **Audit trails**: Capture human decisions and timestamps for tracking

### Use Cases

#### Quality Control Workflows
- **Content review**: Human approval of AI-generated content before publication
- **Analysis validation**: Expert review of AI analysis before making decisions
- **Safety checks**: Human oversight for potentially sensitive operations

#### Interactive Analysis
- **Collaborative investigation**: Iterative human-AI collaboration on complex problems
- **Guided exploration**: Human-directed deep dives into specific aspects
- **Feedback incorporation**: Real-time improvement of AI outputs based on human input

#### Approval Processes
- **Multi-stage approval**: Sequential approvals from different stakeholders
- **Conditional processing**: Different workflows based on approval outcomes
- **Documentation**: Automatic capture of approval decisions and reasoning

#### Training and Improvement
- **Feedback collection**: Gathering human input to improve AI models
- **Quality assessment**: Human evaluation of AI performance
- **Preference learning**: Understanding human preferences for better AI alignment

The human approval system enables sophisticated human-AI collaboration while maintaining the declarative, configuration-driven approach of the orchestration system.

## Step Configuration Details

### Document Processing Step
```json
{
  "id": "process_doc",
  "type": "document_processing",
  "config": {
    "enhance_with_llm": true,        // Use LLM to improve extracted text
    "semantic_analysis": true,       // Perform semantic analysis
    "vision_fallback": true         // Use vision models for difficult PDFs
  },
  "inputs": {
    "file_path": "$document_path"
  },
  "outputs": ["text", "enhanced_text", "semantic_analysis", "page_count"]
}
```

### Memory Store Step
```json
{
  "id": "store_content",
  "type": "memory_store",
  "config": {
    "metadata": {
      "document_type": "research_paper",
      "processing_date": "2024-01-01"
    }
  },
  "inputs": {
    "content": {"from_step": "chunk_text", "field": "chunks"},
    "metadata": {
      "source_file": "$document_path",
      "concepts": {"from_step": "extract_concepts", "field": "concepts"}
    }
  },
  "outputs": ["memory_id", "stored_content", "metadata"]
}
```

### Memory Retrieve Step
```json
{
  "id": "search_memories",
  "type": "memory_retrieve",
  "config": {
    "limit": 5,           // Maximum number of memories to retrieve
    "threshold": 0.7      // Minimum similarity threshold
  },
  "inputs": {
    "query": {"from_step": "get_question", "field": "question"}
  },
  "outputs": ["memories", "count"]
}
```

### LLM Chat Step
```json
{
  "id": "generate_response",
  "type": "llm_chat",
  "config": {
    "provider": "gemini",           // or "ollama"
    "model": "gemini-1.5-pro",     // specific model name
    "temperature": 0.7,            // creativity level
    "max_tokens": 1000             // response length limit
  },
  "inputs": {
    "message": "Your prompt here with {variables}"
  },
  "outputs": ["response", "message", "provider", "model"]
}
```

### LLM Vision Step
```json
{
  "id": "analyze_image",
  "type": "llm_vision",
  "config": {
    "provider": "gemini",
    "model": "gemini-1.5-pro-vision"
  },
  "inputs": {
    "image_path": "$image_file",
    "prompt": "Describe this image in detail"
  },
  "outputs": ["analysis", "image_path", "prompt"]
}
```

### Chunking Step
```json
{
  "id": "chunk_document",
  "type": "chunk_text",
  "config": {
    "target_size": 1000,    // Target chunk size in characters
    "tolerance": 0.2        // Allow 20% variance from target size
  },
  "inputs": {
    "text": {"from_step": "process_document", "field": "text"}
  },
  "outputs": ["chunks", "chunk_count", "total_length"]
}
```

### Human Approval Step
```json
{
  "id": "review_decision",
  "type": "human_approval",
  "config": {
    "approval_type": "approve_reject|multiple_choice|custom_input",
    "prompt": "Clear, specific prompt for human reviewer",
    "options": ["option1", "option2", "option3"],          // For multiple_choice only
    "timeout_seconds": 300,                                // Time limit (default: 300)
    "default_action": "reject",                           // Fallback action
    "show_context": true,                                 // Show context (default: true)
    "context_fields": ["field1", "field2"]              // Specific fields to show
  },
  "inputs": {
    "analysis": {"from_step": "ai_analysis", "field": "summary"},
    "confidence": {"from_step": "ai_analysis", "field": "confidence"}
  },
  "outputs": ["decision", "feedback", "timestamp", "user_input"],  // user_input for custom_input type
  "routes": {
    "approve": "next_step",
    "reject": "error_step",
    "enhance": "enhancement_step"
  }
}
```

## Python Usage

### Basic Orchestration
```python
from agent_orchestration import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator()

# Load workflow from file
workflow = orchestrator.load_workflow_from_file("my_agent.json")

# Execute with initial inputs
result = orchestrator.execute_workflow(workflow, {
    "document_path": "/path/to/document.pdf",
    "user_query": "What are the main findings?"
})

if result.success:
    print(f"Final output: {result.final_output}")
    print(f"Execution time: {result.execution_time:.2f}s")
else:
    print(f"Error: {result.error}")
```

### Custom Workflow Creation
```python
from agent_orchestration import AgentOrchestrator

# Define workflow as dictionary
custom_workflow = {
    "name": "text_summarizer",
    "description": "Summarizes input text",
    "steps": [
        {
            "id": "get_text",
            "type": "input",
            "config": {"value": "Long text to summarize..."},
            "outputs": ["text"]
        },
        {
            "id": "summarize",
            "type": "llm_chat",
            "config": {"provider": "gemini"},
            "inputs": {"message": "Summarize: {text}"},
            "outputs": ["summary"]
        },
        {
            "id": "output",
            "type": "output",
            "inputs": {"result": {"from_step": "summarize", "field": "summary"}}
        }
    ]
}

# Create and execute
orchestrator = AgentOrchestrator()
workflow = orchestrator.load_workflow_from_dict(custom_workflow)
result = orchestrator.execute_workflow(workflow)
```

## Advanced Features

### Error Handling
- Graceful service fallbacks when components aren't available
- Comprehensive error reporting with step-level details
- Execution context preservation for debugging

### Performance Optimization
- Lazy service initialization
- Efficient memory usage
- Parallel-ready architecture (future enhancement)

### Extensibility
- Easy to add new step types
- Custom step handlers
- Plugin architecture ready

## Best Practices

### Workflow Design
1. **Start Simple**: Begin with basic workflows and add complexity gradually
2. **Clear Naming**: Use descriptive IDs and descriptions for all steps
3. **Modular Design**: Break complex workflows into reusable components
4. **Error Handling**: Include fallback steps for robust execution

### Performance
1. **Chunk Wisely**: Use appropriate chunk sizes for your use case
2. **Memory Thresholds**: Adjust similarity thresholds based on your data
3. **Model Selection**: Choose appropriate models for task complexity

### Maintenance
1. **Document Workflows**: Include clear descriptions for all steps
2. **Version Control**: Track workflow configurations in version control
3. **Test Thoroughly**: Validate workflows with representative data
4. **Monitor Execution**: Track execution times and error rates

## Troubleshooting

### Common Issues
1. **Service Not Available**: Check environment configuration and service status
2. **Input/Output Mapping**: Verify step references and field names
3. **Model Errors**: Confirm API keys and model availability
4. **Memory Issues**: Check vector database connection and configuration

### Debugging
1. **Step-by-step**: Execute workflows step by step to isolate issues
2. **Log Analysis**: Review execution context and step outputs
3. **Service Testing**: Test individual services outside of workflows
4. **Configuration Validation**: Validate JSON against schema

This orchestration system provides a powerful, flexible way to build sophisticated AI agents using the project's building blocks while maintaining simplicity and extensibility.