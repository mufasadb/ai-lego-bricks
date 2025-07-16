# 🎯 Prompt Management System

A comprehensive prompt management system for AI Lego Bricks that enables externalized prompt management, versioning, evaluation, and training data collection.

## ✨ Features

### 🔧 Core Capabilities
- **Externalized Prompt Management** - Manage prompts outside of code
- **Semantic Versioning** - Track prompt evolution with proper versioning
- **Template Support** - Dynamic prompts with Jinja2 template engine
- **Multi-Role Prompts** - Support for system, user, and assistant messages
- **Storage Backends** - File-based and Supabase storage options
- **Caching Layer** - Fast access with configurable TTL

### 📊 Evaluation & Analytics
- **Execution Logging** - Automatic tracking of prompt usage
- **Performance Metrics** - Response time, success rate, token usage
- **A/B Testing** - Compare prompt versions
- **Training Data Export** - Generate datasets for model training
- **Evaluation Reports** - Comprehensive performance analysis

### 🤖 Agent Integration
- **Workflow Integration** - Use prompts in agent orchestration
- **Dynamic Templating** - Inject workflow variables into prompts
- **Fallback Mechanisms** - Graceful handling of missing prompts
- **Execution Tracking** - Built-in logging for all prompt usage

## 🚀 Quick Start

### Installation

The prompt management system is part of AI Lego Bricks. Ensure you have the required dependencies:

```bash
pip install jinja2>=3.0.0 semantic-version>=2.10.0
```

### Basic Usage

```python
from prompt import create_prompt_service, PromptRole, PromptStatus

# Create prompt service (auto-detects storage backend)
prompt_service = create_prompt_service("auto")

# Create a simple prompt
content = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user", 
        "content": {
            "template": "{{ user_question }}",
            "required_variables": ["user_question"]
        }
    }
]

prompt = prompt_service.create_prompt(
    prompt_id="helpful_assistant",
    name="Helpful Assistant Prompt",
    content=content,
    version="1.0.0",
    status=PromptStatus.ACTIVE
)

# Render prompt with context
rendered = prompt_service.render_prompt(
    "helpful_assistant", 
    context={"user_question": "What is machine learning?"}
)

print(rendered)
# [
#   {"role": "system", "content": "You are a helpful AI assistant."},
#   {"role": "user", "content": "Answer this question: What is machine learning?"}
# ]
```

### Agent Workflow Integration

Use managed prompts in agent workflows:

```json
{
  "id": "ai_response",
  "type": "llm_chat",
  "prompt_ref": {
    "prompt_id": "helpful_assistant",
    "version": "1.0.0",
    "context_variables": {
      "assistant_style": "professional"
    }
  },
  "config": {
    "provider": "gemini"
  },
  "inputs": {
    "user_question": "$user_input"
  }
}
```

## 📚 Core Concepts

### Prompt Structure

Prompts consist of:

- **ID**: Unique identifier
- **Name**: Human-readable name
- **Content**: List of role-based messages
- **Version**: Semantic version (e.g., "1.2.0")
- **Status**: Lifecycle status (draft, active, deprecated, archived)
- **Metadata**: Tags, categories, author, description

### Template System

Dynamic prompts using Jinja2:

```python
{
    "role": "user",
    "content": {
        "template": "Analyze {{ document_type }} focusing on {{ focus_areas | join(', ') }}",
        "variables": {
            "document_type": "report",
            "focus_areas": ["metrics", "trends"]
        },
        "required_variables": ["document_type"]
    }
}
```

### Versioning

Prompts use semantic versioning:

- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, improvements

## 🗄️ Storage Backends

### File Storage

Simple file-based storage for development:

```python
prompt_service = create_prompt_service("file", storage_path="./prompts")
```

Directory structure:
```
./prompts/
├── prompts/
│   ├── helpful_assistant_v1.0.0.json
│   └── helpful_assistant_v1.1.0.json
└── executions/
    ├── exec_001.json
    └── exec_002.json
```

### Supabase Storage

Production storage with Supabase:

```python
# Requires SUPABASE_URL and SUPABASE_ANON_KEY environment variables
prompt_service = create_prompt_service("supabase")
```

Required tables:
- `prompts`: Store prompt definitions
- `prompt_executions`: Store execution logs

## 📈 Evaluation & Analytics

### Execution Logging

Automatic logging of all prompt executions:

```python
# Logged automatically when using prompts in workflows
execution_id = prompt_service.log_execution(
    prompt_id="helpful_assistant",
    prompt_version="1.0.0",
    execution_context={"user_question": "What is AI?"},
    rendered_messages=[...],
    llm_provider="gemini",
    llm_response="AI stands for...",
    execution_time_ms=1200,
    success=True
)
```

### Performance Metrics

Get comprehensive evaluation reports:

```python
from prompt.evaluation_service import EvaluationService

eval_service = EvaluationService(prompt_service.storage)
report = eval_service.generate_evaluation_report("helpful_assistant")

print(f"Success Rate: {report['metrics']['success_rate']:.1%}")
print(f"Avg Response Time: {report['metrics']['average_response_time_ms']:.0f}ms")
print(f"Performance Grade: {report['grade']}")
```

### A/B Testing

Compare prompt versions:

```python
comparison = eval_service.compare_prompt_versions(
    "helpful_assistant", "1.0.0",
    "helpful_assistant", "1.1.0"
)

print(f"Winner: {comparison.winner}")
print(f"Confidence: {comparison.confidence_level:.1%}")
```

### Training Data Export

Export execution history for model training:

```python
# Export as JSONL for fine-tuning
training_data = prompt_service.export_training_data(
    "helpful_assistant", 
    format="jsonl",
    limit=1000
)

# Export as CSV for analysis
csv_data = prompt_service.export_training_data(
    "helpful_assistant",
    format="csv"
)
```

### Concept Evaluation Framework

Advanced concept-based evaluation for comprehensive prompt testing:

```python
from prompt.concept_evaluation_service import ConceptEvaluationService
from prompt.concept_eval_storage import create_concept_eval_storage

# Create evaluation service
storage = create_concept_eval_storage('auto')
eval_service = ConceptEvaluationService(storage)

# Run evaluation with test cases
evaluation_definition = {
    "id": "routing_accuracy",
    "name": "Coordinator Routing Accuracy",
    "description": "Tests if coordinator routes queries to correct experts",
    "test_cases": [
        {
            "input": "How do I optimize database queries?",
            "expected_concepts": ["technical", "database"],
            "expected_expert": "technical_expert"
        }
    ],
    "evaluation_criteria": {
        "routing_accuracy": 0.9,
        "response_quality": 0.8
    }
}

results = eval_service.run_evaluation(evaluation_definition)
print(f"Overall Score: {results.overall_score:.1%}")
```

## 🛠️ Advanced Usage

### Prompt Versioning

Create new versions:

```python
# Create improved version
new_version = prompt_service.create_prompt_version(
    "helpful_assistant",
    "1.1.0",
    content=updated_content,
    changelog="Improved response quality and added context awareness"
)

# Activate new version
prompt_service.activate_prompt("helpful_assistant", "1.1.0")
```

### Template Validation

Validate templates before deployment:

```python
validation = prompt_service.validate_template(
    template="Hello {{ name }}, your score is {{ score }}",
    required_variables=["name", "score"],
    test_context={"name": "Alice", "score": 95}
)

if validation["valid"]:
    print("Template is valid!")
else:
    print(f"Errors: {validation['errors']}")
```

### Conditional Prompt Selection

Use different prompts based on context:

```json
{
  "id": "smart_routing",
  "type": "condition",
  "config": {
    "condition_type": "llm_decision",
    "condition_prompt": "Route to: technical or general assistance",
    "route_options": ["technical", "general"]
  },
  "routes": {
    "technical": "technical_support_step",
    "general": "general_chat_step"
  }
}
```

## 📁 File Structure

```
prompt/
├── __init__.py                    # Package exports
├── prompt_service.py             # Main service
├── prompt_models.py              # Pydantic models
├── prompt_storage.py             # Storage backends
├── prompt_registry.py            # Registry with caching
├── evaluation_service.py         # Evaluation and A/B testing
├── concept_evaluation_service.py # Concept-based evaluation framework
├── concept_eval_models.py        # Concept evaluation data models
├── concept_eval_storage.py       # Concept evaluation storage
├── concept_judge.py              # Concept evaluation judge
├── eval_builder.py               # Evaluation builder utilities
├── coordinator_expert_evaluation.json   # Expert evaluation definitions
├── coordinator_routing_evaluation.json  # Routing evaluation definitions
└── prompt-readme.md              # This file
```

## 🔧 Configuration

Environment variables:

```bash
# Storage configuration
PROMPT_STORAGE_BACKEND=supabase  # or 'file' or 'auto'
PROMPT_STORAGE_PATH=./prompts    # for file backend
PROMPT_CACHE_TTL=3600           # cache timeout in seconds

# Evaluation configuration
PROMPT_EVALUATION_ENABLED=true  # enable execution logging

# Supabase configuration (if using Supabase backend)
SUPABASE_URL=your-project-url
SUPABASE_ANON_KEY=your-anon-key
```

## 📊 Best Practices

### Prompt Design
1. **Clear Instructions**: Be specific about desired output format
2. **Context Variables**: Use templates for dynamic content
3. **Role Separation**: Use system messages for behavior, user messages for tasks
4. **Version Incrementally**: Make small, testable changes

### Versioning Strategy
1. **Semantic Versions**: Follow semantic versioning principles
2. **Changelog**: Document changes for each version
3. **Gradual Rollout**: Test new versions before full activation
4. **Rollback Plan**: Keep previous versions available

### Performance Optimization
1. **Cache Usage**: Monitor cache hit rates
2. **Template Efficiency**: Minimize complex template logic
3. **Evaluation Frequency**: Regular performance monitoring
4. **A/B Testing**: Data-driven prompt improvements

### Security Considerations
1. **Access Control**: Restrict prompt modification permissions
2. **Input Validation**: Validate template variables
3. **Sensitive Data**: Avoid hardcoding secrets in prompts
4. **Audit Trail**: Monitor prompt changes and usage

## 🤝 Integration Examples

### Document Analysis Agent

```python
# Create specialized prompt for document analysis
doc_analysis_prompt = prompt_service.create_prompt(
    prompt_id="document_analyzer",
    name="Document Analysis Expert",
    content=[
        {
            "role": "system",
            "content": "You are an expert document analyst with deep knowledge of {{ domain }}."
        },
        {
            "role": "user",
            "content": {
                "template": """
                Analyze the following {{ document_type }}:

                {{ document_content }}

                Focus on:
                {% for area in focus_areas %}
                - {{ area }}
                {% endfor %}

                Provide a structured analysis with key insights.
                """,
                "required_variables": ["document_type", "document_content", "focus_areas"]
            }
        }
    ],
    metadata={
        "category": "analysis",
        "domain": "general",
        "tags": ["document", "analysis", "expert"]
    }
)
```

### Customer Support Bot

```python
# Multi-version support bot
support_prompt = prompt_service.create_prompt(
    prompt_id="customer_support",
    name="Customer Support Assistant",
    content=[
        {
            "role": "system",
            "content": {
                "template": """
                You are a {{ support_level }} customer support representative for {{ company_name }}.
                
                Guidelines:
                - Be helpful and professional
                - {{ tone_guidance }}
                - Escalate if: {{ escalation_triggers | join(', ') }}
                """,
                "variables": {
                    "support_level": "senior",
                    "company_name": "TechCorp",
                    "tone_guidance": "Maintain a friendly but professional tone",
                    "escalation_triggers": ["technical issues", "billing disputes", "urgent requests"]
                }
            }
        }
    ]
)
```

## 🔍 Monitoring & Observability

### Real-time Metrics
- Prompt execution count
- Success/failure rates
- Response time percentiles
- Token usage trends

### Alerting
- High error rates
- Performance degradation
- Template rendering failures
- Cache miss rates

### Dashboards
- Prompt usage analytics
- Performance comparisons
- A/B test results
- Cost optimization insights

## 🎯 Future Enhancements

- **Prompt Analytics Dashboard**: Web interface for monitoring
- **Auto-optimization**: ML-driven prompt improvements
- **Multi-language Support**: Internationalization features
- **Prompt Marketplace**: Sharing and discovery platform
- **Advanced A/B Testing**: Statistical significance testing
- **Integration Plugins**: Pre-built connectors for popular tools

---

For more examples and detailed API documentation, explore the agent orchestration examples that demonstrate prompt management integration and the concept evaluation framework for testing prompt effectiveness.