# Prompt Management System

## Overview

The AI Lego Bricks prompt management system provides externalized prompt management with versioning, evaluation, and A/B testing capabilities. This allows teams to manage prompts outside of code, track their evolution, and optimize performance through data-driven insights.

## Core Concepts

### Prompt Structure
- **ID**: Unique identifier across all versions
- **Name**: Human-readable name for the prompt
- **Content**: List of role-based messages (system, user, assistant)
- **Version**: Semantic version following semver (e.g., "1.2.0")
- **Status**: Lifecycle status (draft, active, deprecated, archived)
- **Metadata**: Tags, categories, author, description, use cases

### Template System
Prompts support dynamic content using Jinja2 templates:
- **Variables**: Static default values
- **Required Variables**: Must be provided at render time
- **Template Logic**: Loops, conditionals, filters
- **Context Resolution**: Variables from workflow, step inputs, and global context

### Versioning Strategy
- **Major** (1.0.0 → 2.0.0): Breaking changes to template variables or behavior
- **Minor** (1.0.0 → 1.1.0): New features, improved prompts, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, minor improvements

## Architecture

### Storage Backends
1. **File Storage**: JSON files for development and testing
2. **Supabase Storage**: PostgreSQL with JSON columns for production

### Registry & Caching
- **PromptRegistry**: Central registry with caching layer
- **TTL-based cache**: Configurable cache timeout (default 1 hour)
- **Active prompt tracking**: Fast access to currently active prompts

### Evaluation Service
- **Execution Logging**: Automatic tracking of all prompt usage
- **Performance Metrics**: Success rate, response time, token usage
- **A/B Testing**: Statistical comparison between prompt versions
- **Training Data Export**: Generate datasets for model fine-tuning

### Concept-Based Evaluation System ⭐ **NEW**
- **LLM-as-Judge**: Automated quality assessment using LLM judges
- **Concept Checking**: Verify presence/absence of specific concepts in outputs
- **Binary Decision Validation**: Validate yes/no responses for correctness
- **Structured Test Suites**: Template + context + concept checks framework
- **Quality Gates**: Pass/fail thresholds for automated quality control
- **Agent Integration**: Built-in evaluation steps for workflow quality assurance

## Integration with Agents

### Workflow Configuration
Use `prompt_ref` in step configurations instead of hardcoded prompts:

```json
{
  "id": "analyze_content",
  "type": "llm_structured",
  "prompt_ref": {
    "prompt_id": "content_analyzer",
    "version": "2.1.0",
    "context_variables": {
      "analysis_type": "detailed",
      "focus_areas": ["sentiment", "topics", "entities"]
    }
  },
  "config": {
    "provider": "gemini",
    "response_schema": "analysis_result"
  }
}
```

### Variable Resolution Order
1. **Context Variables**: Static variables from prompt reference
2. **Step Inputs**: Dynamic data from workflow execution
3. **Global Variables**: Workflow-level variables

### Automatic Execution Logging
Every prompt execution is automatically logged with:
- Template context and rendered output
- LLM provider, model, and configuration
- Response time and token usage
- Success/failure status and error details
- Workflow context (step ID, execution ID)

## Concept-Based Evaluation System

### Overview
The concept-based evaluation system extends prompt management with sophisticated quality assessment using LLM-as-judge methodology. This enables automated testing of prompt outputs against specific criteria.

### Core Components

#### Concept Checks
Three types of evaluation criteria:
1. **Must Contain**: Output should include specific concepts or information
2. **Must Not Contain**: Output should avoid certain concepts or patterns
3. **Binary Decision**: For yes/no outputs, validate correctness against expected answers

#### Evaluation Structure
```
Evaluation = Prompt Template + Test Cases + Concept Checks
```

- **Prompt Template**: Jinja2 template with variables
- **Test Cases**: Different contexts to test the template
- **Concept Checks**: What the LLM judge should verify in outputs

#### LLM Judge Process
1. **Template Execution**: Fill template with test context → Generate LLM output
2. **Concept Assessment**: LLM judge evaluates output against each concept check
3. **Scoring**: Pass/fail decisions with confidence scores and reasoning
4. **Results**: Comprehensive evaluation with recommendations

### Usage in Agent Workflows

#### Concept Evaluation Step
```json
{
  "id": "evaluate_output",
  "type": "concept_evaluation",
  "config": {
    "eval_id": "summary_quality_eval",
    "llm_provider": "gemini",
    "judge_model": "gemini",
    "min_score": 0.7,
    "include_detailed_results": true
  },
  "inputs": {
    "context_variables": {
      "document_type": "$doc_type",
      "content": "$document_text"
    }
  },
  "outputs": ["evaluation_results"]
}
```

#### Quality Gates
Use evaluation results for workflow control:
```json
{
  "id": "quality_gate",
  "type": "condition",
  "condition": {
    "field": {"from_step": "evaluate_output", "field": "quality_gate_passed"},
    "operator": "==",
    "value": true
  },
  "routes": {
    "true": "deploy_prompt",
    "false": "improvement_needed"
  }
}
```

#### Evaluation Results
The concept evaluation step outputs:
- **overall_score**: 0-1 quality score
- **grade**: Letter grade (A-F)
- **passed_test_cases**: Number of successful test cases
- **quality_gate_passed**: Boolean for threshold checks
- **recommendations**: Actionable improvement suggestions
- **concept_breakdown**: Pass rates by concept type

### Creating Evaluations

#### Builder Pattern
```python
from prompt.eval_builder import EvaluationBuilder

builder = EvaluationBuilder("Document Summarizer Evaluation")
builder.with_prompt_template("Summarize this {{doc_type}}: {{content}}")

# Add concept checks
builder.add_concept_check(
    check_type="must_contain",
    description="Contains key facts",
    concept="specific facts and data from the source",
    check_id="key_facts"
)

builder.add_concept_check(
    check_type="must_not_contain",
    description="Avoids opinions",
    concept="personal opinions or subjective judgments",
    check_id="objective"
)

# Add test cases
builder.add_test_case(
    context={"doc_type": "report", "content": "Revenue grew 15%..."},
    concept_check_refs=["key_facts", "objective"],
    name="financial_report_test"
)

eval_def = builder.build("doc_summarizer_eval")
```

#### Quick Templates
```python
from prompt.eval_builder import QuickEvaluationBuilder

# For accuracy testing
accuracy_eval = QuickEvaluationBuilder.create_accuracy_evaluation(
    name="Q&A Accuracy Test",
    prompt_template="Answer: {{question}}",
    test_cases=[{"context": {"question": "What is AI?"}}]
)

# For binary decisions
binary_eval = QuickEvaluationBuilder.create_binary_decision_evaluation(
    name="Spam Detection",
    prompt_template="Is this spam? {{email}}",
    decision_cases=[{"context": {"email": "Click here!"}, "expected_decision": "yes"}]
)
```

### Best Practices for Concept Evaluation

#### Evaluation Design
1. **Start Simple**: Begin with basic must contain/not contain checks
2. **Iterate Based on Results**: Refine concept descriptions based on actual outputs
3. **Balance Coverage**: Include both happy path and edge case scenarios
4. **Clear Concepts**: Make concept descriptions specific and measurable

#### Concept Check Guidelines
1. **Be Specific**: "Contains revenue figures" vs "Contains financial data"
2. **Avoid Overlap**: Don't test the same thing multiple ways
3. **Use Weights**: Prioritize important checks with higher weights
4. **Test Incrementally**: Add checks one at a time to isolate issues

#### Judge Configuration
1. **Low Temperature**: Use 0.1-0.2 for consistency in evaluation
2. **Appropriate Models**: Use stronger models (Gemini, Claude) for complex concept evaluation
3. **Chain-of-Thought**: Judges provide reasoning before pass/fail decisions
4. **Batch Evaluation**: Group similar evaluations for efficiency

### Integration Patterns

#### Continuous Quality Monitoring
- Run evaluations on prompt versions before activation
- Set up automated evaluation on production prompt usage
- Monitor concept-specific pass rates over time
- Alert on quality degradation

#### A/B Testing Enhancement
- Compare not just performance metrics but quality scores
- Use concept evaluations to understand why one prompt performs better
- Test different concept criteria to optimize for specific outcomes

#### Training Data Generation
- Use evaluation results to identify high-quality prompt executions
- Export successful examples for fine-tuning datasets
- Filter training data based on concept evaluation scores

## Best Practices

### Prompt Design
1. **Clear Instructions**: Specific about format, tone, and requirements
2. **Role Separation**: System messages for behavior, user messages for tasks
3. **Template Variables**: Use dynamic content for reusability
4. **Example Outputs**: Include examples in system messages when needed

### Version Management
1. **Incremental Changes**: Small, testable improvements
2. **A/B Testing**: Compare versions before full deployment
3. **Rollback Strategy**: Keep previous versions active for quick rollback
4. **Documentation**: Clear changelogs for each version

### Performance Optimization
1. **Template Efficiency**: Minimize complex Jinja2 logic
2. **Context Size**: Keep variable context reasonably sized
3. **Cache Utilization**: Monitor cache hit rates
4. **Regular Evaluation**: Weekly/monthly performance reviews

### Security Considerations
1. **Access Control**: Restrict prompt modification permissions
2. **Input Validation**: Validate template variables
3. **No Secrets**: Never hardcode API keys or sensitive data
4. **Audit Trail**: Track all prompt changes and usage

## Common Patterns

### Multi-Modal Analysis
```python
# Create a multi-modal prompt for document + image analysis
content = [
    {
        "role": "system",
        "content": "You are an expert analyst reviewing {{ document_type }} with supporting visuals."
    },
    {
        "role": "user",
        "content": {
            "template": """
            Document: {{ document_text }}
            
            {% if has_images %}
            Supporting images have been analyzed with these insights:
            {% for insight in image_insights %}
            - {{ insight }}
            {% endfor %}
            {% endif %}
            
            Provide a {{ analysis_depth }} analysis focusing on {{ focus_areas | join(', ') }}.
            """,
            "required_variables": ["document_text", "analysis_depth"],
            "variables": {
                "has_images": False,
                "image_insights": [],
                "focus_areas": ["key findings", "recommendations"]
            }
        }
    }
]
```

### Conversational Context
```python
# Chat assistant with personality and context awareness
content = [
    {
        "role": "system",
        "content": {
            "template": """
            You are {{ assistant_name }}, a {{ personality_trait }} AI assistant.
            
            Current context:
            - User expertise: {{ user_expertise_level }}
            - Conversation topic: {{ current_topic }}
            - Preferred communication style: {{ communication_style }}
            
            Guidelines:
            {{ guidelines | join('\n') }}
            """,
            "variables": {
                "assistant_name": "Claude",
                "personality_trait": "helpful and knowledgeable",
                "user_expertise_level": "intermediate",
                "current_topic": "general",
                "communication_style": "professional",
                "guidelines": [
                    "Be concise but thorough",
                    "Ask clarifying questions when needed",
                    "Provide examples when helpful"
                ]
            }
        }
    }
]
```

### Conditional Processing
```python
# Content routing based on classification
content = [
    {
        "role": "system",
        "content": "You are a content classifier and processor."
    },
    {
        "role": "user",
        "content": {
            "template": """
            {% if content_type == "technical" %}
            Analyze this technical content for accuracy and completeness:
            {{ content }}
            
            Focus on:
            - Technical accuracy
            - Implementation feasibility
            - Best practices compliance
            {% elif content_type == "creative" %}
            Review this creative content for engagement and impact:
            {{ content }}
            
            Focus on:
            - Narrative structure
            - Emotional resonance
            - Audience appeal
            {% else %}
            Perform a general analysis of this content:
            {{ content }}
            
            Focus on:
            - Clarity and coherence
            - Factual accuracy
            - Overall quality
            {% endif %}
            """,
            "required_variables": ["content", "content_type"]
        }
    }
]
```

## Evaluation & Optimization

### Performance Metrics
- **Success Rate**: Percentage of successful executions
- **Response Time**: Average time from prompt to response
- **Token Efficiency**: Tokens used per successful response
- **Error Patterns**: Common failure modes and causes

### A/B Testing Workflow
1. **Create Variant**: New prompt version with hypothesis
2. **Traffic Split**: Route percentage of traffic to new version
3. **Data Collection**: Gather metrics over sufficient sample size
4. **Statistical Analysis**: Compare performance with confidence intervals
5. **Decision**: Promote winner or iterate further

### Training Data Generation
Export execution logs for model fine-tuning:
- **JSONL Format**: For transformer fine-tuning
- **CSV Format**: For analysis and reporting
- **Filtering**: By date range, success status, or performance metrics
- **Privacy**: Automatic removal of sensitive information

## Migration & Deployment

### Gradual Rollout Strategy
1. **Canary Deployment**: Test with small percentage of traffic
2. **Performance Monitoring**: Watch key metrics during rollout
3. **Rollback Plan**: Immediate rollback capability if issues arise
4. **Full Deployment**: Complete migration after validation

### Environment Management
- **Development**: File-based storage for rapid iteration
- **Staging**: Supabase storage with production-like data
- **Production**: Supabase with monitoring and alerting

### Backup & Recovery
- **Version History**: All versions preserved for rollback
- **Data Export**: Regular backups of prompt definitions
- **Execution Logs**: Retained for analysis and compliance
- **Disaster Recovery**: Procedures for system restoration

This prompt management system provides a robust foundation for enterprise-grade prompt engineering with full lifecycle management, evaluation capabilities, and seamless integration with the AI Lego Bricks orchestration system.