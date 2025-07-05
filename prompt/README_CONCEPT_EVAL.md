# Concept-Based Prompt Evaluation System

A comprehensive LLM-as-judge evaluation framework for testing prompt quality through concept verification.

## Overview

This system allows you to:
- Create structured evaluations for prompt templates
- Test prompts with different contexts and inputs
- Use LLM judges to verify specific concepts are present/absent
- Validate binary decision outputs
- Track evaluation history and performance trends
- Generate actionable recommendations for prompt improvement

## Quick Start

```python
from prompt.eval_builder import EvaluationBuilder
from prompt.concept_eval_storage import create_concept_eval_storage
from prompt.concept_evaluation_service import ConceptEvaluationService

# 1. Create an evaluation
builder = EvaluationBuilder("My Evaluation", "Tests prompt quality")
builder.with_prompt_template("Summarize this {{doc_type}}: {{content}}")

# Add concept checks
builder.add_concept_check(
    check_type="must_contain",
    description="Contains key facts",
    concept="specific facts and data from the source",
    check_id="facts"
)

builder.add_concept_check(
    check_type="must_not_contain", 
    description="Avoids opinions",
    concept="personal opinions or subjective judgments",
    check_id="objective"
)

# Add test cases
builder.add_test_case(
    context={"doc_type": "report", "content": "Revenue grew 15% to $2.3B..."},
    concept_check_refs=["facts", "objective"],
    name="financial_report_test"
)

eval_def = builder.build("my_eval")

# 2. Run the evaluation
storage = create_concept_eval_storage("file")
service = ConceptEvaluationService(storage)

# Save and run
storage.save_evaluation_definition(eval_def)
results = service.run_evaluation_by_id("my_eval")

print(f"Score: {results.overall_score:.1%}")
```

## Core Concepts

### Evaluation Structure
```
Evaluation = Prompt Template + Test Cases + Concept Checks
```

- **Prompt Template**: Jinja2 template with variables (e.g., `{{variable}}`)
- **Test Cases**: Different contexts to fill the template
- **Concept Checks**: What the LLM judge should verify in outputs

### Concept Check Types

1. **Must Contain** - Output should include specific concepts
2. **Must Not Contain** - Output should avoid specific concepts  
3. **Binary Decision** - For yes/no outputs, verify correctness

### Workflow

1. **Template + Context** → Generate LLM output
2. **LLM Judge** → Evaluate output against concept criteria
3. **Results** → Pass/fail + reasoning + confidence scores

## Builder Patterns

### Basic Builder
```python
builder = EvaluationBuilder("Evaluation Name")
builder.with_prompt_template("Your template here")
builder.add_concept_check(...)
builder.add_test_case(...)
eval_def = builder.build()
```

### Quick Builders
```python
# For accuracy testing
QuickEvaluationBuilder.create_accuracy_evaluation(name, template, test_cases)

# For style testing  
QuickEvaluationBuilder.create_style_evaluation(name, template, test_cases)

# For binary decisions
QuickEvaluationBuilder.create_binary_decision_evaluation(name, template, decision_cases)
```

### Pre-built Templates
```python
# Summarization
EvaluationTemplates.get_summarization_eval(test_cases)

# Classification
EvaluationTemplates.get_classification_eval(categories, test_cases)

# Q&A
EvaluationTemplates.get_qa_eval(test_cases)
```

## Common Concept Checks

The system includes predefined checks for common scenarios:

```python
builder.add_common_check("contains_factual_info")
builder.add_common_check("avoids_opinions") 
builder.add_common_check("professional_tone")
builder.add_common_check("answers_question")
builder.add_common_check("correct_format")
builder.add_common_check("no_hallucination")
```

## Storage Options

### File Storage (Development)
```python
storage = create_concept_eval_storage("file", storage_path="./evaluations")
```

### Supabase Storage (Production)
```python
storage = create_concept_eval_storage("supabase", 
    supabase_url="your-url", 
    supabase_key="your-key"
)
```

### Auto Selection
```python
storage = create_concept_eval_storage("auto")  # Tries Supabase, falls back to file
```

## LLM Judge Configuration

### Judge Models
- **Gemini** (default): Fast, good reasoning
- **Anthropic**: High quality, detailed reasoning
- **Ollama**: Local deployment option

### Judge Settings
```python
ConceptEvaluationService(
    storage_backend=storage,
    default_llm_provider="gemini",  # For generating outputs
    default_judge_model="gemini"    # For evaluating outputs
)
```

## Results Analysis

### Execution Results
```python
results = service.run_evaluation(evaluation)

# Overall metrics
print(f"Score: {results.overall_score:.1%}")
print(f"Passed: {results.passed_test_cases}/{results.total_test_cases}")

# Concept breakdown
for check_type, breakdown in results.concept_breakdown.items():
    print(f"{check_type}: {breakdown['pass_rate']:.1%}")

# Recommendations
for rec in results.recommendations:
    print(f"- {rec}")
```

### History Tracking
```python
# Get previous runs
history = service.get_evaluation_history("eval_id")

# Compare with previous run
comparison = service.compare_evaluations("eval_id")
print(f"Score change: {comparison['score_change']:+.1%}")
```

## Advanced Features

### Weighted Concept Checks
```python
builder.add_concept_check(
    check_type="must_contain",
    description="Critical requirement",
    concept="essential information",
    weight=2.0,  # Higher importance
    check_id="critical"
)
```

### Inline vs Referenced Checks
```python
# Referenced (reusable)
builder.add_test_case(
    context={"var": "value"},
    concept_check_refs=["check1", "check2"]
)

# Inline (one-off)
builder.add_test_case_with_inline_checks(
    context={"var": "value"},
    inline_checks=[{
        "type": "binary_decision",
        "description": "Specific check",
        "concept": "Custom criteria",
        "expected_value": "yes"
    }]
)
```

### CSV Import
```python
from prompt.eval_builder import EvaluationImporter

eval_def = EvaluationImporter.from_csv(
    csv_path="test_cases.csv",
    name="Imported Evaluation",
    prompt_template="Process {{input_text}}",
    context_columns=["input_text", "category"],
    concept_checks=[{
        "type": "must_contain",
        "description": "Has category",
        "concept": "valid category classification"
    }]
)
```

## Integration with Agents

The evaluation system integrates with the agent orchestration system:

```json
{
  "id": "evaluate_prompt",
  "type": "concept_evaluation",
  "config": {
    "eval_id": "my_evaluation",
    "llm_provider": "gemini"
  },
  "inputs": {
    "context_variables": {"doc_type": "report", "content": "$document_text"}
  },
  "outputs": ["evaluation_results"]
}
```

## Best Practices

### Evaluation Design
1. **Start Simple**: Begin with basic concept checks
2. **Iterate**: Refine checks based on actual outputs
3. **Balance Coverage**: Test happy path + edge cases
4. **Clear Concepts**: Make concept descriptions specific and measurable

### Concept Checks
1. **Be Specific**: "Contains revenue figures" vs "Contains financial data"
2. **Avoid Overlap**: Don't test the same thing multiple ways
3. **Use Weights**: Prioritize important checks
4. **Test Incrementally**: Add checks one at a time

### Test Cases
1. **Diverse Contexts**: Cover different input variations
2. **Edge Cases**: Include challenging scenarios
3. **Expected Outputs**: Provide references when possible
4. **Meaningful Names**: Use descriptive test case names

### Judge Configuration
1. **Low Temperature**: Use 0.1-0.2 for consistency
2. **Appropriate Models**: Stronger models for complex concepts
3. **Batch Evaluation**: Group similar checks for efficiency

## Troubleshooting

### Common Issues
1. **Judge Inconsistency**: Lower temperature, clearer concept descriptions
2. **Low Scores**: Check if concept descriptions match actual requirements
3. **Slow Execution**: Use batch evaluation, optimize concept checks
4. **Parse Errors**: Ensure judge model outputs proper format

### Debug Tips
1. **Check Judge Reasoning**: Review `check_result.judge_reasoning`
2. **Validate Templates**: Test template rendering separately
3. **Start Small**: Begin with one test case and concept check
4. **Monitor Confidence**: Low confidence may indicate unclear concepts

## Examples

See `examples/concept_evaluation_example.py` for comprehensive usage examples including:
- Document summarization evaluation
- Text classification evaluation  
- Binary decision evaluation
- Advanced multi-modal evaluation
- History tracking and comparison
- Integration patterns

## Files

- `concept_eval_models.py` - Data models and schemas
- `concept_eval_storage.py` - Storage backends (file/Supabase)
- `concept_judge.py` - LLM-as-judge service
- `concept_evaluation_service.py` - Main evaluation orchestration
- `eval_builder.py` - Builder tools and templates
- `README_CONCEPT_EVAL.md` - This documentation