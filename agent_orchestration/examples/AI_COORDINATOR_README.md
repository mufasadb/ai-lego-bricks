# AI Coordinator System

This system implements an AI coordinator that intelligently routes user requests to specialized experts using Ollama with the R1 abliterated model.

## System Overview

The coordinator analyzes incoming requests and routes them to one of three experts:

1. **Japanese Expert** - Handles Japanese language, culture, travel, and related topics
2. **Home Assistant Expert** - Handles smart home automation, IoT devices, and home networking
3. **General Expert** - Handles programming, science, business, and general knowledge

## Files

- `ai_coordinator_agent.json` - Main orchestrator configuration with routing logic
- `../prompt/coordinator_routing_evaluation.json` - Routing accuracy evaluation definition
- `../prompt/coordinator_expert_evaluation.json` - Expert response quality evaluation definitions

## How It Works

### Step 1: Request Analysis
The coordinator uses a specialized prompt to analyze the user's request and determine which expert should handle it. It returns a JSON response with:
- `expert`: The chosen expert (japanese/home_assistant/general)
- `confidence`: Confidence score (0.0-1.0)
- `reasoning`: Explanation of the routing decision

### Step 2: Expert Response
Once routed, the appropriate expert responds with streaming output using specialized prompts tailored to their domain expertise.

### Step 3: Final Output
The system displays both the routing decision and the expert's response.

## Running the Coordinator

### Prerequisites
- Ollama installed and running locally
- R1 abliterated model available in Ollama
- Python environment with the AI Lego Bricks dependencies

### Basic Usage

```bash
# Navigate to the examples directory
cd agent_orchestration/examples

# Run the coordinator
python -m agent_orchestration.orchestrator ai_coordinator_agent.json
```

You'll be prompted to enter your question, and the system will:
1. Route your request to the appropriate expert
2. Stream the expert's response in real-time
3. Display routing information

### Testing with Sample Queries

Try these example queries to see the routing in action:

**Japanese Expert:**
- "How do you say 'thank you' in Japanese?"
- "What are the customs for visiting a Japanese temple?"
- "When is the best time to see cherry blossoms in Tokyo?"

**Home Assistant Expert:**
- "How do I set up a motion sensor automation?"
- "What's the best smart thermostat for my HVAC system?"
- "Help me troubleshoot my Zigbee devices"

**General Expert:**
- "Explain how binary search works"
- "What are the benefits of renewable energy?"
- "Help me write a professional email"

## Evaluation

The coordinator system uses the standard prompt evaluation framework for testing. Run the comprehensive evaluation suite:

```bash
cd prompt
python run_coordinator_evaluations.py
```

This evaluates both routing accuracy and expert response quality, providing:
- Overall coordinator system score
- Individual expert performance
- Detailed concept verification results
- Recommendations for improvement

### Manual Evaluation

You can also run individual evaluations using the framework directly:

#### Routing Accuracy Test
```python
from concept_evaluation_service import ConceptEvaluationService
from concept_eval_storage import create_concept_eval_storage

# Test routing decisions
storage = create_concept_eval_storage('auto')
service = ConceptEvaluationService(storage)

with open('coordinator_routing_evaluation.json', 'r') as f:
    results = service.run_evaluation(json.load(f))
    print(f'Routing Accuracy: {results.overall_score:.1%}')
```

#### Expert Quality Test
```python
# Test expert response quality
with open('coordinator_expert_evaluation.json', 'r') as f:
    expert_evals = json.load(f)
    
for expert_name, expert_eval in expert_evals['expert_evaluations'].items():
    results = service.run_evaluation(expert_eval)
    print(f'{expert_name}: {results.overall_score:.1%}')
```

### Evaluation Features

- **Routing Accuracy**: Tests 8 scenarios including edge cases with cross-domain queries
- **Expert Quality**: Tests domain-specific knowledge and response quality
- **Concept Verification**: Uses LLM judges to verify specific concepts in responses
- **Weighted Scoring**: Important checks (like correct routing) have higher weights
- **History Tracking**: Compare results over time as prompts are refined

## Customization

### Adding New Experts

1. Add a new routing option in the coordinator prompt
2. Create a new expert response step with specialized system prompt
3. Update the routing condition to include the new expert
4. Add test cases to the evaluation data

### Modifying Expert Prompts

Edit the `system_message` field in each expert's configuration to customize their behavior:

- **Japanese Expert**: Adjust cultural sensitivity, language complexity, or regional focus
- **Home Assistant Expert**: Add support for new devices, protocols, or platforms
- **General Expert**: Modify tone, technical depth, or domain coverage

### Tuning Model Parameters

Adjust these settings in each step's config:
- `temperature`: Controls response creativity (0.1-0.4 recommended)
- `model`: Change to different Ollama models
- `max_tokens`: Adjust response length limits

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check model availability: `ollama list`
   - Install R1 model: `ollama pull r1`

2. **Poor Routing Accuracy**
   - Review coordinator system prompt for clarity
   - Add more specific routing criteria
   - Test with edge cases and adjust prompts

3. **Expert Response Quality**
   - Refine expert system prompts
   - Adjust temperature settings
   - Add more domain-specific guidance

### Model Requirements

- **R1 Abliterated Model**: Ensure this model is available in your Ollama installation
- **Alternative Models**: You can substitute with other models by changing the `model` field in the JSON config

## Architecture Benefits

Using the orchestrator JSON approach provides:

1. **No Code Changes**: Modify behavior by editing JSON configuration
2. **Easy Testing**: Swap models and prompts without touching Python
3. **Modular Design**: Each expert is independently configurable
4. **Streaming Support**: Real-time response delivery for better UX
5. **Evaluation Framework**: Built-in testing and metrics collection

## Next Steps

1. **Add More Experts**: Create specialists for other domains (medical, legal, etc.)
2. **Implement Confidence Thresholding**: Route uncertain cases to human review
3. **Add Context Memory**: Maintain conversation history across exchanges
4. **Create Web Interface**: Build a web UI for easier interaction
5. **Performance Optimization**: Cache common routing decisions for faster response