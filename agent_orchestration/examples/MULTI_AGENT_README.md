# Multi-Agent Workflow Examples

This directory contains examples demonstrating multi-agent workflows where a parent agent can call and coordinate child specialist agents.

## Example Files

### 1. `multi_agent_coordinator.json`
**Parent Agent** - Coordinates the overall text analysis workflow
- Gets user input text
- Validates input quality
- Calls specialist agent for detailed analysis
- Enhances results with coordinator insights
- Provides comprehensive final output

### 2. `text_analyzer_specialist.json` 
**Child Agent** - Specialist for text analysis tasks
- Performs detailed text analysis (topics, sentiment, entities)
- Extracts key terms and concepts
- Generates executive summary
- Returns structured results to parent

## Multi-Agent Features Demonstrated

### Agent Composition
```json
{
  "id": "call_text_specialist",
  "type": "agent_call",
  "config": {
    "agent_file": "./text_analyzer_specialist.json",
    "timeout": 180,
    "inherit_context": true,
    "return_outputs": ["specialist_results"]
  }
}
```

### Data Flow Between Agents
- **Parent → Child**: User input text passed to specialist
- **Child → Parent**: Analysis results returned to coordinator  
- **Parent Processing**: Coordinator enhances specialist results

### Context Inheritance
- Child agent receives parent's execution context
- Seamless data passing via `from_step` references
- Shared configuration and service access

## Usage

### Prerequisites
- Implement the `AGENT_CALL` step type (see implementation plan)
- Ensure both JSON files are in the same directory
- Configure LLM provider (Gemini) credentials

### Running the Example

```bash
# Once agent_call is implemented:
cd agent_orchestration/examples
python -m agent_orchestration.orchestrator multi_agent_coordinator.json
```

### Expected Flow
1. **Input**: User provides text to analyze
2. **Validation**: Coordinator checks input quality
3. **Specialist Call**: Text analyzer performs detailed analysis
4. **Enhancement**: Coordinator adds insights and recommendations
5. **Output**: Comprehensive analysis combining both agents' work

## Testing Multi-Agent Behavior

### Sample Input Text
```
"Artificial intelligence is revolutionizing healthcare through predictive analytics, 
diagnostic imaging, and personalized treatment plans. Machine learning algorithms 
can now detect patterns in medical data that human doctors might miss, leading to 
earlier disease detection and more effective treatments. However, concerns about 
data privacy, algorithmic bias, and the need for human oversight remain significant 
challenges in AI healthcare adoption."
```

### Expected Results Structure
```json
{
  "original_text": "...",
  "specialist_results": {
    "full_analysis": "Detailed analysis from specialist...",
    "keywords": "AI, healthcare, machine learning, diagnostics...",
    "summary": "Executive summary..."
  },
  "coordinator_insights": "Additional recommendations and applications..."
}
```

## Architecture Benefits

### Modularity
- **Reusable Specialists**: Text analyzer can be called by multiple coordinators
- **Clear Separation**: Each agent has distinct responsibilities
- **Independent Testing**: Agents can be tested separately

### Scalability  
- **Nested Hierarchies**: Coordinators can call other coordinators
- **Parallel Specialist Calls**: Multiple specialists can work simultaneously
- **Resource Management**: Timeout and context controls

### Maintainability
- **Pure JSON Configuration**: No Python code changes needed
- **Version Control**: Each agent workflow is independently versioned
- **Easy Debugging**: Clear execution paths and data flow

## Error Handling

### Timeout Protection
- Individual agent calls have configurable timeouts
- Parent agent continues with error handling if child fails

### Validation
- Input validation prevents invalid data from reaching specialists
- Graceful error messages for user-facing issues

### Context Isolation
- Failed child agents don't crash parent workflow
- Clean separation of concerns between agent layers

## Next Steps

1. **Implement Core Features**: Add `AGENT_CALL` step type to models and handlers
2. **Test the Example**: Run multi-agent coordinator with sample text
3. **Extend Functionality**: Add more specialist agents (sentiment, language detection, etc.)
4. **Optimize Performance**: Add caching and parallel execution for multiple specialists