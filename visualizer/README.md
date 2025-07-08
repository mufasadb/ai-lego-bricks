# Agent Workflow Visualizer

A powerful visualization service for JSON orchestrated agent workflows using interactive Mermaid diagrams. This tool helps make complex agent flows more understandable and debuggable.

## Features

- 🔍 **Visual Workflow Analysis**: Convert JSON workflows into interactive Mermaid flowcharts
- 🎨 **Color-Coded Step Types**: Different node styles for 35+ step types (LLM, memory, conditions, etc.)
- 📊 **Workflow Statistics**: Complexity scores, step counts, and flow analysis
- 🌐 **Web Interface**: Drag-and-drop file upload or paste JSON directly
- 📁 **Example Integration**: Browse and visualize existing agent examples
- 💾 **Export Options**: Download diagrams as PNG images
- 🔄 **Loop Detection**: Special highlighting for iterative workflows
- 📈 **Complexity Scoring**: Automated workflow complexity assessment

## Quick Start

### Web Interface (Recommended)

1. **Start the web server:**
   ```bash
   cd visualizer/web
   pip install -r ../requirements.txt
   python app.py
   ```

2. **Open your browser:**
   ```
   http://localhost:5001
   ```

3. **Upload or paste your JSON workflow** and see the visual diagram instantly!

### Command Line Usage

```python
from visualizer import WorkflowParser, MermaidGenerator

# Parse workflow
parser = WorkflowParser()
workflow = parser.parse_workflow_file("my_agent.json")

# Generate diagram
generator = MermaidGenerator()
result = generator.generate_with_statistics(workflow)

print(result['diagram'])  # Mermaid diagram code
print(f"Complexity: {result['complexity_score']}")
```

### Programmatic Usage

You can also use the visualizer programmatically to process workflows and generate diagrams:

```python
from visualizer import WorkflowParser, MermaidGenerator

# Parse workflow file
parser = WorkflowParser()
workflow = parser.parse_workflow_file("path/to/workflow.json")

# Generate diagram with statistics
generator = MermaidGenerator()
result = generator.generate_with_statistics(workflow)

# Access components
print(result['diagram'])           # Mermaid diagram code
print(result['complexity_score'])  # Complexity score
print(result['statistics'])        # Detailed statistics
```

## Supported Step Types

The visualizer supports all agent orchestration step types with unique visual styling:

| Step Type | Visual Style | Purpose |
|-----------|--------------|---------|
| `input` | Stadium (Blue) | Data collection and user input |
| `llm_chat` | Rectangle (Green) | LLM text generation |
| `llm_vision` | Rectangle (Green) | LLM image analysis |
| `condition` | Diamond (Orange) | Conditional logic and routing |
| `loop` | Circle (Pink) | Iteration and repetition |
| `memory_store` | Subroutine (Lime) | Memory operations |
| `memory_retrieve` | Subroutine (Lime) | Memory search |
| `document_processing` | Rectangle (Blue) | PDF and document handling |
| `output` | Stadium (Purple) | Results and final output |
| `human_approval` | Stadium (Red) | Human-in-the-loop processes |

## File Structure

```
visualizer/
├── __init__.py                 # Package initialization
├── workflow_parser.py          # JSON workflow parsing
├── mermaid_generator.py        # Mermaid diagram generation
├── requirements.txt            # Python dependencies
├── web/                        # Web interface
│   ├── app.py                 # Flask application
│   ├── templates/
│   │   └── index.html         # Main web interface
│   └── static/
│       ├── style.css          # Styling
│       └── script.js          # Frontend logic
└── examples/
    └── output/                # Generated diagrams
```

## Web Interface Features

### Upload Options
- **File Upload**: Drag and drop JSON files
- **Paste JSON**: Copy and paste workflow JSON directly
- **Examples**: Browse and load example workflows

### Visualization Features
- **Interactive Diagrams**: Zoom, pan, and explore complex workflows
- **Statistics View**: Step counts, complexity metrics, and type breakdown
- **Raw JSON View**: Inspect the original workflow configuration
- **Legend**: Understand diagram symbols and styling

### Export Options
- **PNG Download**: Save diagrams as high-quality images
- **Markdown Files**: Complete documentation with diagrams and stats

## Integration with Agent Orchestration

This visualizer is designed specifically for the AI Lego Bricks agent orchestration system. It automatically:

- Detects all step types and their configurations
- Maps input/output relationships between steps
- Identifies conditional routing and loop-back patterns
- Analyzes workflow complexity and potential issues
- Provides actionable insights for workflow optimization

## Example Workflows

The visualizer works with all existing agent examples:

- **Simple Chat Agents**: Basic conversational workflows
- **Document Analysis**: PDF processing with memory storage
- **Multi-step Analysis**: Complex reasoning chains
- **Loop-back Workflows**: Iterative improvement patterns
- **Human-in-the-loop**: Approval and feedback workflows
- **Streaming Chat**: Real-time conversation handling

## Development

### Adding New Step Types

To add support for new step types:

1. Update `step_types` set in `WorkflowParser`
2. Add visual styling in `MermaidGenerator.step_styles`
3. Test with example workflows

### Customizing Visualization

The Mermaid generator supports:
- Custom node shapes and colors
- Connection styling (solid, dashed, colored)
- Automatic layout optimization
- Responsive design for different screen sizes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
2. **Missing Examples**: Verify the `agent_orchestration/examples/` directory exists
3. **Web Interface Not Loading**: Check that Flask is installed and port 5001 is available
4. **Diagram Not Rendering**: Verify JSON format and Mermaid syntax

### Debug Mode

Run the Flask app in debug mode for detailed error messages:

```python
app.run(debug=True)
```

## Contributing

To extend the visualizer:

1. Add new step type support in the parser
2. Create custom visual styles for new step types  
3. Enhance the web interface with additional features
4. Add new export formats or analysis capabilities

## License

Part of the AI Lego Bricks project - modular building blocks for LLM agentic work.