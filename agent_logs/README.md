# Agent Logs

This directory contains logging utilities and log files for agent workflow tracking.

## Files

### `agent_logger.py`
Simple logging utility for agent examples that provides:
- Structured JSON logging to files
- Stage-based workflow tracking
- Model switching event logging
- Error tracking with context
- Console output for immediate feedback

### `log_viewer.py`
Command-line utility for viewing and analyzing agent log files:
- Human-readable formatting of JSON log entries
- Optional filtering by stage name
- Toggle display of model responses
- List available log files with metadata
- Summary statistics for log analysis

### Log Files
Log files are automatically created with the format: `{agent_name}_{timestamp}.log`

Example: `MultiModelPDFAgent_20240704_143052.log`

## Usage

### Agent Logger

```python
from agent_logs.agent_logger import AgentLogger

# Initialize logger
logger = AgentLogger("MyAgent")

# Log workflow stages
logger.log_stage("PROCESSING", "Starting data processing", {"input_size": 1000})

# Log model switches
logger.log_model_switch("llama2", "codellama", "code generation")

# Log processing stages
logger.log_processing_start("DATA_TRANSFORM", {"input_format": "csv"})
logger.log_processing_end("DATA_TRANSFORM", {"output_format": "json"})

# Log errors
logger.log_error("PROCESSING", exception, {"context": "data validation"})
```

### Log Viewer

```bash
# List available log files
python agent_logs/log_viewer.py --list

# View a specific log file
python agent_logs/log_viewer.py MultiModelPDFAgent_20240704_143052.log

# Filter by stage
python agent_logs/log_viewer.py --filter "MODEL_SWITCH" agent.log

# Hide model responses for cleaner view
python agent_logs/log_viewer.py --no-responses agent.log
```

## Log Format

Each log entry is a JSON object with:
- `timestamp`: ISO format timestamp
- `agent`: Agent name
- `session`: Session ID (timestamp-based)
- `stage`: Stage identifier
- `message`: Human-readable message
- `data`: Additional structured data

## Example Log Entry

```json
{
  "timestamp": "2024-07-04T14:30:52.123456",
  "agent": "MultiModelPDFAgent",
  "session": "20240704_143052",
  "stage": "MODEL_SWITCH",
  "message": "Switching from llama2 to codellama",
  "data": {
    "from_model": "llama2",
    "to_model": "codellama",
    "purpose": "code generation"
  }
}
```

## Benefits

1. **Monitoring**: Track agent workflow progress in real-time
2. **Debugging**: Detailed error context and stage information
3. **Analysis**: Structured data for post-execution analysis
4. **Auditability**: Complete workflow history with timestamps
5. **Simplicity**: Easy to read and parse log format