# AI Lego Bricks Testing Framework

This directory contains a comprehensive testing framework with three main components:

1. **Integration Tests** (`integration/`) - Real API calls with VCR recording
2. **Unit Tests** (`unit/`) - Fast tests using recorded cassettes
3. **Cassettes** (`cassettes/`) - Recorded HTTP interactions

## Quick Start

### Install Dependencies
```bash
pip install -e ".[dev]"
```

### Run Unit Tests (Fast, No Network)
```bash
pytest tests/unit/ --record-mode=none
```

### Run Integration Tests (Records Real API Calls)
```bash
# First time - records new cassettes
pytest tests/integration/ --record-mode=once

# Add new recordings while keeping existing ones
pytest tests/integration/ --record-mode=new_episodes
```

### Update All Cassettes
```bash
pytest tests/integration/ --record-mode=rewrite
```

## Environment Setup

### For Integration Tests
Create a `.env` file in the project root with your API keys:

```bash
# Required for LLM services
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GEMINI_API_KEY=your-gemini-key

# Required for memory services
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### For Unit Tests
No environment variables needed - uses mocked responses from cassettes.

## Recording Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `once` | Record missing interactions, replay existing | Initial recording |
| `new_episodes` | Record new interactions, keep existing | Adding new tests |
| `none` | Only replay, no recording | CI/unit tests |
| `all` | Always record | Debugging |
| `rewrite` | Delete and re-record everything | API changes |

## Test Organization

```
tests/
├── integration/           # Real API calls, populates cassettes
│   ├── test_chat_integration.py
│   ├── test_memory_integration.py
│   └── test_orchestrator_integration.py
├── unit/                  # Fast tests using cassettes
│   ├── test_chat_unit.py
│   ├── test_memory_unit.py
│   └── test_orchestrator_unit.py
├── cassettes/             # Recorded HTTP interactions
│   ├── chat/
│   ├── memory/
│   └── orchestrator/
├── conftest.py           # Pytest configuration
├── vcr_config.py         # VCR security settings
└── README.md            # This file
```

## Security

All cassettes are automatically filtered to remove:
- API keys from headers (`Authorization`, `X-API-Key`, etc.)
- Sensitive query parameters (`api_key`, `token`, etc.)
- Sensitive POST data (`client_secret`, `password`, etc.)

**Always review cassettes before committing** to ensure no secrets leaked through.

## Command Line Options

### Custom Pytest Options
```bash
# Record mode control
pytest --record-mode=once
pytest --record-mode=none

# Run specific test types
pytest --integration-only
pytest --unit-only

# Standard pytest filtering
pytest -m "integration"
pytest -m "unit"
pytest -m "not slow"
```

### With Tape Management CLI
```bash
# Record fresh cassettes for all services
ailego test record

# Update specific service cassettes
ailego test update-tapes --service=chat

# Clean outdated cassettes
ailego test clean-tapes
```

## CI/CD Integration

### GitHub Actions - Unit Tests (Fast)
```yaml
- name: Run Unit Tests
  run: pytest tests/unit/ --record-mode=none
```

### Manual Trigger - Integration Tests (Records)
```yaml
- name: Update Cassettes
  run: pytest tests/integration/ --record-mode=new_episodes
```

## Best Practices

1. **Start with Integration Tests**: Write integration tests first to establish the API contract
2. **Generate Unit Tests**: Create corresponding unit tests that use the recorded cassettes
3. **Keep Cassettes Fresh**: Regenerate cassettes when APIs change
4. **Review Security**: Always check cassettes for leaked credentials
5. **Test Both Paths**: Ensure both integration and unit tests cover the same scenarios

## Troubleshooting

### Cassette Playback Errors
- **Error**: Cassette not found
- **Solution**: Run integration tests with `--record-mode=once`

### API Key Errors in Unit Tests
- **Error**: Missing API key
- **Solution**: Unit tests shouldn't need real keys - check your mocking

### Outdated Cassettes
- **Error**: API response format changed
- **Solution**: Re-record with `--record-mode=rewrite`

### Security Concerns
- **Issue**: API key in cassette
- **Solution**: Update `vcr_config.py` filters and re-record