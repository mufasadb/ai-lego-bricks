# AI Lego Bricks Testing Framework

This directory contains the comprehensive test suite for the AI Lego Bricks project, providing unit tests, integration tests, performance benchmarks, and testing infrastructure.

## Test Structure

```
tests/
├── conftest.py                 # pytest configuration and shared fixtures
├── test_runner.py             # convenient test runner script
├── unit/                      # unit tests for individual modules
│   ├── test_agent_orchestration.py
│   ├── test_llm_services.py
│   ├── test_memory_services.py
│   ├── test_chat_services.py
│   ├── test_prompt_services.py
│   ├── test_tts_services.py
│   └── test_chunking_service.py
├── integration/               # integration tests for workflows
│   └── test_workflow_execution.py
├── performance/              # performance benchmarks
│   ├── test_memory_benchmarks.py
│   └── test_llm_benchmarks.py
└── fixtures/                 # test data and mock responses
    ├── mock_responses.py
    ├── test_workflows.json
    └── sample_data.py
```

## Running Tests

### Using the Test Runner (Recommended)

The test runner provides convenient commands for different test scenarios:

```bash
# Run all unit tests
python tests/test_runner.py unit

# Run integration tests
python tests/test_runner.py integration

# Run performance tests
python tests/test_runner.py performance

# Run all tests
python tests/test_runner.py all

# Run quick development tests
python tests/test_runner.py quick

# Run quality checks (linting, type checking, security)
python tests/test_runner.py quality

# Run full CI pipeline
python tests/test_runner.py ci

# Generate coverage report
python tests/test_runner.py coverage
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/                    # Unit tests only
pytest tests/integration/ -m integration  # Integration tests only
pytest tests/performance/ -m performance  # Performance tests only

# Run tests with coverage
pytest --cov=ailego --cov-report=html

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/test_llm_services.py

# Run specific test
pytest tests/unit/test_llm_services.py::TestGenerationService::test_generate_text
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual modules and classes in isolation with mocked dependencies.

**Coverage includes:**
- Agent orchestration and workflow execution
- LLM service providers and generation
- Memory storage and retrieval services
- Chat and conversation management
- Prompt management and evaluation
- Text-to-speech services
- Text chunking functionality

**Key features:**
- All external dependencies mocked
- Fast execution (< 10 seconds total)
- High code coverage (>80% target)
- Isolated test environment

### Integration Tests (`tests/integration/`)

Test complete workflows and service interactions end-to-end.

**Test scenarios:**
- Complete workflow execution
- Multi-step agent interactions
- Service interoperability
- Error handling and recovery
- Provider switching
- Memory-enhanced conversations

**Requirements:**
- Mock environment variables provided
- External services mocked with realistic responses
- Focus on data flow and integration points

### Performance Tests (`tests/performance/`)

Benchmark critical operations and identify performance regressions.

**Benchmarks include:**
- Memory operation performance (store/retrieve)
- LLM generation speed and throughput
- Concurrent operation handling
- Scalability with data size
- Cache effectiveness
- Resource usage patterns

**Metrics tracked:**
- Execution time (average, median, 95th percentile)
- Throughput (operations per second)
- Resource usage (memory, CPU)
- Scalability characteristics

## Test Configuration

### pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = """
    --cov=ailego 
    --cov=agent_orchestration 
    --cov=llm 
    --cov=memory 
    --cov=chat 
    --cov=prompt 
    --cov=tts 
    --cov=chunking
    --cov-report=html 
    --cov-report=term-missing 
    --cov-report=xml
    --strict-markers
    --strict-config
    -v
"""
```

### Test Markers

- `@pytest.mark.integration` - Integration tests requiring external services
- `@pytest.mark.performance` - Performance benchmarks and load tests
- `@pytest.mark.slow` - Slow-running tests (>5 seconds)

### Fixtures and Mocks

**Shared fixtures** (`conftest.py`):
- `mock_environment_variables` - Mock environment variables
- `mock_llm_service` - Mock LLM service with realistic responses
- `mock_memory_service` - Mock memory service with CRUD operations
- `mock_tts_service` - Mock TTS service
- `temporary_workspace` - Temporary directory for test files
- `sample_workflow_file` - Sample workflow JSON for testing

**Mock responses** (`fixtures/mock_responses.py`):
- Realistic API responses for different providers
- Sample workflow execution results
- Test data for various scenarios

## CI/CD Integration

### GitHub Actions Workflow

The test suite is integrated with GitHub Actions (`.github/workflows/test.yml`) and includes:

**Test Matrix:**
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating system: Ubuntu (with caching)

**Pipeline stages:**
1. **Linting** - Code formatting and style checks
2. **Type checking** - Static type analysis with mypy
3. **Unit tests** - Fast, isolated tests with parallel execution
4. **Integration tests** - End-to-end workflow testing
5. **Performance tests** - Benchmarks (main branch only)
6. **Security scanning** - Bandit and Safety checks
7. **Coverage reporting** - Codecov integration
8. **Build verification** - Package build and validation

### Quality Gates

Tests must pass these criteria:
- **Coverage**: >80% for all modules
- **Performance**: Key operations <100ms (95th percentile)
- **Security**: No high-severity vulnerabilities
- **Type safety**: No mypy errors
- **Style**: Black formatting and flake8 compliance

## Development Workflow

### Adding New Tests

1. **Choose test type**: Unit, integration, or performance
2. **Follow naming convention**: `test_*.py` files, `Test*` classes, `test_*` methods
3. **Use appropriate fixtures**: Import from `conftest.py` or create module-specific ones
4. **Mock external dependencies**: Keep tests isolated and fast
5. **Add performance assertions**: Include timing/resource checks for critical paths

### Test-Driven Development

```bash
# 1. Write failing test
pytest tests/unit/test_new_feature.py::test_new_functionality -v

# 2. Implement feature
# ... write code ...

# 3. Verify test passes
pytest tests/unit/test_new_feature.py::test_new_functionality -v

# 4. Run full test suite
python tests/test_runner.py unit
```

### Debugging Tests

```bash
# Run with verbose output and stop on first failure
pytest -v -x

# Run with pdb debugging
pytest --pdb

# Run specific test with detailed output
pytest tests/unit/test_module.py::TestClass::test_method -v -s

# Show test coverage for specific module
pytest --cov=module_name --cov-report=term-missing
```

## Performance Monitoring

### Benchmark Execution

Performance tests automatically track:
- **Execution time** trends
- **Resource usage** patterns
- **Throughput** metrics
- **Scalability** characteristics

### Performance Thresholds

Current thresholds (adjust as needed):
- Memory operations: <100ms (95th percentile)
- LLM generation: <500ms (average)
- Workflow execution: <1s (simple workflows)
- Concurrent operations: <5s (20 operations, 5 workers)

## Troubleshooting

### Common Issues

**Tests failing locally but passing in CI:**
- Check Python version compatibility
- Verify environment variables are set
- Ensure dependencies are installed correctly

**Slow test execution:**
- Use `pytest -n auto` for parallel execution
- Focus on unit tests during development
- Use `--lf` to run only last failed tests

**Import errors:**
- Install package in development mode: `pip install -e .`
- Check PYTHONPATH includes project root
- Verify all dependencies are installed

**Mock-related issues:**
- Check mock setup in `conftest.py`
- Verify fixture scope (function vs session)
- Ensure external dependencies are properly mocked

### Getting Help

1. **Check test logs**: Look for detailed error messages and stack traces
2. **Review CI output**: GitHub Actions provides detailed step-by-step logs
3. **Run specific tests**: Isolate the failing test to understand the issue
4. **Check dependencies**: Ensure all required packages are installed

## Contributing

### Test Guidelines

1. **Write tests first**: Follow TDD practices when possible
2. **Keep tests simple**: One concept per test method
3. **Use descriptive names**: Test names should explain what they verify
4. **Mock external dependencies**: Tests should be isolated and fast
5. **Include edge cases**: Test error conditions and boundary cases
6. **Maintain test data**: Update fixtures when adding new test scenarios

### Code Coverage

Aim for high test coverage:
- **New code**: 100% coverage for new functionality
- **Existing code**: Minimum 80% coverage
- **Critical paths**: 100% coverage for core workflows
- **Edge cases**: Cover error conditions and unusual inputs

The comprehensive test suite ensures code quality, catches regressions early, and provides confidence for refactoring and new feature development.