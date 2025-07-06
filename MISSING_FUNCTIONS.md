# Missing Functions Documentation

This document tracks functions that are tested but not yet implemented in the codebase.

## Missing Functions by Module

### llm.generation_service
- `create_generation_service()` - Factory function to create generation service instances
- `GenerationService` class may be missing some methods

### chat.conversation_service
- `create_conversation_service()` - Factory function to create conversation service instances

### tts.tts_service
- `create_tts_service()` - Factory function to create TTS service instances

### agent_orchestration.orchestrator
- `WorkflowOrchestrator` class - Main workflow orchestration class
- Import issue with relative imports from models module

### agent_orchestration.models
- Import path issue: `from models import` should be `from .models import`

### prompt.concept_judge
- Import issue: `from llm_factory import` should be `from llm.llm_factory import`

## Import Path Issues

### Relative vs Absolute Imports
1. **agent_orchestration/orchestrator.py**: Line 13 - `from models import` should be `from .models import`
2. **prompt/concept_judge.py**: Line 18 - `from llm_factory import` should be `from llm.llm_factory import`

### Missing Classes/Functions
Several factory functions are expected by tests but don't exist:
- `create_generation_service`
- `create_conversation_service` 
- `create_tts_service`
- `WorkflowOrchestrator` class

## Next Steps

1. Fix import paths to use proper relative/absolute imports
2. Implement missing factory functions or adjust tests to match existing API
3. Ensure all classes referenced in tests actually exist
4. Add missing imports to module `__init__.py` files if needed

## Status
- **Created**: During test framework setup
- **Priority**: High - blocking initial test execution
- **Resolution**: Fix imports first, then implement missing functions as needed