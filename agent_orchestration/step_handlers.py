"""
Step handlers for different workflow step types
"""

from typing import Any, Dict, Callable, Type, List

try:
    from .models import (
        StepConfig,
        StepType,
        ExecutionContext,
        ThinkingTokensMode,
    )
    from .streaming_buffer import StreamBuffer
except ImportError:
    from models import (
        StepConfig,
        StepType,
        ExecutionContext,
        ThinkingTokensMode,
    )
    from streaming_buffer import StreamBuffer
from pydantic import BaseModel
import importlib
import json
import pathlib
from datetime import datetime
import sys
import os

# Conditional imports that work both as package and standalone
try:
    from llm.llm_types import LLMProvider, VisionProvider
    from pdf_to_text.pdf_to_text_service import PDFExtractOptions
    from pdf_to_text.visual_to_text_service import (
        VisualToTextService,
        VisualExtractOptions,
    )
except ImportError:
    try:
        from llm.llm_types import LLMProvider, VisionProvider
        from pdf_to_text.pdf_to_text_service import PDFExtractOptions
        from pdf_to_text.visual_to_text_service import (
            VisualToTextService,
            VisualExtractOptions,
        )
    except ImportError:
        # Fallback when dependencies not available
        LLMProvider = None
        VisionProvider = None
        PDFExtractOptions = None
        VisualToTextService = None
        VisualExtractOptions = None


def _serialize_for_json(obj):
    """Helper function to serialize objects for JSON output"""
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        # Handle objects with __dict__ like DynamicSchema
        return {k: _serialize_for_json(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        # Fallback to string representation
        return str(obj)


class StepHandlerRegistry:
    """Registry of step handlers for different step types"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.handlers: Dict[StepType, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register all default step handlers"""
        self.handlers[StepType.INPUT] = self._handle_input
        self.handlers[StepType.DOCUMENT_PROCESSING] = self._handle_document_processing
        self.handlers[StepType.MEMORY_STORE] = self._handle_memory_store
        self.handlers[StepType.MEMORY_RETRIEVE] = self._handle_memory_retrieve
        self.handlers[StepType.LLM_CHAT] = self._handle_llm_chat
        self.handlers[StepType.LLM_STRUCTURED] = self._handle_llm_structured
        self.handlers[StepType.LLM_VISION] = self._handle_llm_vision
        self.handlers[StepType.CHUNK_TEXT] = self._handle_chunk_text
        self.handlers[StepType.CONDITION] = self._handle_condition
        self.handlers[StepType.LOOP] = self._handle_loop
        self.handlers[StepType.OUTPUT] = self._handle_output
        self.handlers[StepType.FILE_OUTPUT] = self._handle_file_output
        self.handlers[StepType.HUMAN_APPROVAL] = self._handle_human_approval
        self.handlers[StepType.CONCEPT_EVALUATION] = self._handle_concept_evaluation
        # Conversation management handlers
        self.handlers[StepType.START_CONVERSATION] = self._handle_start_conversation
        self.handlers[StepType.ADD_TO_CONVERSATION] = self._handle_add_to_conversation
        self.handlers[StepType.CONTINUE_CONVERSATION] = (
            self._handle_continue_conversation
        )
        # Audio processing handlers
        self.handlers[StepType.TTS] = self._handle_tts
        self.handlers[StepType.STT] = self._handle_stt
        # Python function execution handlers
        self.handlers[StepType.PYTHON_FUNCTION] = self._handle_python_function
        # Graph memory formatting handlers
        self.handlers[StepType.GRAPH_MEMORY_FORMAT] = self._handle_graph_memory_format
        # Tool calling handlers
        self.handlers[StepType.TOOL_CALL] = self._handle_tool_call
        # HTTP request handlers
        self.handlers[StepType.HTTP_REQUEST] = self._handle_http_request
        # Multi-agent handlers
        self.handlers[StepType.AGENT_CALL] = self._handle_agent_call

    def get_handler(self, step_type: StepType) -> Callable:
        """Get handler for a specific step type"""
        return self.handlers.get(step_type)

    def register_handler(self, step_type: StepType, handler: Callable):
        """Register a custom handler for a step type"""
        self.handlers[step_type] = handler

    def _process_json_props(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Process JSON props and add them to the inputs context"""
        if not step.json_props:
            return inputs

        # Import JsonStructure class for processing
        from prompt.prompt_models import JsonStructure

        # Build template context for JSON prop rendering
        template_context = {}
        template_context.update(inputs)  # Step inputs
        template_context.update(context.global_variables)  # Global variables

        # Process each JSON prop
        enhanced_inputs = inputs.copy()
        for prop_name, prop_config in step.json_props.items():
            try:
                # Create JsonStructure instance from config
                json_structure = JsonStructure(
                    structure=prop_config.get("structure", {}),
                    description=prop_config.get("description"),
                    variables=prop_config.get("variables", {}),
                    required_variables=prop_config.get("required_variables", []),
                )

                # Render JSON prop and add to inputs
                enhanced_inputs[f"json_{prop_name}"] = json_structure.render(
                    template_context
                )
                enhanced_inputs[f"json_{prop_name}_dict"] = (
                    json_structure.render_as_dict(template_context)
                )

            except Exception as e:
                raise ValueError(f"Failed to process JSON prop '{prop_name}': {e}")

        return enhanced_inputs

    def _process_template_string(
        self, template_str: str, inputs: Dict[str, Any], context: ExecutionContext
    ) -> str:
        """Process template variables in a string like {{variable_name}}"""
        import re

        # Build template context
        template_context = {}
        template_context.update(inputs)
        template_context.update(context.global_variables)

        # Find all template variables in the string
        pattern = r"\{\{([^}]+)\}\}"
        matches = re.findall(pattern, template_str)

        result = template_str
        for match in matches:
            var_name = match.strip()
            if var_name in template_context:
                result = result.replace(
                    f"{{{{{var_name}}}}}", str(template_context[var_name])
                )
            else:
                raise ValueError(f"Template variable '{var_name}' not found in context")

        return result

    def _handle_input(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle input step - typically collects user input or external data"""
        # Check for workflow inputs first
        for output_key in step.outputs:
            if output_key in inputs:
                return inputs[output_key]

        # Check common input keys
        if "user_input" in inputs:
            return inputs["user_input"]
        elif "user_query" in inputs:
            return inputs["user_query"]
        elif "user_message" in inputs:
            return inputs["user_message"]

        # Check for configured value
        if "value" in step.config:
            return step.config["value"]
        elif "prompt" in step.config:
            # Only prompt interactively if no inputs provided
            if not inputs:
                return input(step.config["prompt"])
            else:
                # Return first available input if we have inputs but no direct match
                return list(inputs.values())[0]
        else:
            return inputs

    def _handle_concept_evaluation(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle concept evaluation step - run prompt evaluation using LLM judge"""
        try:
            # Import concept evaluation components
            from prompt.concept_eval_storage import create_concept_eval_storage
            from prompt.concept_evaluation_service import ConceptEvaluationService

            # Get configuration
            eval_id = step.config.get("eval_id")
            if not eval_id:
                raise ValueError("eval_id required for concept evaluation")

            llm_provider = step.config.get("llm_provider", "gemini")
            judge_model = step.config.get("judge_model", "gemini")
            storage_backend = step.config.get("storage_backend", "auto")
            save_results = step.config.get("save_results", True)

            # Create storage backend
            storage = create_concept_eval_storage(storage_backend)

            # Create evaluation service
            eval_service = ConceptEvaluationService(
                storage_backend=storage,
                default_llm_provider=llm_provider,
                default_judge_model=judge_model,
            )

            # Check if we need to create evaluation on-the-fly
            if "evaluation_definition" in step.config:
                # Inline evaluation definition
                from concept_eval_models import ConceptEvalDefinition

                eval_def_data = step.config["evaluation_definition"]
                eval_def = ConceptEvalDefinition(**eval_def_data)

                # Save it temporarily
                storage.save_evaluation_definition(eval_def)
                eval_id = eval_def.eval_id

            # Handle context variables for test cases
            context_variables = inputs.get("context_variables", {})
            if context_variables:
                # If context variables provided, we need to modify the evaluation
                # to use these variables in test cases
                eval_def = storage.get_evaluation_definition(eval_id)
                if eval_def:
                    # Update test case contexts with provided variables
                    for test_case in eval_def.test_cases:
                        test_case["context"].update(context_variables)

                    # Create a temporary evaluation with updated context
                    temp_eval_id = f"{eval_id}_temp_{context.workflow_id}"
                    eval_def.eval_id = temp_eval_id
                    storage.save_evaluation_definition(eval_def)
                    eval_id = temp_eval_id

            # Run the evaluation
            results = eval_service.run_evaluation_by_id(
                eval_id, llm_provider=llm_provider, save_results=save_results
            )

            if not results:
                raise ValueError(
                    f"Failed to run evaluation '{eval_id}' - evaluation not found or execution failed"
                )

            # Format results for workflow consumption
            formatted_results = {
                "evaluation_name": results.evaluation_name,
                "overall_score": results.overall_score,
                "grade": self._get_performance_grade(results.overall_score),
                "passed_test_cases": results.passed_test_cases,
                "total_test_cases": results.total_test_cases,
                "pass_rate": (
                    results.passed_test_cases / results.total_test_cases
                    if results.total_test_cases > 0
                    else 0
                ),
                "execution_time_ms": results.total_execution_time_ms,
                "judge_model": results.judge_model_used,
                "llm_provider": results.llm_provider_used,
                "summary": results.summary,
                "recommendations": results.recommendations,
                "concept_breakdown": results.concept_breakdown,
                "started_at": results.started_at.isoformat(),
                "completed_at": results.completed_at.isoformat(),
            }

            # Add detailed test results if requested
            if step.config.get("include_detailed_results", False):
                formatted_results["test_case_results"] = []
                for test_result in results.test_case_results:
                    formatted_results["test_case_results"].append(
                        {
                            "name": test_result.test_case_name,
                            "passed": test_result.overall_passed,
                            "score": test_result.overall_score,
                            "llm_output": test_result.llm_output,
                            "execution_time_ms": test_result.execution_time_ms,
                            "concept_checks": [
                                {
                                    "type": check.check_type.value,
                                    "concept": check.concept,
                                    "passed": check.passed,
                                    "confidence": check.confidence,
                                    "reasoning": check.judge_reasoning,
                                }
                                for check in test_result.check_results
                            ],
                        }
                    )

            # Add quality gate checking
            min_score = step.config.get("min_score")
            if min_score is not None:
                formatted_results["quality_gate_passed"] = (
                    results.overall_score >= min_score
                )
                if not formatted_results["quality_gate_passed"]:
                    formatted_results["quality_gate_message"] = (
                        f"Evaluation score {results.overall_score:.1%} below minimum threshold {min_score:.1%}"
                    )

            return formatted_results

        except Exception as e:
            # Return error information for debugging
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "evaluation_id": step.config.get("eval_id", "unknown"),
                "timestamp": datetime.now().isoformat(),
            }

    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _handle_tts(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle text-to-speech step"""
        tts_service = self.orchestrator.get_service("tts")
        if not tts_service:
            raise RuntimeError("TTS service not available")

        text = inputs.get("text")
        if not text:
            raise ValueError("text required for TTS step")

        # Get configuration
        voice = step.config.get("voice")
        output_path = step.config.get("output_path")
        provider = step.config.get("provider")
        speed = step.config.get("speed", 1.0)
        output_format = step.config.get("output_format", "mp3")

        # Create TTS parameters
        tts_params = {}
        if voice:
            tts_params["voice"] = voice
        if output_path:
            tts_params["output_path"] = output_path
        if speed != 1.0:
            tts_params["speed"] = speed
        if output_format != "mp3":
            tts_params["output_format"] = output_format

        # Add any extra parameters from config
        extra_params = step.config.get("extra_params", {})
        tts_params.update(extra_params)

        try:
            # If provider is specified and different from current, create new service
            if provider and provider != tts_service.config.provider.value:
                # Import TTS factory to create new service
                from tts.tts_factory import create_tts_service

                tts_service = create_tts_service(provider, **tts_params)

            # Generate speech
            response = tts_service.text_to_speech(text, **tts_params)

            if not response.success:
                return {
                    "success": False,
                    "error": response.error_message,
                    "provider": response.provider,
                    "text": text,
                }

            return {
                "success": True,
                "audio_file_path": response.audio_file_path,
                "audio_url": response.audio_url,
                "audio_data": response.audio_data,
                "duration_ms": response.duration_ms,
                "provider": response.provider,
                "voice_used": response.voice_used,
                "format_used": response.format_used,
                "text": text,
                "text_length": len(text),
                "metadata": response.metadata,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider": provider or "unknown",
                "text": text,
                "error_type": type(e).__name__,
            }

    def _handle_stt(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle speech-to-text step"""
        stt_service = self.orchestrator.get_service("stt")
        if not stt_service:
            raise RuntimeError("STT service not available")

        audio_file_path = inputs.get("audio_file_path")
        if not audio_file_path:
            raise ValueError("audio_file_path required for STT step")

        # Get configuration
        language = step.config.get("language")
        model = step.config.get("model")
        provider = step.config.get("provider")
        enable_word_timestamps = step.config.get("enable_word_timestamps", False)
        enable_speaker_diarization = step.config.get(
            "enable_speaker_diarization", False
        )
        temperature = step.config.get("temperature", 0.0)
        beam_size = step.config.get("beam_size", 5)

        # Create STT parameters
        stt_params = {}
        if language:
            stt_params["language"] = language
        if model:
            stt_params["model"] = model
        if enable_word_timestamps:
            stt_params["enable_word_timestamps"] = enable_word_timestamps
        if enable_speaker_diarization:
            stt_params["enable_speaker_diarization"] = enable_speaker_diarization
        if temperature != 0.0:
            stt_params["temperature"] = temperature
        if beam_size != 5:
            stt_params["beam_size"] = beam_size

        # Add any extra parameters from config
        extra_params = step.config.get("extra_params", {})
        stt_params.update(extra_params)

        try:
            # If provider is specified and different from current, create new service
            if provider and provider != stt_service.config.provider.value:
                # Import STT factory to create new service
                from stt.stt_factory import create_stt_service

                stt_service = create_stt_service(provider, **stt_params)

            # Transcribe speech
            response = stt_service.speech_to_text(audio_file_path, **stt_params)

            if not response.success:
                return {
                    "success": False,
                    "error": response.error_message,
                    "provider": response.provider,
                    "audio_file_path": audio_file_path,
                }

            return {
                "success": True,
                "transcript": response.transcript,
                "language_detected": response.language_detected,
                "confidence": response.confidence,
                "word_timestamps": [wt.model_dump() for wt in response.word_timestamps],
                "speaker_segments": [
                    ss.model_dump() for ss in response.speaker_segments
                ],
                "duration_seconds": response.duration_seconds,
                "provider": response.provider,
                "model_used": response.model_used,
                "audio_file_path": audio_file_path,
                "metadata": response.metadata,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "provider": provider or "unknown",
                "audio_file_path": audio_file_path,
                "error_type": type(e).__name__,
            }

    def _handle_start_conversation(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle starting a new conversation thread"""
        conversation_id = step.config.get("conversation_id") or inputs.get(
            "conversation_id"
        )
        if not conversation_id:
            # Generate a default conversation ID
            conversation_id = f"conversation_{len(context.conversations) + 1}"

        conversation_name = step.config.get("name") or inputs.get("name")

        # Create the conversation
        conversation = context.create_conversation(conversation_id, conversation_name)

        # Set as active conversation
        context.set_active_conversation(conversation_id)

        # Add initial system message if provided
        system_message = step.config.get("system_message") or inputs.get(
            "system_message"
        )
        if system_message:
            conversation.add_message("system", system_message)

        return {
            "conversation_id": conversation_id,
            "conversation_name": conversation_name,
            "system_message": system_message,
            "message_count": len(conversation.messages),
        }

    def _handle_add_to_conversation(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle adding a message to a conversation thread"""
        conversation_id = step.config.get("conversation_id") or inputs.get(
            "conversation_id"
        )
        role = step.config.get("role") or inputs.get("role")
        content = inputs.get("content") or inputs.get("message")

        if not content:
            raise ValueError("content or message required for add_to_conversation")

        if not role:
            raise ValueError("role required for add_to_conversation")

        # Get the conversation (use active if not specified)
        if conversation_id:
            if conversation_id not in context.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            conversation = context.conversations[conversation_id]
        else:
            conversation = context.get_active_conversation()
            if not conversation:
                raise ValueError(
                    "No active conversation found and no conversation_id specified"
                )
            conversation_id = conversation.id

        # Add the message
        metadata = inputs.get("metadata", {})
        message = conversation.add_message(role, content, metadata)

        return {
            "conversation_id": conversation_id,
            "message_role": role,
            "message_content": content,
            "message_timestamp": message.timestamp.isoformat(),
            "total_messages": len(conversation.messages),
        }

    def _handle_continue_conversation(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle continuing a conversation with an LLM response"""
        conversation_id = step.config.get("conversation_id") or inputs.get(
            "conversation_id"
        )
        user_message = inputs.get("message") or inputs.get("user_message")

        if not user_message:
            raise ValueError(
                "message or user_message required for continue_conversation"
            )

        # Get the conversation (use active if not specified)
        if conversation_id:
            if conversation_id not in context.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            conversation = context.conversations[conversation_id]
        else:
            conversation = context.get_active_conversation()
            if not conversation:
                raise ValueError(
                    "No active conversation found and no conversation_id specified"
                )
            conversation_id = conversation.id

        # Add user message to conversation
        conversation.add_message("user", user_message)

        # Get LLM configuration
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")

        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")

        # Get credential manager from orchestrator
        credential_manager = getattr(self.orchestrator, "_credential_manager", None)

        # Create LLM client with credential manager
        if provider == "gemini":
            client = llm_factory.create_text_client(
                LLMProvider.GEMINI, model, credential_manager=credential_manager
            )
        elif provider == "ollama":
            client = llm_factory.create_text_client(
                LLMProvider.OLLAMA, model, credential_manager=credential_manager
            )
        elif provider == "anthropic":
            client = llm_factory.create_text_client(
                LLMProvider.ANTHROPIC, model, credential_manager=credential_manager
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Get conversation history for context
        include_system = step.config.get("include_system_messages", True)
        history_limit = step.config.get("history_limit", 0)  # 0 = all messages

        if history_limit > 0:
            recent_messages = conversation.get_last_messages(
                history_limit, include_system
            )
            llm_messages = [
                {"role": msg.role, "content": msg.content} for msg in recent_messages
            ]
        else:
            llm_messages = conversation.get_messages_for_llm(include_system)

        # Generate response using conversation history
        # For now, we'll use a simple approach - create a prompt with history
        if len(llm_messages) > 1:
            # Build conversation context
            context_parts = []
            for msg in llm_messages[
                :-1
            ]:  # All but the last message (current user message)
                context_parts.append(f"{msg['role']}: {msg['content']}")

            conversation_context = "\n".join(context_parts)
            full_prompt = f"Previous conversation:\n{conversation_context}\n\nCurrent message: {user_message}"
        else:
            full_prompt = user_message

        # Generate response
        response = client.chat(full_prompt)

        # Add assistant response to conversation
        conversation.add_message("assistant", response)

        return {
            "conversation_id": conversation_id,
            "user_message": user_message,
            "assistant_response": response,
            "total_messages": len(conversation.messages),
            "provider": provider,
            "model": model,
            "history_used": len(llm_messages) > 1,
        }

    def _handle_human_approval(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle human approval step - presents context and collects human input"""
        import signal
        import time

        # Configuration
        approval_type = step.config.get("approval_type", "approve_reject")
        prompt = step.config.get("prompt", "Please review and provide your decision:")
        options = step.config.get("options", ["approve", "reject"])
        timeout_seconds = step.config.get("timeout_seconds", 300)  # 5 minute default
        default_action = step.config.get("default_action", "reject")
        show_context = step.config.get("show_context", True)
        context_fields = step.config.get("context_fields", [])

        # Display context to human
        if show_context:
            self._display_approval_context(inputs, context, context_fields)

        # Present the prompt and options
        print(f"\n{'='*60}")
        print("HUMAN APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"{prompt}\n")

        if approval_type == "approve_reject":
            print("Options: [a]pprove, [r]eject")
            valid_inputs = {
                "a": "approve",
                "approve": "approve",
                "r": "reject",
                "reject": "reject",
            }

        elif approval_type == "multiple_choice":
            print(
                f"Options: {', '.join(f'[{i+1}] {opt}' for i, opt in enumerate(options))}"
            )
            valid_inputs = {}
            for i, option in enumerate(options):
                valid_inputs[str(i + 1)] = option
                valid_inputs[option.lower()] = option

        elif approval_type == "custom_input":
            print("Please provide your input:")
            valid_inputs = None  # Any input is valid

        else:
            raise ValueError(f"Unknown approval_type: {approval_type}")

        # Setup timeout handling
        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError("Human approval timed out")

        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            # Get human input
            while True:
                try:
                    if timeout_seconds > 0:
                        print(
                            f"\n(Timeout in {timeout_seconds} seconds, default: {default_action})"
                        )

                    user_input = input("\nYour response: ").strip().lower()

                    if approval_type == "custom_input":
                        # For custom input, any non-empty response is valid
                        if user_input:
                            decision = user_input
                            feedback = user_input
                            break
                        else:
                            print("Please provide a non-empty response.")
                            continue

                    elif user_input in valid_inputs:
                        decision = valid_inputs[user_input]
                        feedback = f"Selected: {decision}"
                        break

                    else:
                        print(
                            f"Invalid input. Please choose from: {list(valid_inputs.keys())}"
                        )
                        continue

                except EOFError:
                    # Handle Ctrl+D
                    print(
                        f"\nInput interrupted. Using default action: {default_action}"
                    )
                    decision = default_action
                    feedback = "Default action used due to input interruption"
                    break

        except TimeoutError:
            print(f"\nTimeout reached. Using default action: {default_action}")
            decision = default_action
            feedback = f"Default action used due to timeout ({timeout_seconds}s)"

        finally:
            # Restore original signal handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # Return results that can be used by subsequent steps
        result = {
            "decision": decision,
            "feedback": feedback,
            "approval_type": approval_type,
            "prompt": prompt,
            "timestamp": time.time(),
            "timeout_seconds": timeout_seconds,
            "context_shown": show_context,
        }

        # For custom input, also include the raw input
        if approval_type == "custom_input":
            result["user_input"] = user_input

        print(f"\nDecision recorded: {decision}")
        print(f"{'='*60}\n")

        return result

    def _display_approval_context(
        self,
        inputs: Dict[str, Any],
        context: ExecutionContext,
        context_fields: List[str],
    ) -> None:
        """Display relevant context information to help human make decision"""
        print(f"\n{'='*60}")
        print("CONTEXT INFORMATION")
        print(f"{'='*60}")

        # Show step inputs
        if inputs:
            print("\nCurrent Step Inputs:")
            for key, value in inputs.items():
                if not context_fields or key in context_fields:
                    formatted_value = self._format_context_value(value)
                    print(f"  {key}: {formatted_value}")

        # Show relevant step outputs from workflow
        if context.step_outputs:
            print("\nPrevious Step Results:")
            for key, value in context.step_outputs.items():
                # If context_fields specified, only show those
                if context_fields:
                    # Check if this output matches any requested field
                    field_name = key.split(".")[-1] if "." in key else key
                    if field_name not in context_fields:
                        continue

                formatted_value = self._format_context_value(value)
                print(f"  {key}: {formatted_value}")

        # Show global variables if relevant
        if context.global_variables:
            print("\nGlobal Variables:")
            for key, value in context.global_variables.items():
                if not context_fields or key in context_fields:
                    formatted_value = self._format_context_value(value)
                    print(f"  {key}: {formatted_value}")

    def _format_context_value(self, value: Any, max_length: int = 200) -> str:
        """Format a context value for display, truncating if too long"""
        if isinstance(value, (str, int, float, bool)):
            str_value = str(value)
        elif isinstance(value, dict):
            # For dicts, show structure but limit content
            if len(str(value)) <= max_length:
                str_value = str(value)
            else:
                # Show just the keys for large dicts
                str_value = f"{{dict with keys: {list(value.keys())}}}"
        elif isinstance(value, list):
            if len(str(value)) <= max_length:
                str_value = str(value)
            else:
                str_value = f"[list with {len(value)} items]"
        else:
            str_value = str(value)

        # Truncate if still too long
        if len(str_value) > max_length:
            str_value = str_value[: max_length - 3] + "..."

        return str_value

    def _handle_document_processing(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle document processing step - now supports visual content (PDFs and images)"""
        file_path = inputs.get("file_path")
        base64_image = inputs.get("base64_image")

        if not file_path and not base64_image:
            raise ValueError(
                "file_path or base64_image required for document processing"
            )

        # Handle base64 image input
        if base64_image:
            return self._handle_base64_image_processing(base64_image, step.config)

        # Validate file path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file extension
        file_ext = pathlib.Path(file_path).suffix.lower()

        # Handle markdown files
        if file_ext == ".md":
            return self._handle_markdown_processing(file_path, step.config)

        # Handle visual content (PDFs and images) using the new visual to text service
        elif file_ext in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return self._handle_visual_processing(file_path, step.config)

        else:
            supported_formats = [
                ".pdf",
                ".md",
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".webp",
            ]
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}"
            )

    def _handle_markdown_processing(
        self, file_path: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle markdown file processing"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            raise RuntimeError(f"Error reading markdown file: {str(e)}")

        # Create semantic chunks if requested
        semantic_chunks = None
        if config.get("semantic_analysis", False):
            # Split markdown content into logical chunks (by headers, paragraphs, etc.)
            chunks = self._create_markdown_chunks(content)
            semantic_chunks = [{"content": chunk, "metadata": {}} for chunk in chunks]

        return {
            "text": content,
            "original_text": content,
            "vision_text": None,
            "semantic_chunks": semantic_chunks,
            "page_count": 1,  # Markdown files are considered single-page documents
            "processing_method": "markdown",
        }

    def _handle_visual_processing(
        self, file_path: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle visual file processing (PDFs and images) using the new visual to text service"""
        # Create visual to text service
        visual_service = VisualToTextService()

        # Configure extraction options
        options = VisualExtractOptions()

        # Map configuration options
        if "enhance_with_llm" in config:
            # For images, we always use vision processing
            pass
        if "semantic_analysis" in config:
            options.create_semantic_chunks = config["semantic_analysis"]
        if "extract_tables" in config:
            options.extract_tables = config["extract_tables"]
        if "include_bounding_boxes" in config:
            options.include_bounding_boxes = config["include_bounding_boxes"]
        if "vision_prompt" in config:
            options.vision_prompt = config["vision_prompt"]
        if "preserve_layout" in config:
            options.preserve_layout = config["preserve_layout"]
        if "page_range" in config:
            options.page_range = tuple(config["page_range"])

        # Extract text from visual content
        result = visual_service.extract_text_from_file(file_path, options)

        return {
            "text": result.text,
            "original_text": result.text,
            "vision_text": result.text if result.vision_processing_used else None,
            "semantic_chunks": result.semantic_chunks,
            "page_count": result.page_count,
            "processing_method": result.processing_method,
            "source_type": result.source_type,
            "converted_images": result.converted_images,
            "bounding_boxes": result.bounding_boxes,
        }

    def _handle_base64_image_processing(
        self, base64_image: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle base64 image processing using the visual to text service"""
        # Create visual to text service
        visual_service = VisualToTextService()

        # Configure extraction options
        options = VisualExtractOptions()

        # Map configuration options
        if "semantic_analysis" in config:
            options.create_semantic_chunks = config["semantic_analysis"]
        if "extract_tables" in config:
            options.extract_tables = config["extract_tables"]
        if "include_bounding_boxes" in config:
            options.include_bounding_boxes = config["include_bounding_boxes"]
        if "vision_prompt" in config:
            options.vision_prompt = config["vision_prompt"]

        # Extract text from base64 image
        result = visual_service.extract_text_from_base64_image(base64_image, options)

        return {
            "text": result.text,
            "original_text": result.text,
            "vision_text": result.text,
            "semantic_chunks": result.semantic_chunks,
            "page_count": 1,  # Base64 images are single-page
            "processing_method": result.processing_method,
            "source_type": result.source_type,
            "bounding_boxes": result.bounding_boxes,
        }

    def _handle_pdf_processing(
        self, file_path: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle PDF file processing (legacy method - kept for backward compatibility)"""
        pdf_service = self.orchestrator.get_service("pdf_processor")
        if not pdf_service:
            raise RuntimeError("PDF processor service not available")

        # Configure extraction options
        options = PDFExtractOptions()
        if "enhance_with_llm" in config:
            # Map enhance_with_llm to available vision processing options
            options.use_vision_fallback = config["enhance_with_llm"]
        if "semantic_analysis" in config:
            # Map semantic_analysis to semantic chunking
            options.create_semantic_chunks = config["semantic_analysis"]

        # Extract text from PDF
        result = pdf_service.extract_text_from_file(file_path, options)

        return {
            "text": result.vision_text if result.vision_text else result.text,
            "original_text": result.text,
            "vision_text": result.vision_text,
            "semantic_chunks": result.semantic_chunks,
            "page_count": result.page_count,
            "processing_method": result.processing_method,
        }

    def _create_markdown_chunks(self, content: str) -> List[str]:
        """Create logical chunks from markdown content"""
        chunks = []
        lines = content.split("\n")
        current_chunk = []

        for line in lines:
            # Start new chunk on headers (# ## ###)
            if line.strip().startswith("#") and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        return chunks

    def _handle_memory_store(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle memory storage step"""
        memory_service = self.orchestrator.get_service("memory")
        if not memory_service:
            raise RuntimeError("Memory service not available")

        content = inputs.get("content")
        if not content:
            raise ValueError("content required for memory storage")

        metadata = inputs.get("metadata", {})
        metadata.update(step.config.get("metadata", {}))

        # Handle case where content is a list of chunks (from chunking step)
        if isinstance(content, list):
            # Store each chunk as a separate memory
            memory_ids = []
            for i, chunk in enumerate(content):
                if isinstance(chunk, str) and chunk.strip():
                    # Add chunk index to metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(content)

                    memory_id = memory_service.store_memory(chunk, chunk_metadata)
                    memory_ids.append(memory_id)

            return {
                "memory_ids": memory_ids,
                "stored_count": len(memory_ids),
                "total_chunks": len(content),
                "metadata": metadata,
            }
        else:
            # Single content item - original behavior
            memory_id = memory_service.store_memory(content, metadata)

            return {
                "memory_id": memory_id,
                "stored_content": content,
                "metadata": metadata,
            }

    def _handle_memory_retrieve(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle memory retrieval step"""
        memory_service = self.orchestrator.get_service("memory")
        if not memory_service:
            raise RuntimeError("Memory service not available")

        query = inputs.get("query")
        if not query:
            raise ValueError("query required for memory retrieval")

        # Configure retrieval
        limit = step.config.get("limit", 5)
        # threshold = step.config.get("threshold", 0.7)  # Currently unused, handled internally

        # Retrieve memories (note: threshold filtering is handled internally by the memory service)
        memories = memory_service.retrieve_memories(query, limit=limit)

        return {
            "memories": [
                {
                    "content": m.content,
                    "metadata": m.metadata,
                    "similarity": getattr(m, "similarity", None),
                }
                for m in memories
            ],
            "count": len(memories),
        }

    def _handle_llm_chat(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle LLM chat step - uses generation service for one-shot, conversation service for multi-turn"""
        # Check if using managed prompt
        if step.prompt_ref:
            return self._handle_llm_chat_with_prompt(step, inputs, context)

        # Process JSON props first
        enhanced_inputs = self._process_json_props(step, inputs, context)

        message = enhanced_inputs.get("message")
        if not message:
            raise ValueError("message required for LLM chat")

        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        temperature = step.config.get("temperature", 0.7)
        max_tokens = step.config.get("max_tokens", 1000)

        # Check if streaming is enabled
        use_streaming = step.config.get("stream", False)

        # Check if we should use conversation context
        use_conversation = step.config.get("use_conversation", False)
        conversation_id = step.config.get("conversation_id")

        # Configure thinking tokens handling
        thinking_tokens_mode = step.config.get("thinking_tokens_mode")
        preserve_thinking = step.config.get("preserve_thinking", False)

        # Get global thinking tokens config if not specified at step level
        global_config = getattr(context, "global_config", None)
        if thinking_tokens_mode is None and global_config:
            thinking_tokens_mode = global_config.thinking_tokens_mode

        # Convert string to enum if needed
        if isinstance(thinking_tokens_mode, str):
            thinking_tokens_mode = ThinkingTokensMode(thinking_tokens_mode)
        elif thinking_tokens_mode is None:
            thinking_tokens_mode = ThinkingTokensMode.AUTO

        if use_conversation:
            # Use conversation service for multi-turn conversations
            conversation_service = self.orchestrator.get_service("conversation")
            if not conversation_service:
                raise RuntimeError("Conversation service not available")

            # Check if this is a new conversation or continuing existing one
            if conversation_id:
                # Try to load existing conversation state if available
                # For now, create a new conversation service instance
                # In a full implementation, you might want to persist/restore conversation state
                from llm.llm_types import LLMProvider

                provider_enum = LLMProvider(provider)
                conversation_service = type(conversation_service)(
                    provider_enum, model, temperature, max_tokens, conversation_id
                )

            # Add system message if provided
            system_message = step.config.get("system_message")
            if system_message:
                conversation_service.add_system_message(system_message)

            # Send message and get response (streaming or regular)
            if use_streaming:
                # Enhanced streaming with intelligent buffering
                stream_generator = conversation_service.send_message_stream(message)

                # Check for enhanced streaming buffer configuration
                if step.stream_buffer:
                    buffer = StreamBuffer(step.stream_buffer)
                    response_chunks = []
                    buffered_chunks = []

                    for chunk in stream_generator:
                        response_chunks.append(chunk)

                        # Process through buffer for intelligent forwarding
                        ready_chunks = buffer.process_chunk(chunk)
                        buffered_chunks.extend(ready_chunks)

                    # Flush any remaining content
                    final_chunks = buffer.flush()
                    buffered_chunks.extend(final_chunks)

                    response = "".join(response_chunks)
                    streaming_info = {
                        "chunks": response_chunks,
                        "buffered_chunks": buffered_chunks,
                        "streamed": True,
                        "buffer_strategy": step.stream_buffer.forward_on,
                        "buffer_stats": buffer.get_stats(),
                    }
                else:
                    # Legacy streaming - collect all chunks
                    response_chunks = []
                    for chunk in stream_generator:
                        response_chunks.append(chunk)
                    response = "".join(response_chunks)
                    streaming_info = {"chunks": response_chunks, "streamed": True}
            else:
                response = conversation_service.send_message(message)
                streaming_info = {"streamed": False}

            return {
                "response": response,
                "message": message,
                "provider": provider,
                "model": model,
                "conversation_id": conversation_service.conversation.id,
                "total_messages": conversation_service.get_conversation_length(),
                "first_prompt": conversation_service.get_first_prompt(),
                "last_response": conversation_service.get_last_response(),
                "service_type": "conversation",
                **streaming_info,
            }

        else:
            # Use generation service for one-shot interactions
            generation_service = self.orchestrator.get_service("generation")
            if not generation_service:
                raise RuntimeError("Generation service not available")

            # Create generation service with specific config for this call
            from llm.llm_types import LLMProvider

            provider_enum = LLMProvider(provider)
            gen_service = type(generation_service)(
                provider_enum,
                model,
                temperature,
                max_tokens,
                thinking_tokens_mode=thinking_tokens_mode,
            )

            # Check for system prompt and handle streaming
            system_prompt = step.config.get("system_message")

            if use_streaming:
                # Enhanced streaming with intelligent buffering
                if system_prompt:
                    stream_generator = gen_service.generate_with_system_prompt_stream(
                        message, system_prompt
                    )
                else:
                    stream_generator = gen_service.generate_stream(message)

                # Check for enhanced streaming buffer configuration
                if step.stream_buffer:
                    buffer = StreamBuffer(step.stream_buffer)
                    response_chunks = []
                    buffered_chunks = []

                    for chunk in stream_generator:
                        response_chunks.append(chunk)

                        # Process through buffer for intelligent forwarding
                        ready_chunks = buffer.process_chunk(chunk)
                        buffered_chunks.extend(ready_chunks)

                    # Flush any remaining content
                    final_chunks = buffer.flush()
                    buffered_chunks.extend(final_chunks)

                    response = "".join(response_chunks)
                    streaming_info = {
                        "chunks": response_chunks,
                        "buffered_chunks": buffered_chunks,
                        "streamed": True,
                        "buffer_strategy": step.stream_buffer.forward_on,
                        "buffer_stats": buffer.get_stats(),
                    }
                else:
                    # Legacy streaming - collect all chunks
                    response_chunks = []
                    for chunk in stream_generator:
                        response_chunks.append(chunk)
                    response = "".join(response_chunks)
                    streaming_info = {"chunks": response_chunks, "streamed": True}
            else:
                # Use regular generation methods
                if system_prompt:
                    response = gen_service.generate_with_system_prompt(
                        message,
                        system_prompt,
                        thinking_tokens_mode=thinking_tokens_mode,
                        preserve_for_structured=preserve_thinking,
                    )
                else:
                    response = gen_service.generate(
                        message,
                        thinking_tokens_mode=thinking_tokens_mode,
                        preserve_for_structured=preserve_thinking,
                    )
                streaming_info = {"streamed": False}

            # Check if we should return thinking tokens information
            result = {
                "response": response,
                "message": message,
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt,
                "service_type": "generation",
                "thinking_tokens_mode": (
                    thinking_tokens_mode.value if thinking_tokens_mode else None
                ),
                **streaming_info,
            }

            # If extract mode is requested, get detailed thinking tokens info
            if thinking_tokens_mode == ThinkingTokensMode.EXTRACT and not use_streaming:
                if system_prompt:
                    thinking_info = gen_service.generate_with_thinking_tokens(
                        message, system_prompt
                    )
                else:
                    thinking_info = gen_service.generate_with_thinking_tokens(message)

                result.update(
                    {
                        "thinking_content": thinking_info.get("thinking_content"),
                        "has_thinking_tokens": thinking_info.get(
                            "has_thinking_tokens", False
                        ),
                        "original_response": thinking_info.get("original_response"),
                    }
                )

            return result

    def _handle_llm_chat_with_prompt(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle LLM chat step using managed prompt"""
        import time

        prompt_service = self.orchestrator.get_service("prompt")
        if not prompt_service:
            raise RuntimeError("Prompt service not available")

        # Get the prompt
        prompt = prompt_service.get_prompt(
            step.prompt_ref.prompt_id, step.prompt_ref.version
        )
        if not prompt:
            raise ValueError(f"Prompt not found: {step.prompt_ref.prompt_id}")

        # Build context for template rendering
        template_context = {}
        template_context.update(step.prompt_ref.context_variables)  # Static variables
        template_context.update(inputs)  # Step inputs
        template_context.update(context.global_variables)  # Global variables

        # Render the prompt
        try:
            rendered_messages = prompt.render(template_context)
        except Exception as e:
            raise ValueError(
                f"Failed to render prompt {step.prompt_ref.prompt_id}: {e}"
            )

        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")
        temperature = step.config.get("temperature", 0.7)
        max_tokens = step.config.get("max_tokens", 1000)

        # Check if we should use conversation context
        use_conversation = step.config.get("use_conversation", False)
        conversation_id = step.config.get("conversation_id")

        start_time = time.time()

        if use_conversation:
            # Use conversation service for multi-turn conversations
            conversation_service = self.orchestrator.get_service("conversation")
            if not conversation_service:
                raise RuntimeError("Conversation service not available")

            # For conversation service, we need to handle the rendered messages differently
            # Extract system message if present
            system_message = None
            user_message = None

            for msg in rendered_messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_message = msg["content"]

            if not user_message:
                raise ValueError("No user message found in rendered prompt")

            # Create conversation service instance for this call
            from llm.llm_types import LLMProvider

            provider_enum = LLMProvider(provider)
            conversation_service = type(conversation_service)(
                provider_enum, model, temperature, max_tokens, conversation_id
            )

            # Add system message if provided
            if system_message:
                conversation_service.add_system_message(system_message)

            # Send message and get response
            response = conversation_service.send_message(user_message)

            execution_time_ms = (time.time() - start_time) * 1000

            # Log execution
            prompt_service.log_execution(
                prompt_id=step.prompt_ref.prompt_id,
                prompt_version=prompt.version.version,
                execution_context=template_context,
                rendered_messages=rendered_messages,
                llm_provider=provider,
                llm_model=model,
                llm_response=response,
                execution_time_ms=execution_time_ms,
                success=True,
                metadata={"step_id": step.id, "use_conversation": True},
            )

            return {
                "response": response,
                "prompt_id": step.prompt_ref.prompt_id,
                "prompt_version": prompt.version.version,
                "rendered_messages": rendered_messages,
                "provider": provider,
                "model": model,
                "conversation_id": conversation_service.conversation.id,
                "total_messages": conversation_service.get_conversation_length(),
                "service_type": "conversation",
                "execution_time_ms": execution_time_ms,
            }

        else:
            # Use generation service for one-shot interactions
            generation_service = self.orchestrator.get_service("generation")
            if not generation_service:
                raise RuntimeError("Generation service not available")

            # Create generation service with specific config for this call
            from llm.llm_types import LLMProvider

            provider_enum = LLMProvider(provider)
            gen_service = type(generation_service)(
                provider_enum, model, temperature, max_tokens
            )

            # For generation service, combine all messages into a single prompt
            # Handle system and user messages appropriately
            system_prompt = None
            user_prompt = None

            for msg in rendered_messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_prompt = msg["content"]

            if not user_prompt:
                raise ValueError("No user message found in rendered prompt")

            # Generate response
            if system_prompt:
                response = gen_service.generate_with_system_prompt(
                    user_prompt, system_prompt
                )
            else:
                response = gen_service.generate(user_prompt)

            execution_time_ms = (time.time() - start_time) * 1000

            # Log execution
            prompt_service.log_execution(
                prompt_id=step.prompt_ref.prompt_id,
                prompt_version=prompt.version.version,
                execution_context=template_context,
                rendered_messages=rendered_messages,
                llm_provider=provider,
                llm_model=model,
                llm_response=response,
                execution_time_ms=execution_time_ms,
                success=True,
                metadata={"step_id": step.id, "use_conversation": False},
            )

            return {
                "response": response,
                "prompt_id": step.prompt_ref.prompt_id,
                "prompt_version": prompt.version.version,
                "rendered_messages": rendered_messages,
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt,
                "service_type": "generation",
                "execution_time_ms": execution_time_ms,
            }

    def _handle_llm_structured(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle structured LLM response step"""
        # Check if using managed prompt
        if step.prompt_ref:
            return self._handle_llm_structured_with_prompt(step, inputs, context)

        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")

        # Process JSON props first
        enhanced_inputs = self._process_json_props(step, inputs, context)

        message = enhanced_inputs.get("message")
        if not message:
            raise ValueError("message required for LLM structured response")

        # Get schema configuration
        schema_config = step.config.get("response_schema")
        if not schema_config:
            raise ValueError("response_schema required for structured LLM step")

        # Process template variables in schema config
        if isinstance(schema_config, str):
            processed_schema = self._process_template_string(
                schema_config, enhanced_inputs, context
            )
            # If the result looks like JSON, parse it as a dictionary
            if processed_schema.strip().startswith("{"):
                import json

                try:
                    schema_config = json.loads(processed_schema)
                except json.JSONDecodeError:
                    schema_config = processed_schema
            else:
                schema_config = processed_schema

        # Parse schema - can be either a class reference or a direct Pydantic model
        schema_class = self._parse_schema_config(schema_config)

        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")

        # Get credential manager from orchestrator
        credential_manager = getattr(self.orchestrator, "_credential_manager", None)

        # Create structured client
        structured_client = llm_factory.create_structured_client(
            provider=provider,
            schema=schema_class,
            model=model,
            temperature=step.config.get("temperature", 0.7),
            max_tokens=step.config.get("max_tokens", 1000),
            credential_manager=credential_manager,
        )

        # Generate structured response
        try:
            structured_response = structured_client.chat(message)

            # Convert to dict for easier access in workflows
            response_dict = structured_response.model_dump()

            return {
                "structured_response": structured_response,
                "response_dict": response_dict,
                "message": message,
                "provider": provider,
                "model": model,
                "schema": schema_class.__name__,
                **response_dict,  # Include all fields at top level for easy access
            }

        except Exception as e:
            # Handle structured response failures gracefully
            return {
                "error": str(e),
                "message": message,
                "provider": provider,
                "model": model,
                "schema": schema_class.__name__,
                "structured_response": None,
                "response_dict": {},
            }

    def _handle_llm_structured_with_prompt(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle structured LLM response step using managed prompt"""
        import time

        prompt_service = self.orchestrator.get_service("prompt")
        if not prompt_service:
            raise RuntimeError("Prompt service not available")

        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")

        # Get the prompt
        prompt = prompt_service.get_prompt(
            step.prompt_ref.prompt_id, step.prompt_ref.version
        )
        if not prompt:
            raise ValueError(f"Prompt not found: {step.prompt_ref.prompt_id}")

        # Build context for template rendering
        template_context = {}
        template_context.update(step.prompt_ref.context_variables)  # Static variables
        template_context.update(inputs)  # Step inputs
        template_context.update(context.global_variables)  # Global variables

        # Render the prompt
        try:
            rendered_messages = prompt.render(template_context)
        except Exception as e:
            raise ValueError(
                f"Failed to render prompt {step.prompt_ref.prompt_id}: {e}"
            )

        # Get schema configuration
        schema_config = step.config.get("response_schema")
        if not schema_config:
            raise ValueError("response_schema required for structured LLM step")

        # Parse schema - can be either a class reference or a direct Pydantic model
        schema_class = self._parse_schema_config(schema_config)

        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")

        # Get credential manager from orchestrator
        credential_manager = getattr(self.orchestrator, "_credential_manager", None)

        # Create structured client
        structured_client = llm_factory.create_structured_client(
            provider=provider,
            schema=schema_class,
            model=model,
            temperature=step.config.get("temperature", 0.7),
            max_tokens=step.config.get("max_tokens", 1000),
            credential_manager=credential_manager,
        )

        # For structured responses, we need to convert multi-message prompt to single message
        # Combine system and user messages appropriately
        combined_message = ""

        for msg in rendered_messages:
            if msg["role"] == "system":
                combined_message += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                combined_message += msg["content"]

        if not combined_message:
            raise ValueError("No message content found in rendered prompt")

        start_time = time.time()

        # Generate structured response
        try:
            structured_response = structured_client.chat(combined_message.strip())

            execution_time_ms = (time.time() - start_time) * 1000

            # Convert to dict for easier access in workflows
            response_dict = structured_response.model_dump()

            # Log execution
            prompt_service.log_execution(
                prompt_id=step.prompt_ref.prompt_id,
                prompt_version=prompt.version.version,
                execution_context=template_context,
                rendered_messages=rendered_messages,
                llm_provider=provider,
                llm_model=model,
                llm_response=str(structured_response),
                execution_time_ms=execution_time_ms,
                success=True,
                metadata={"step_id": step.id, "schema": schema_class.__name__},
            )

            return {
                "structured_response": structured_response,
                "response_dict": response_dict,
                "prompt_id": step.prompt_ref.prompt_id,
                "prompt_version": prompt.version.version,
                "rendered_messages": rendered_messages,
                "provider": provider,
                "model": model,
                "schema": schema_class.__name__,
                "execution_time_ms": execution_time_ms,
                **response_dict,  # Include all fields at top level for easy access
            }

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            # Log failed execution
            prompt_service.log_execution(
                prompt_id=step.prompt_ref.prompt_id,
                prompt_version=prompt.version.version,
                execution_context=template_context,
                rendered_messages=rendered_messages,
                llm_provider=provider,
                llm_model=model,
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e),
                metadata={"step_id": step.id, "schema": schema_class.__name__},
            )

            # Handle structured response failures gracefully
            return {
                "error": str(e),
                "prompt_id": step.prompt_ref.prompt_id,
                "prompt_version": prompt.version.version,
                "rendered_messages": rendered_messages,
                "provider": provider,
                "model": model,
                "schema": schema_class.__name__,
                "structured_response": None,
                "response_dict": {},
                "execution_time_ms": execution_time_ms,
            }

    def _parse_schema_config(self, schema_config: Any) -> Type[BaseModel]:
        """Parse schema configuration into a Pydantic model class"""
        if isinstance(schema_config, str):
            # String reference - try to import as module.ClassName
            if "." in schema_config:
                module_path, class_name = schema_config.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    schema_class = getattr(module, class_name)
                    if not (
                        isinstance(schema_class, type)
                        and issubclass(schema_class, BaseModel)
                    ):
                        raise ValueError(
                            f"Schema class {schema_config} is not a Pydantic BaseModel"
                        )
                    return schema_class
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Could not import schema class {schema_config}: {e}"
                    )
            else:
                # Try to create a dynamic schema from predefined common schemas
                return self._create_dynamic_schema(schema_config)

        elif isinstance(schema_config, dict):
            # Dictionary schema definition - create dynamic model
            return self._create_schema_from_dict(schema_config)

        elif isinstance(schema_config, type) and issubclass(schema_config, BaseModel):
            # Direct Pydantic model class
            return schema_config

        else:
            raise ValueError(f"Invalid schema configuration: {schema_config}")

    def _create_dynamic_schema(self, schema_name: str) -> Type[BaseModel]:
        """Create a dynamic schema for common use cases"""
        common_schemas = {
            "simple_response": self._create_simple_response_schema,
            "classification": self._create_classification_schema,
            "extraction": self._create_extraction_schema,
            "decision": self._create_decision_schema,
            "summary": self._create_summary_schema,
        }

        if schema_name in common_schemas:
            return common_schemas[schema_name]()
        else:
            raise ValueError(f"Unknown predefined schema: {schema_name}")

    def _create_schema_from_dict(self, schema_dict: Dict[str, Any]) -> Type[BaseModel]:
        """Create a Pydantic model from a dictionary definition"""
        from pydantic import create_model

        # Extract fields from dictionary
        fields = {}
        for field_name, field_config in schema_dict.get("fields", {}).items():
            if isinstance(field_config, dict):
                field_type = field_config.get("type", str)
                field_default = field_config.get("default", ...)
                # field_description = field_config.get("description", "")  # Currently unused

                # Convert string type names to actual types
                if isinstance(field_type, str):
                    type_mapping = {
                        "str": str,
                        "string": str,
                        "int": int,
                        "integer": int,
                        "float": float,
                        "bool": bool,
                        "boolean": bool,
                        "list": list,
                        "dict": dict,
                    }
                    field_type = type_mapping.get(field_type, str)

                if field_default == ...:
                    fields[field_name] = (field_type, ...)  # Required field
                else:
                    fields[field_name] = (field_type, field_default)
            else:
                # Simple field definition
                fields[field_name] = (str, ...)

        # Create dynamic model
        model_name = schema_dict.get("name", "DynamicSchema")
        return create_model(model_name, **fields)

    def _create_simple_response_schema(self) -> Type[BaseModel]:
        """Create a simple response schema"""

        class SimpleResponse(BaseModel):
            response: str
            confidence: float = 1.0

        return SimpleResponse

    def _create_classification_schema(self) -> Type[BaseModel]:
        """Create a classification schema"""

        class Classification(BaseModel):
            category: str
            confidence: float
            reasoning: str = ""

        return Classification

    def _create_extraction_schema(self) -> Type[BaseModel]:
        """Create an extraction schema"""

        class Extraction(BaseModel):
            extracted_data: Dict[str, Any]
            confidence: float
            source_text: str = ""

        return Extraction

    def _create_decision_schema(self) -> Type[BaseModel]:
        """Create a decision schema"""

        class Decision(BaseModel):
            decision: str
            reasoning: str
            confidence: float
            alternatives: List[str] = []

        return Decision

    def _create_summary_schema(self) -> Type[BaseModel]:
        """Create a summary schema"""

        class Summary(BaseModel):
            summary: str
            key_points: List[str]
            word_count: int = 0

        return Summary

    def _handle_llm_vision(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle LLM vision step"""
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")

        image_path = inputs.get("image_path")
        prompt = inputs.get("prompt", "Describe this image")

        if not image_path:
            raise ValueError("image_path required for LLM vision")

        # Configure vision model
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")

        # Create vision client
        if provider == "gemini":
            client = llm_factory.create_vision_client(
                VisionProvider.GEMINI_VISION, model
            )
        elif provider == "ollama" or provider == "llava":
            client = llm_factory.create_vision_client(VisionProvider.LLAVA, model)
        else:
            raise ValueError(f"Unsupported vision provider: {provider}")

        # Process image - handle different input formats
        if isinstance(image_path, dict):
            # Handle error cases from previous steps
            if not image_path.get("success", True):
                raise ValueError(
                    f"Previous step failed: {image_path.get('error', 'Unknown error')}"
                )
            # Extract actual data from success case
            if "base64_images" in image_path:
                image_path = image_path["base64_images"]
            elif "result" in image_path:
                image_path = image_path["result"]
            else:
                raise ValueError(f"Could not extract image data from: {image_path}")

        if isinstance(image_path, list):
            # If it's a list (e.g., from PDF conversion), use the first image
            if len(image_path) > 0:
                image_data = image_path[0]
            else:
                raise ValueError("Empty image list provided")
        elif isinstance(image_path, str):
            # If it's a file path, read and encode it
            import base64

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            # If it's already image data, use it directly
            image_data = image_path

        # Ensure image_data is a string
        if not isinstance(image_data, str):
            raise ValueError(
                f"Expected base64 string, got {type(image_data)}: {image_data}"
            )

        response = client.process_image(image_data, prompt)

        return {
            "analysis": response,
            "image_path": image_path,
            "prompt": prompt,
            "provider": provider,
            "model": model,
        }

    def _handle_chunk_text(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle text chunking step using factory pattern"""
        chunking_factory = self.orchestrator.get_service("chunking_factory")
        if not chunking_factory:
            raise RuntimeError("Chunking service factory not available")

        text = inputs.get("text")
        if not text:
            raise ValueError("text required for chunking")

        # Handle case where text is a dict with response content
        if isinstance(text, dict):
            # Extract text from common response formats
            if "response" in text:
                text = text["response"]
            elif "content" in text:
                text = text["content"]
            elif "message" in text:
                text = text["message"]
            else:
                # Convert dict to string representation
                text = str(text)

        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        # Prepare configuration dictionary from step config
        config_dict = {
            "target_size": step.config.get("target_size", 1000),
            "tolerance": step.config.get("tolerance", 200),
            "preserve_paragraphs": step.config.get("preserve_paragraphs", True),
            "preserve_sentences": step.config.get("preserve_sentences", True),
            "preserve_words": step.config.get("preserve_words", True),
            "paragraph_separator": step.config.get("paragraph_separator", "\n\n"),
            "sentence_pattern": step.config.get("sentence_pattern", r"[.!?]+\s+"),
        }

        # Get or create chunking service with the specified configuration
        chunking_service = chunking_factory.get_or_create_service(config_dict)

        # Chunk the text
        chunks = chunking_service.chunk_text(text)

        result = {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "total_length": len(text),
        }

        # If step has specific outputs defined, map chunks to those output names
        if step.outputs:
            for output_name in step.outputs:
                if output_name not in result:
                    # Map chunks to the custom output name
                    result[output_name] = chunks

        return result

    def _handle_condition(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle conditional step - evaluates condition and returns routing decision"""
        condition_type = step.config.get("condition_type", "llm_decision")

        if condition_type == "llm_decision":
            return self._handle_llm_condition(step, inputs, context)
        elif condition_type == "simple_comparison":
            return self._handle_simple_condition(step, inputs, context)
        elif condition_type == "field_value":
            return self._handle_field_value_condition(step, inputs, context)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

    def _handle_llm_condition(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle LLM-based conditional evaluation"""
        llm_factory = self.orchestrator.get_service("llm_factory")
        if not llm_factory:
            raise RuntimeError("LLM factory not available")

        # Get the condition prompt
        condition_prompt = step.config.get("condition_prompt")
        if not condition_prompt:
            raise ValueError("condition_prompt required for LLM conditional step")

        # Get route options
        route_options = step.config.get("route_options", ["true", "false"])

        # Create a decision schema with the available routes
        decision_schema = {
            "name": "ConditionalDecision",
            "fields": {
                "decision": {
                    "type": "string",
                    "description": f"Choose one of: {', '.join(route_options)}",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explain why this route was chosen",
                },
                "confidence": {
                    "type": "float",
                    "description": "Confidence in the decision (0.0 to 1.0)",
                },
            },
        }

        # Configure LLM
        provider = step.config.get("provider", "gemini")
        model = step.config.get("model")

        # Create structured client
        structured_client = llm_factory.create_structured_client(
            provider=provider,
            schema=self._create_schema_from_dict(decision_schema),
            model=model,
            temperature=step.config.get(
                "temperature", 0.3
            ),  # Lower temp for more consistent decisions
        )

        # Build the full prompt with context
        full_prompt = self._build_condition_prompt(condition_prompt, inputs, context)

        try:
            # Get structured decision
            decision_response = structured_client.chat(full_prompt)

            # Validate the decision is one of the allowed options
            chosen_route = decision_response.decision
            if chosen_route not in route_options:
                # Fallback to first option if invalid
                chosen_route = route_options[0]

            return {
                "condition_result": chosen_route,
                "reasoning": decision_response.reasoning,
                "confidence": decision_response.confidence,
                "route_options": route_options,
                "prompt": full_prompt,
                "provider": provider,
                "model": model,
            }

        except Exception as e:
            # Fallback to default route on error
            default_route = step.config.get("default_route", route_options[0])
            return {
                "condition_result": default_route,
                "reasoning": f"Error in condition evaluation: {str(e)}",
                "confidence": 0.0,
                "route_options": route_options,
                "error": str(e),
            }

    def _handle_simple_condition(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle simple comparison-based conditional evaluation"""
        # Get comparison parameters
        left_value = step.config.get("left_value")
        operator = step.config.get("operator", "==")
        right_value = step.config.get("right_value")

        if left_value is None or right_value is None:
            raise ValueError("left_value and right_value required for simple condition")

        # Resolve values if they're references
        left_value = self._resolve_condition_value(left_value, context, inputs)
        right_value = self._resolve_condition_value(right_value, context, inputs)

        # Evaluate condition
        result = False
        if operator == "==":
            result = left_value == right_value
        elif operator == "!=":
            result = left_value != right_value
        elif operator == ">":
            result = left_value > right_value
        elif operator == "<":
            result = left_value < right_value
        elif operator == ">=":
            result = left_value >= right_value
        elif operator == "<=":
            result = left_value <= right_value
        elif operator == "contains":
            result = str(right_value) in str(left_value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        return {
            "condition_result": "true" if result else "false",
            "left_value": left_value,
            "operator": operator,
            "right_value": right_value,
            "reasoning": f"{left_value} {operator} {right_value} = {result}",
        }

    def _handle_field_value_condition(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle field value routing condition - routes based on the value of a specific field"""
        field_to_check = step.config.get("field_to_check")
        if not field_to_check:
            raise ValueError("field_to_check required for field_value condition")

        route_options = step.config.get("route_options", [])
        default_route = step.config.get("default_route", "default")

        # Get the field value
        field_value = inputs.get(field_to_check)
        if field_value is None:
            # Try to extract from a string response (common with LLM outputs)
            for key, value in inputs.items():
                if isinstance(value, str) and field_to_check.lower() in value.lower():
                    # Look for the field value in the string
                    for option in route_options:
                        if option.lower() in value.lower():
                            field_value = option
                            break

        # Determine route
        if field_value in route_options:
            route = field_value
        else:
            # Check if field_value is a string containing one of the options
            route = default_route
            if isinstance(field_value, str):
                for option in route_options:
                    if option.lower() in field_value.lower():
                        route = option
                        break

        return {
            "route": route,
            "field_checked": field_to_check,
            "field_value": field_value,
            "available_routes": route_options,
            "reasoning": f"Field '{field_to_check}' value '{field_value}' routed to '{route}'",
        }

    def _build_condition_prompt(
        self, base_prompt: str, inputs: Dict[str, Any], context: ExecutionContext
    ) -> str:
        """Build a complete condition prompt with context"""
        prompt_parts = [base_prompt]

        # Add input context
        if inputs:
            prompt_parts.append("\nInput data:")
            for key, value in inputs.items():
                prompt_parts.append(f"- {key}: {value}")

        # Add relevant step outputs from context
        if context.step_outputs:
            prompt_parts.append("\nPrevious step results:")
            for key, value in context.step_outputs.items():
                # Only include recent/relevant outputs to avoid overwhelming the prompt
                if isinstance(value, (str, int, float, bool)):
                    prompt_parts.append(f"- {key}: {value}")
                elif isinstance(value, dict) and len(str(value)) < 200:
                    prompt_parts.append(f"- {key}: {value}")

        return "\n".join(prompt_parts)

    def _resolve_condition_value(
        self, value: Any, context: ExecutionContext, inputs: Dict[str, Any]
    ) -> Any:
        """Resolve a condition value that might be a reference"""
        if isinstance(value, dict) and "from_step" in value:
            # This is a reference to another step's output
            step_id = value["from_step"]
            field = value["field"]

            # Look for the output in step_outputs
            full_key = f"{step_id}.{field}"
            if full_key in context.step_outputs:
                return context.step_outputs[full_key]
            elif step_id in context.step_outputs:
                # Try to get field from the step's result
                step_result = context.step_outputs[step_id]
                if isinstance(step_result, dict) and field in step_result:
                    return step_result[field]
                else:
                    return step_result
            else:
                raise ValueError(
                    f"Could not resolve condition reference: {step_id}.{field}"
                )
        elif isinstance(value, str) and value.startswith("$"):
            # Global variable reference
            var_name = value[1:]
            if var_name in context.global_variables:
                return context.global_variables[var_name]
            elif var_name in inputs:
                return inputs

    def _handle_loop(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle loop step - execute loop body for each item in a list"""
        loop_type = step.config.get("loop_type", "iterate_list")
        max_iterations = step.config.get("max_iterations", 10)

        if loop_type != "iterate_list":
            raise ValueError(f"Unsupported loop type: {loop_type}")

        # Get items to process
        items_to_process = inputs.get("items_to_process", [])
        if not isinstance(items_to_process, list):
            # If it's a string, try to parse it as JSON or split by lines
            if isinstance(items_to_process, str):
                try:
                    items_to_process = json.loads(items_to_process)
                except json.JSONDecodeError:
                    # Try splitting by newlines and filtering empty lines
                    items_to_process = [
                        line.strip()
                        for line in items_to_process.split("\n")
                        if line.strip()
                    ]
            elif isinstance(items_to_process, dict):
                # Convert dict to list of key-value pairs or values
                items_to_process = (
                    list(items_to_process.values()) if items_to_process else []
                )
            else:
                items_to_process = [items_to_process]

        # Limit iterations
        if len(items_to_process) > max_iterations:
            items_to_process = items_to_process[:max_iterations]

        # Get loop body steps
        loop_body = step.config.get("loop_body", [])
        if not loop_body:
            raise ValueError("loop_body is required for loop step")

        # Execute loop body for each item
        all_results = []
        successful_steps = []

        for iteration_count, current_item in enumerate(items_to_process, 1):
            # Create iteration context
            iteration_context = {
                "iteration_count": iteration_count,
                "total_items": len(items_to_process),
                "current_item": current_item,
            }

            # Create new context for this iteration
            loop_context = ExecutionContext(
                global_variables=context.global_variables.copy(),
                conversations=context.conversations.copy(),
                step_outputs=context.step_outputs.copy(),
            )

            # Add loop context variables
            loop_context.global_variables["current_item"] = current_item
            loop_context.global_variables["iteration_context"] = iteration_context

            # Execute loop body steps
            iteration_results = {}
            try:
                for loop_step_config in loop_body:
                    # Convert dict to StepConfig if needed
                    if isinstance(loop_step_config, dict):
                        loop_step = StepConfig(**loop_step_config)
                    else:
                        loop_step = loop_step_config

                    # Resolve inputs for this step
                    resolved_inputs = self._resolve_loop_inputs(
                        loop_step, loop_context, current_item, iteration_context
                    )

                    # Get handler for this step type
                    handler = self.orchestrator.step_registry.get_handler(
                        StepType(loop_step.type)
                    )
                    if not handler:
                        raise ValueError(
                            f"No handler found for step type: {loop_step.type}"
                        )

                    # Execute the step
                    result = handler(loop_step, resolved_inputs, loop_context)

                    # Store result in context
                    loop_context.step_outputs[loop_step.id] = result
                    iteration_results[loop_step.id] = result

                all_results.append(iteration_results)
                successful_steps.append(current_item)

            except Exception as e:
                # Handle step failure
                error_result = {
                    "error": str(e),
                    "failed_item": current_item,
                    "iteration_count": iteration_count,
                }
                all_results.append(error_result)
                # Continue with next iteration unless we want to stop on error
                continue

        # Aggregate results
        return {
            "all_step_results": all_results,
            "successful_steps": successful_steps,
            "total_iterations": len(items_to_process),
            "successful_iterations": len(successful_steps),
        }

    def _resolve_loop_inputs(
        self,
        step: StepConfig,
        context: ExecutionContext,
        current_item: Any,
        iteration_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve inputs for a loop step, handling special loop variables"""
        resolved_inputs = {}

        if not step.inputs:
            return resolved_inputs

        for input_key, input_value in step.inputs.items():
            # Handle special loop variables
            if input_value == "$current_item":
                resolved_inputs[input_key] = current_item
            elif isinstance(input_value, str) and input_value.startswith(
                "$iteration_context."
            ):
                # Extract the context field
                context_field = input_value[len("$iteration_context.") :]
                resolved_inputs[input_key] = iteration_context.get(context_field)
            else:
                # Use standard input resolution
                resolved_inputs[input_key] = self._resolve_input_value(
                    input_value, context
                )

        return resolved_inputs

    def _resolve_input_value(self, input_value: Any, context: ExecutionContext) -> Any:
        """Resolve input value from context, handling references and variables"""
        if isinstance(input_value, dict) and "from_step" in input_value:
            # Reference to another step's output
            step_id = input_value["from_step"]
            field = input_value["field"]

            if step_id in context.step_outputs:
                step_output = context.step_outputs[step_id]
                if isinstance(step_output, dict) and field in step_output:
                    return step_output[field]
                elif hasattr(step_output, field):
                    return getattr(step_output, field)
                else:
                    return step_output
            else:
                raise ValueError(f"Step {step_id} output not found in context")

        elif isinstance(input_value, str) and input_value.startswith("$"):
            # Global variable reference
            var_name = input_value[1:]
            if var_name in context.global_variables:
                return context.global_variables[var_name]
            else:
                raise ValueError(f"Global variable {var_name} not found")

        else:
            # Direct value
            return input_value

    def _handle_file_output(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle file output step - writes data to a file"""
        import json
        import os
        from datetime import datetime
        from pathlib import Path

        # Get configuration
        file_path = step.config.get("file_path")
        format_type = step.config.get("format", "json")
        create_dirs = step.config.get("create_dirs", True)
        append_timestamp = step.config.get("append_timestamp", False)

        # Get content to write
        content = inputs.get("content")
        if content is None:
            # If no specific content field, use all inputs
            content = inputs

        # Handle file path
        if not file_path:
            # Generate default filename based on workflow
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"workflow_output_{timestamp}.{format_type}"
            file_path = os.path.join("output", default_name)

        # Add timestamp to filename if requested
        if append_timestamp:
            file_path = Path(file_path)
            timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
            file_path = (
                file_path.parent / f"{file_path.stem}{timestamp}{file_path.suffix}"
            )
            file_path = str(file_path)

        # Create directories if needed
        if create_dirs:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Format content based on format type
        try:
            if format_type == "json":
                if isinstance(content, (dict, list)):
                    formatted_content = json.dumps(
                        content, indent=2, ensure_ascii=False
                    )
                else:
                    # Wrap non-dict/list content in a structure
                    formatted_content = json.dumps(
                        {"content": content}, indent=2, ensure_ascii=False
                    )

            elif format_type == "text" or format_type == "txt":
                if isinstance(content, dict):
                    # Convert dict to readable text format
                    lines = []
                    for key, value in content.items():
                        if isinstance(value, (dict, list)):
                            lines.append(f"{key}:")
                            lines.append(f"  {json.dumps(value, indent=2)}")
                        else:
                            lines.append(f"{key}: {value}")
                    formatted_content = "\n".join(lines)
                elif isinstance(content, list):
                    formatted_content = "\n".join(str(item) for item in content)
                else:
                    formatted_content = str(content)

            elif format_type == "markdown" or format_type == "md":
                if isinstance(content, dict):
                    lines = ["# Workflow Output", ""]
                    for key, value in content.items():
                        lines.append(f"## {key.replace('_', ' ').title()}")
                        if isinstance(value, (dict, list)):
                            lines.append("```json")
                            lines.append(json.dumps(value, indent=2))
                            lines.append("```")
                        else:
                            lines.append(str(value))
                        lines.append("")
                    formatted_content = "\n".join(lines)
                else:
                    formatted_content = f"# Workflow Output\n\n{str(content)}"

            elif format_type == "csv":
                import csv
                import io

                if isinstance(content, list) and all(
                    isinstance(item, dict) for item in content
                ):
                    # List of dictionaries - convert to CSV
                    output = io.StringIO()
                    if content:
                        writer = csv.DictWriter(output, fieldnames=content[0].keys())
                        writer.writeheader()
                        writer.writerows(content)
                    formatted_content = output.getvalue()
                else:
                    raise ValueError("CSV format requires a list of dictionaries")

            else:
                # Default to string representation
                formatted_content = str(content)

            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)

            # Get file stats
            file_size = os.path.getsize(file_path)

            return {
                "file_path": os.path.abspath(file_path),
                "format": format_type,
                "file_size": file_size,
                "success": True,
                "content_written": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "file_path": file_path,
                "format": format_type,
                "success": False,
                "error": str(e),
                "content_written": False,
                "timestamp": datetime.now().isoformat(),
            }

    def _handle_output(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle output step - formats and returns final output"""
        # If format is specified in config, format the output
        if "format" in step.config:
            format_type = step.config["format"]
            if format_type == "json":
                import json

                serializable_inputs = _serialize_for_json(inputs)
                return json.dumps(serializable_inputs, indent=2)
            elif format_type == "text":
                if isinstance(inputs, dict):
                    return "\n".join(f"{k}: {v}" for k, v in inputs.items())
                else:
                    return str(inputs)

        return inputs

    def _handle_python_function(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle Python function execution step"""
        import traceback
        import time

        function_config = step.config.get("function")
        if not function_config:
            raise ValueError("function configuration required for python_function step")

        start_time = time.time()

        try:
            # Get the function to execute
            if isinstance(function_config, str):
                # Function reference string: "module.function_name"
                function_obj = self._import_function_from_string(function_config)
            elif isinstance(function_config, dict):
                if "module" in function_config and "name" in function_config:
                    # Function reference dict: {"module": "mymodule", "name": "myfunction"}
                    module_name = function_config["module"]
                    function_name = function_config["name"]
                    function_obj = self._import_function_from_module(
                        module_name, function_name
                    )
                elif "code" in function_config:
                    # Inline function code
                    function_obj = self._create_function_from_code(
                        function_config["code"],
                        function_config.get("name", "dynamic_function"),
                    )
                else:
                    raise ValueError(
                        "Function config must contain either 'module'+'name' or 'code'"
                    )
            else:
                raise ValueError("Function config must be string or dict")

            # Prepare function arguments
            function_args = self._prepare_function_arguments(
                function_obj, inputs, step.config
            )

            # Execute the function
            if isinstance(function_args, list):
                # Positional arguments (for built-in functions)
                result = function_obj(*function_args)
            else:
                # Keyword arguments (for regular functions)
                result = function_obj(**function_args)

            execution_time_ms = (time.time() - start_time) * 1000

            # Handle different return types
            if result is None:
                formatted_result = {"success": True, "result": None}
            elif isinstance(result, dict):
                formatted_result = {"success": True, **result}
            else:
                formatted_result = {"success": True, "result": result}

            # Add execution metadata
            formatted_result.update(
                {
                    "execution_time_ms": execution_time_ms,
                    "function_name": getattr(function_obj, "__name__", "unknown"),
                    "function_args": function_args,
                    "step_id": step.id,
                }
            )

            # If step has specific outputs defined, map result to those output names
            if step.outputs:
                for output_name in step.outputs:
                    if output_name not in formatted_result:
                        # Map the main result to the custom output name
                        if "result" in formatted_result:
                            formatted_result[output_name] = formatted_result["result"]
                        else:
                            # For dict results that were spread, try to use the entire result
                            formatted_result[output_name] = result

            return formatted_result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "execution_time_ms": execution_time_ms,
                "function_config": function_config,
                "inputs": inputs,
                "step_id": step.id,
            }

    def _import_function_from_string(self, function_string: str):
        """Import a function from a string like 'module.submodule.function_name'"""
        if "." not in function_string:
            raise ValueError(
                "Function string must contain module and function name separated by '.'"
            )

        module_path, function_name = function_string.rsplit(".", 1)

        # Handle built-in functions specially
        if module_path == "builtins":
            import builtins

            if hasattr(builtins, function_name):
                return getattr(builtins, function_name)
            else:
                raise AttributeError(f"No built-in function named '{function_name}'")

        return self._import_function_from_module(module_path, function_name)

    def _import_function_from_module(self, module_name: str, function_name: str):
        """Import a function from a module"""
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, function_name):
                raise AttributeError(
                    f"Module '{module_name}' has no function '{function_name}'"
                )

            function_obj = getattr(module, function_name)
            if not callable(function_obj):
                raise TypeError(
                    f"'{function_name}' in module '{module_name}' is not callable"
                )

            return function_obj

        except ImportError as e:
            raise ImportError(f"Could not import module '{module_name}': {e}")

    def _create_function_from_code(
        self, code: str, function_name: str = "dynamic_function"
    ):
        """Create a function from code string"""
        import types
        import time

        # Create a new module to execute the code in
        module = types.ModuleType("dynamic_module")

        # Add common imports that might be needed
        module.__dict__.update(
            {
                "__builtins__": __builtins__,
                "__file__": os.path.join(
                    os.getcwd(), "dynamic_function.py"
                ),  # Provide __file__ for relative paths
                "__name__": "dynamic_function",
                "__package__": None,
                "json": json,
                "os": os,
                "sys": sys,
                "time": time,
                "datetime": __import__("datetime"),
                "re": __import__("re"),
                "math": __import__("math"),
                "random": __import__("random"),
            }
        )

        try:
            # Execute the code in the module namespace
            exec(code, module.__dict__)

            # Look for the function
            if function_name in module.__dict__:
                function_obj = module.__dict__[function_name]
                if not callable(function_obj):
                    raise TypeError(f"'{function_name}' is not callable")
                return function_obj
            else:
                # If specific function name not found, look for any callable
                callables = [
                    obj
                    for name, obj in module.__dict__.items()
                    if callable(obj) and not name.startswith("_")
                ]

                if len(callables) == 1:
                    return callables[0]
                elif len(callables) > 1:
                    callable_names = [obj.__name__ for obj in callables]
                    raise ValueError(
                        f"Multiple functions found in code: {callable_names}. "
                        f"Please specify function name or define only one function."
                    )
                else:
                    raise ValueError("No callable function found in the provided code")

        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in function code: {e}")
        except Exception as e:
            raise RuntimeError(f"Error executing function code: {e}")

    def _prepare_function_arguments(
        self, function_obj, inputs: Dict[str, Any], step_config: Dict[str, Any]
    ):
        """Prepare arguments for function execution based on function signature"""
        import inspect

        # Special handling for built-in functions that don't support keyword arguments
        if (
            hasattr(function_obj, "__module__")
            and function_obj.__module__ == "builtins"
        ):
            # For built-in functions, try to match parameters positionally
            # Common built-ins like len, str, int, etc. take one argument
            if function_obj.__name__ in [
                "len",
                "str",
                "int",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
            ]:
                # These functions take one positional argument
                if len(inputs) == 1:
                    return list(inputs.values())  # Return as positional args
                else:
                    # Try to find the most likely argument
                    for key in ["obj", "value", "data", "input"]:
                        if key in inputs:
                            return [inputs[key]]
                    # Fall back to first value
                    return [list(inputs.values())[0]] if inputs else []

        # Get function signature
        try:
            signature = inspect.signature(function_obj)
        except (ValueError, TypeError):
            # If we can't get signature, pass all inputs as kwargs
            return inputs

        function_args = {}

        # Handle each parameter in the function signature
        for param_name, param in signature.parameters.items():
            if param_name in inputs:
                # Direct parameter match
                function_args[param_name] = inputs[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Parameter has default value, skip it
                continue
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # **kwargs parameter - pass all remaining inputs
                used_params = set(function_args.keys())
                remaining_inputs = {
                    k: v for k, v in inputs.items() if k not in used_params
                }
                function_args.update(remaining_inputs)
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                # *args parameter - not supported in this context
                continue
            else:
                # Required parameter not found in inputs
                # Check if it's available in step config parameter mapping
                param_mapping = step_config.get("parameter_mapping", {})
                if param_name in param_mapping:
                    mapped_input = param_mapping[param_name]
                    if mapped_input in inputs:
                        function_args[param_name] = inputs[mapped_input]
                    else:
                        raise ValueError(
                            f"Required parameter '{param_name}' mapped to '{mapped_input}' "
                            f"but '{mapped_input}' not found in inputs"
                        )
                else:
                    raise ValueError(
                        f"Required parameter '{param_name}' not found in inputs"
                    )

        return function_args

    def _handle_graph_memory_format(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle graph memory formatting step"""
        try:
            # Get required services
            generation_service = self.orchestrator.get_service("generation")
            prompt_service = self.orchestrator.get_service("prompt")

            if not generation_service:
                raise RuntimeError(
                    "Generation service not available for graph formatting"
                )
            if not prompt_service:
                raise RuntimeError("Prompt service not available for graph formatting")

            # Import the graph formatter service
            from memory.graph_formatter_service import GraphFormatterService

            # Create formatter instance
            formatter = GraphFormatterService(generation_service, prompt_service)

            # Get content to format
            content = inputs.get("content")
            if not content:
                raise ValueError("content required for graph memory formatting")

            # Get optional configuration
            context_info = inputs.get("context", step.config.get("context"))
            extraction_mode = step.config.get("extraction_mode", "comprehensive")

            # Format the memory as graph
            graph_format = formatter.format_memory_as_graph(
                content=content, context=context_info, extraction_mode=extraction_mode
            )

            # Return structured result
            return {
                "success": True,
                "graph_format": graph_format.dict(),
                "entity_count": len(graph_format.entities),
                "relationship_count": len(graph_format.relationships),
                "extraction_method": graph_format.extraction_metadata.get(
                    "extraction_method"
                ),
                "summary": graph_format.summary,
                "original_content": content,
                "extraction_mode": extraction_mode,
            }

        except Exception as e:
            # Return error information for debugging
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "content": inputs.get("content", ""),
                "extraction_mode": step.config.get("extraction_mode", "comprehensive"),
            }

    def _handle_tool_call(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle tool calling step - executes tools as configured in workflow"""
        try:
            import asyncio
            import uuid
            from tools.tool_service import ToolService
            from llm.llm_types import LLMConfig, LLMProvider, ChatMessage

            # Get configuration from step
            provider = step.config.get("provider", "gemini")
            model = step.config.get("model")
            tools = step.config.get("tools", [])
            tool_category = step.config.get("tool_category")
            tool_choice = step.config.get("tool_choice", "auto")
            auto_execute = step.config.get("auto_execute", True)
            max_iterations = step.config.get("max_iterations", 5)
            temperature = step.config.get("temperature", 0.1)
            
            # Get user message from inputs
            user_message = inputs.get("message", inputs.get("user_input", ""))
            if not user_message:
                return {
                    "success": False,
                    "error": "No user message provided",
                    "tool_calls": [],
                    "responses": [],
                }
            
            # Get system prompt/message
            system_message = step.config.get("prompt", "You are a helpful assistant.")
            
            # Create tool service
            credential_manager = getattr(context, "credential_manager", None)
            tool_service = ToolService(credential_manager=credential_manager)
            
            # Create LLM client
            if provider == "gemini":
                from llm.text_clients import GeminiTextClient
                config = LLMConfig(
                    provider=LLMProvider.GEMINI,
                    model=model or "gemini-1.5-flash",
                    temperature=temperature
                )
                llm_client = GeminiTextClient(config, credential_manager=credential_manager)
            elif provider == "ollama":
                from llm.text_clients import OllamaTextClient
                config = LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model=model or "llama3.1:8b",
                    temperature=temperature
                )
                llm_client = OllamaTextClient(config, credential_manager=credential_manager)
            elif provider == "anthropic":
                from llm.text_clients import AnthropicTextClient
                config = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model=model or "claude-3-5-sonnet-20241022",
                    temperature=temperature
                )
                llm_client = AnthropicTextClient(config, credential_manager=credential_manager)
            elif provider == "openrouter":
                from llm.text_clients import OpenRouterTextClient
                config = LLMConfig(
                    provider=LLMProvider.OPENROUTER,
                    model=model or "anthropic/claude-3.5-sonnet",
                    temperature=temperature
                )
                llm_client = OpenRouterTextClient(config, credential_manager=credential_manager)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Set up conversation history
            conversation_history = []
            if system_message:
                conversation_history.append(ChatMessage(role="system", content=system_message))
            
            # Initialize tracking variables
            all_tool_calls = []
            all_tool_results = []
            all_responses = []
            current_message = user_message
            iterations = 0
            
            # Run the conversation loop with tools
            for iteration in range(max_iterations):
                iterations += 1
                
                # Prepare tools for this iteration if any are configured
                formatted_tools = None
                if tools or tool_category:
                    try:
                        # Get tools in provider-specific format - handle async properly
                        import concurrent.futures
                        
                        def run_tool_preparation():
                            return asyncio.run(
                                tool_service.prepare_tools_for_provider(
                                    provider, tools, tool_category
                                )
                            )
                        
                        try:
                            # Try to get existing loop
                            loop = asyncio.get_running_loop()
                            # We're in an async context, use ThreadPoolExecutor
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(run_tool_preparation)
                                tool_config = future.result()
                        except RuntimeError:
                            # No running loop, we can run asyncio directly
                            tool_config = asyncio.run(
                                tool_service.prepare_tools_for_provider(
                                    provider, tools, tool_category
                                )
                            )
                        
                        if tool_config:
                            # Extract tools in the format expected by each provider
                            if provider == "gemini":
                                # For Gemini, extract function declarations from the nested structure
                                if 'tools' in tool_config and tool_config['tools']:
                                    if 'functionDeclarations' in tool_config['tools'][0]:
                                        formatted_tools = tool_config['tools'][0]['functionDeclarations']
                                    else:
                                        formatted_tools = tool_config['tools']
                                else:
                                    formatted_tools = tool_config
                            else:
                                # For other providers, use the tools list directly
                                formatted_tools = tool_config.get('tools', tool_config)
                    except Exception as e:
                        print(f"Warning: Failed to prepare tools: {e}")
                        formatted_tools = None
                
                # Make LLM call with tool support
                try:
                    llm_response = llm_client.chat_with_tool_support(
                        current_message,
                        tools=formatted_tools,
                        tool_choice=tool_choice,
                        chat_history=conversation_history
                    )
                    
                    text_response = llm_response.get("response", "")
                    tool_calls = llm_response.get("tool_calls", [])
                    finish_reason = llm_response.get("finish_reason", "stop")
                    
                    all_responses.append(text_response)
                    
                    # Add this response to conversation history
                    if text_response:
                        conversation_history.append(ChatMessage(role="assistant", content=text_response))
                    
                    # If no tool calls, we're done
                    if not tool_calls or not auto_execute:
                        if not auto_execute and tool_calls:
                            all_tool_calls.extend(tool_calls)
                        break
                    
                    # Execute tool calls
                    if tool_calls:
                        all_tool_calls.extend(tool_calls)
                        
                        # Convert tool calls to ToolCall objects and execute them
                        from tools.tool_types import ToolCall
                        tool_call_objects = []
                        for tc in tool_calls:
                            if 'function' in tc:
                                func = tc['function']
                                tool_call_objects.append(ToolCall(
                                    id=tc.get('id', str(uuid.uuid4())),
                                    name=func['name'],
                                    parameters=func.get('arguments', {})
                                ))
                        
                        # Execute tools - handle async properly
                        try:
                            def run_tool_execution():
                                return asyncio.run(
                                    tool_service.execute_tool_calls(tool_call_objects)
                                )
                            
                            try:
                                # Try to get existing loop
                                loop = asyncio.get_running_loop()
                                # We're in an async context, use ThreadPoolExecutor
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(run_tool_execution)
                                    tool_results = future.result()
                            except RuntimeError:
                                # No running loop, we can run asyncio directly
                                tool_results = asyncio.run(
                                    tool_service.execute_tool_calls(tool_call_objects)
                                )
                            
                            all_tool_results.extend(tool_results)
                            
                            # Add tool results to conversation
                            tool_results_text = []
                            for result in tool_results:
                                if result.error:
                                    tool_results_text.append(f"Tool {result.name} failed: {result.error}")
                                else:
                                    tool_results_text.append(f"Tool {result.name} result: {result.result}")
                            
                            # Continue conversation with tool results
                            current_message = f"Tool execution results:\n" + "\n".join(tool_results_text) + "\n\nPlease continue or provide a summary based on these results."
                            
                        except Exception as e:
                            print(f"Tool execution failed: {e}")
                            # Continue with error message
                            current_message = f"Tool execution failed: {e}. Please respond without using tools."
                    
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"LLM call failed: {str(e)}",
                        "error_type": type(e).__name__,
                        "tool_calls": all_tool_calls,
                        "tool_results": all_tool_results,
                        "conversation_history": [
                            {"role": msg.role, "content": msg.content} for msg in conversation_history
                        ],
                        "iterations": iterations,
                    }
            
            # Format final response
            final_response = "\n".join(all_responses) if all_responses else "No response generated"
            
            return {
                "success": True,
                "response": final_response,
                "tool_calls": all_tool_calls,
                "tool_results": [
                    {
                        "name": tr.name,
                        "result": tr.result,
                        "error": tr.error,
                        "tool_call_id": tr.tool_call_id
                    } for tr in all_tool_results
                ],
                "conversation_history": [
                    {"role": msg.role, "content": msg.content} for msg in conversation_history
                ],
                "iterations": iterations,
                "provider": provider,
                "model": config.model,
                "tools_used": len(all_tool_calls),
                "auto_execute": auto_execute,
                "finish_reason": "completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "tool_calls": [],
                "tool_results": [],
                "conversation_history": [],
                "iterations": 0,
            }

    def _handle_http_request(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle HTTP request step"""
        try:
            # Import HTTP request service
            from services.http_request_service import (
                HttpRequestService,
                HttpRequestConfig,
                HttpMethod,
            )

            # Get URL (required)
            url = inputs.get("url")
            if not url:
                raise ValueError("url is required for HTTP request step")

            # Get method (default to GET)
            method_str = step.config.get("method", "GET").upper()
            try:
                method = HttpMethod(method_str)
            except ValueError:
                raise ValueError(f"Invalid HTTP method: {method_str}")

            # Get optional parameters
            headers = inputs.get("headers", {})
            params = inputs.get("params", {})
            json_data = inputs.get("json_data")
            form_data = inputs.get("form_data")
            data = inputs.get("data")

            # Get configuration from step config
            timeout = step.config.get("timeout", 30.0)
            follow_redirects = step.config.get("follow_redirects", True)
            verify_ssl = step.config.get("verify_ssl", True)
            max_retries = step.config.get("max_retries", 3)
            backoff_factor = step.config.get("backoff_factor", 1.0)

            # Authentication configuration
            auth_type = step.config.get("auth_type")
            auth_credentials = step.config.get("auth_credentials", {})

            # Also check for auth credentials in inputs (for dynamic auth)
            if "auth_credentials" in inputs:
                auth_credentials.update(inputs["auth_credentials"])

            # Create HTTP request configuration
            config = HttpRequestConfig(
                url=url,
                method=method,
                headers=headers,
                params=params,
                json_data=json_data,
                form_data=form_data,
                data=data,
                timeout=timeout,
                follow_redirects=follow_redirects,
                verify_ssl=verify_ssl,
                auth_type=auth_type,
                auth_credentials=auth_credentials if auth_credentials else None,
            )

            # Create HTTP service with credential manager if available
            credential_manager = None
            if hasattr(context, "credential_manager") and context.credential_manager:
                credential_manager = context.credential_manager

            # Create service instance
            service = HttpRequestService(
                credential_manager=credential_manager,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
            )

            # Execute request
            import asyncio

            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # In an async context, need to run in a new thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, service.request(config))
                        response = future.result()
                else:
                    # No running loop, safe to use asyncio.run
                    response = asyncio.run(service.request(config))
            except RuntimeError:
                # No event loop, safe to use asyncio.run
                response = asyncio.run(service.request(config))

            # Build response
            result = {
                "success": response.is_success,
                "status_code": response.status_code,
                "headers": response.headers,
                "url": response.url,
                "method": response.method,
                "elapsed_time": response.elapsed_time,
                "content_length": len(response.content),
                "is_client_error": response.is_client_error,
                "is_server_error": response.is_server_error,
            }

            # Add response content based on configuration
            include_content = step.config.get("include_content", True)
            include_text = step.config.get("include_text", True)

            if include_content:
                result["content"] = response.content.decode("utf-8", errors="replace")

            if include_text:
                result["text"] = response.text

            # Try to parse JSON if possible
            content_type = response.headers.get(
                "content-type", response.headers.get("Content-Type", "")
            ).lower()
            if "application/json" in content_type or response.text.strip().startswith(
                "{"
            ):
                try:
                    result["json"] = response.json()
                except ValueError:
                    # Not valid JSON, skip
                    pass

            # Add metadata
            result["metadata"] = {
                "request_config": {
                    "url": config.url,
                    "method": config.method.value,
                    "headers": dict(config.headers),
                    "params": dict(config.params),
                    "auth_type": config.auth_type,
                    "timeout": config.timeout,
                    "follow_redirects": config.follow_redirects,
                    "verify_ssl": config.verify_ssl,
                },
                "response_info": {
                    "content_type": content_type,
                    "content_encoding": response.headers.get(
                        "content-encoding", response.headers.get("Content-Encoding")
                    ),
                    "server": response.headers.get(
                        "server", response.headers.get("Server")
                    ),
                },
            }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "status_code": None,
                "url": inputs.get("url"),
                "method": step.config.get("method", "GET"),
                "elapsed_time": 0.0,
                "metadata": {
                    "request_config": {
                        "url": inputs.get("url"),
                        "method": step.config.get("method", "GET"),
                    },
                    "error_details": {"error": str(e), "error_type": type(e).__name__},
                },
            }

    def _handle_agent_call(
        self, step: StepConfig, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Any:
        """Handle calling another agent workflow"""
        import os
        import time
        from .orchestrator import AgentOrchestrator, WorkflowExecutor

        try:
            # Get agent reference configuration
            agent_config = step.config
            agent_file = agent_config.get("agent_file")
            timeout = agent_config.get("timeout", 300)
            inherit_context = agent_config.get("inherit_context", True)
            return_outputs = agent_config.get("return_outputs", [])

            if not agent_file:
                raise ValueError(
                    "agent_file must be specified in agent_call step config"
                )

            # Resolve agent file path (relative to current workflow)
            if not os.path.isabs(agent_file):
                # Get the directory of the current workflow if available
                current_dir = os.getcwd()
                if "_workflow_file_path" in context.global_variables:
                    current_dir = os.path.dirname(
                        context.global_variables["_workflow_file_path"]
                    )
                agent_file = os.path.join(current_dir, agent_file)

            if not os.path.exists(agent_file):
                raise FileNotFoundError(f"Agent file not found: {agent_file}")

            # Check for circular dependencies
            agent_name = os.path.basename(agent_file)
            max_call_depth = 5  # Prevent infinite recursion

            if agent_name in context.agent_call_stack:
                raise RuntimeError(
                    f"Circular agent dependency detected: {agent_name} is already in call stack {context.agent_call_stack}"
                )

            if len(context.agent_call_stack) >= max_call_depth:
                raise RuntimeError(
                    f"Maximum agent call depth ({max_call_depth}) exceeded"
                )

            # Create child orchestrator and load workflow
            child_orchestrator = AgentOrchestrator()
            child_workflow = child_orchestrator.load_workflow_from_file(agent_file)

            # Prepare child context
            if inherit_context:
                # Create a new context that inherits from parent
                child_context = ExecutionContext()
                child_context.step_outputs = (
                    {}
                )  # Start fresh but can access parent via parent_context
                child_context.global_variables = context.global_variables.copy()
                child_context.current_step_index = 0
                child_context.conversations = (
                    context.conversations
                )  # Share conversations
                child_context.active_conversation_id = context.active_conversation_id
                child_context.step_iteration_counts = {}
                child_context.step_iteration_history = {}
                child_context.global_config = context.global_config
                child_context.agent_call_stack = context.agent_call_stack + [agent_name]
                child_context.parent_context = context
                child_context.agent_outputs = context.agent_outputs.copy()
            else:
                # Create isolated context
                child_context = ExecutionContext()
                child_context.agent_call_stack = [agent_name]
                child_context.parent_context = None

            # Create child executor with the inherited/isolated context
            child_executor = WorkflowExecutor(child_orchestrator)
            child_executor.context = child_context

            # Execute child workflow with timeout
            start_time = time.time()
            try:
                # Pass inputs as initial inputs to child workflow
                child_result = child_executor.execute(child_workflow, inputs)
                execution_time = time.time() - start_time

                if execution_time > timeout:
                    raise TimeoutError(
                        f"Agent call timed out after {execution_time:.2f}s (limit: {timeout}s)"
                    )

                # Extract specified outputs or return all
                if return_outputs:
                    filtered_outputs = {}
                    for output_key in return_outputs:
                        if output_key in child_result.step_outputs:
                            filtered_outputs[output_key] = child_result.step_outputs[
                                output_key
                            ]
                    result_data = filtered_outputs
                else:
                    result_data = child_result.step_outputs

                # Store agent outputs in parent context for potential reuse
                context.agent_outputs[agent_name] = result_data

                return {
                    "success": True,
                    "agent_name": agent_name,
                    "execution_time": execution_time,
                    "outputs": result_data,
                    "final_output": child_result.final_output,
                    "metadata": {
                        "agent_file": agent_file,
                        "timeout": timeout,
                        "inherit_context": inherit_context,
                        "return_outputs": return_outputs,
                        "total_steps": len(child_workflow.steps),
                    },
                }

            except Exception as child_error:
                execution_time = time.time() - start_time
                raise RuntimeError(
                    f"Child agent execution failed: {str(child_error)}"
                ) from child_error

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "agent_file": agent_config.get("agent_file"),
                "metadata": {
                    "step_id": step.id,
                    "step_type": "agent_call",
                    "config": agent_config,
                },
            }
