"""
Example usage of the prompt management system
"""

from prompt import create_prompt_service, PromptRole, PromptStatus


def create_sample_prompts():
    """Create sample prompts to demonstrate the system"""
    
    # Create prompt service
    prompt_service = create_prompt_service("file", storage_path="./example_prompts")
    
    print("Creating sample prompts...")
    
    # 1. Simple document analysis prompt
    doc_analysis_content = [
        {
            "role": "system",
            "content": "You are an expert document analyst. Analyze documents and provide structured insights."
        },
        {
            "role": "user", 
            "content": {
                "template": "Please analyze the following document and extract key information:\n\n{{ document_text }}\n\nFocus on: {{ focus_areas | join(', ') }}",
                "required_variables": ["document_text"],
                "variables": {
                    "focus_areas": ["main topics", "key findings", "recommendations"]
                }
            }
        }
    ]
    
    doc_prompt = prompt_service.create_prompt(
        prompt_id="document_analysis",
        name="Document Analysis Prompt",
        content=doc_analysis_content,
        version="1.0.0",
        metadata={
            "author": "AI Team",
            "description": "Analyzes documents and extracts structured information",
            "category": "analysis",
            "tags": ["document", "analysis", "extraction"],
            "model_recommendations": ["gemini-1.5-pro", "claude-3-sonnet"]
        },
        status=PromptStatus.ACTIVE
    )
    
    print(f"Created prompt: {doc_prompt.id} v{doc_prompt.version.version}")
    
    # 2. Chat agent prompt with conversation context
    chat_content = [
        {
            "role": "system",
            "content": {
                "template": "You are {{ agent_personality }}, a helpful AI assistant. {{ special_instructions }}",
                "variables": {
                    "agent_personality": "a friendly and knowledgeable",
                    "special_instructions": "Always provide clear, accurate, and helpful responses."
                }
            }
        },
        {
            "role": "user",
            "content": "{{ user_message }}"
        }
    ]
    
    chat_prompt = prompt_service.create_prompt(
        prompt_id="friendly_chat",
        name="Friendly Chat Assistant",
        content=chat_content,
        version="1.0.0",
        metadata={
            "author": "AI Team",
            "description": "General purpose friendly chat assistant",
            "category": "conversation",
            "tags": ["chat", "assistant", "general"],
            "use_case": "Interactive conversations"
        },
        status=PromptStatus.ACTIVE
    )
    
    print(f"Created prompt: {chat_prompt.id} v{chat_prompt.version.version}")
    
    # 3. Structured data extraction prompt
    extraction_content = [
        {
            "role": "system",
            "content": "You are a data extraction specialist. Extract structured information from unstructured text."
        },
        {
            "role": "user",
            "content": {
                "template": "Extract {{ data_type }} from the following text:\n\n{{ input_text }}\n\nReturn the information in a structured format.",
                "required_variables": ["input_text", "data_type"]
            }
        }
    ]
    
    extraction_prompt = prompt_service.create_prompt(
        prompt_id="data_extraction",
        name="Structured Data Extraction",
        content=extraction_content,
        version="1.0.0", 
        metadata={
            "author": "Data Team",
            "description": "Extracts structured data from unstructured text",
            "category": "extraction",
            "tags": ["extraction", "structured", "data"],
            "model_recommendations": ["gemini-1.5-flash"]
        },
        status=PromptStatus.ACTIVE
    )
    
    print(f"Created prompt: {extraction_prompt.id} v{extraction_prompt.version.version}")
    
    return prompt_service


def demonstrate_prompt_usage():
    """Demonstrate prompt rendering and usage"""
    
    prompt_service = create_prompt_service("file", storage_path="./example_prompts")
    
    print("\n=== Demonstrating Prompt Usage ===")
    
    # 1. Render document analysis prompt
    print("\n1. Document Analysis Prompt:")
    rendered = prompt_service.render_prompt(
        "document_analysis",
        context={
            "document_text": "This quarterly report shows 15% revenue growth...",
            "focus_areas": ["financial performance", "growth metrics", "future outlook"]
        }
    )
    
    if rendered:
        for message in rendered:
            print(f"  {message['role']}: {message['content'][:100]}...")
    
    # 2. Render chat prompt
    print("\n2. Chat Assistant Prompt:")
    rendered = prompt_service.render_prompt(
        "friendly_chat",
        context={
            "user_message": "What's the weather like today?",
            "agent_personality": "a helpful weather expert"
        }
    )
    
    if rendered:
        for message in rendered:
            print(f"  {message['role']}: {message['content']}")
    
    # 3. Create a new version
    print("\n3. Creating new version of chat prompt...")
    new_version = prompt_service.create_prompt_version(
        "friendly_chat",
        "1.1.0",
        changelog="Added more personality options and context awareness"
    )
    
    if new_version:
        print(f"Created new version: {new_version.version.version}")
        print(f"Status: {new_version.status.value}")


def demonstrate_workflow_integration():
    """Show how prompts integrate with agent workflows"""
    
    print("\n=== Workflow Integration Example ===")
    
    # Example workflow configuration using prompt references
    workflow_with_prompts = {
        "name": "document_qa_agent",
        "description": "Agent that analyzes documents and answers questions using managed prompts",
        "config": {
            "default_llm_provider": "gemini"
        },
        "steps": [
            {
                "id": "analyze_document",
                "type": "llm_structured",
                "description": "Analyze document using managed prompt",
                "prompt_ref": {
                    "prompt_id": "document_analysis",
                    "version": "1.0.0",
                    "context_variables": {
                        "focus_areas": ["key insights", "important metrics"]
                    }
                },
                "config": {
                    "provider": "gemini",
                    "response_schema": "classification"
                },
                "inputs": {
                    "document_text": "$document_content"
                },
                "outputs": ["analysis_result"]
            },
            {
                "id": "answer_question",
                "type": "llm_chat",
                "description": "Answer user question using managed prompt",
                "prompt_ref": {
                    "prompt_id": "friendly_chat",
                    "version": "1.0.0"
                },
                "config": {
                    "provider": "gemini"
                },
                "inputs": {
                    "user_message": "$user_question"
                },
                "outputs": ["response"]
            },
            {
                "id": "output_result",
                "type": "output",
                "inputs": {
                    "analysis": {
                        "from_step": "analyze_document",
                        "field": "analysis_result"
                    },
                    "answer": {
                        "from_step": "answer_question", 
                        "field": "response"
                    }
                }
            }
        ]
    }
    
    print("Example workflow configuration:")
    import json
    print(json.dumps(workflow_with_prompts, indent=2))


def demonstrate_evaluation():
    """Demonstrate prompt evaluation capabilities"""
    
    print("\n=== Prompt Evaluation Example ===")
    
    prompt_service = create_prompt_service("file", storage_path="./example_prompts")
    
    # Simulate some execution data
    print("Simulating prompt executions...")
    
    for i in range(5):
        prompt_service.log_execution(
            prompt_id="document_analysis",
            prompt_version="1.0.0",
            execution_context={"document_text": f"Sample document {i}"},
            rendered_messages=[
                {"role": "system", "content": "You are an expert..."},
                {"role": "user", "content": f"Analyze document {i}"}
            ],
            llm_provider="gemini",
            llm_model="gemini-1.5-pro",
            llm_response=f"Analysis result for document {i}",
            execution_time_ms=1200 + i * 100,
            success=True
        )
    
    # Get evaluation report
    from prompt.evaluation_service import EvaluationService
    
    eval_service = EvaluationService(prompt_service.storage)
    report = eval_service.generate_evaluation_report("document_analysis", days_back=1)
    
    print("\nEvaluation Report:")
    print(f"  Prompt: {report['prompt_id']}")
    print(f"  Total Executions: {report['metrics']['total_executions']}")
    print(f"  Success Rate: {report['metrics']['success_rate']:.1%}")
    print(f"  Avg Response Time: {report['metrics']['average_response_time_ms']:.0f}ms")
    print(f"  Performance Grade: {report['grade']}")
    print(f"  Recommendations: {report['recommendations'][0]}")


def main():
    """Run all examples"""
    print("🧱 AI Lego Bricks - Prompt Management System Examples")
    print("=" * 60)
    
    # Create sample prompts
    prompt_service = create_sample_prompts()
    
    # Demonstrate usage
    demonstrate_prompt_usage()
    
    # Show workflow integration
    demonstrate_workflow_integration()
    
    # Demonstrate evaluation
    demonstrate_evaluation()
    
    # Show statistics
    print("\n=== System Statistics ===")
    stats = prompt_service.get_stats()
    print(f"Storage Backend: {stats['storage_backend']}")
    print(f"Active Prompts: {stats['active_prompts_count']}")
    print(f"Cache Size: {stats['cache_size']}")
    
    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()