"""
Complete example demonstrating the prompt management system
integrated with agent orchestration workflows.
"""

import json
from agent_orchestration import AgentOrchestrator
from prompt import create_prompt_service, PromptStatus


def setup_sample_prompts():
    """Create sample prompts for demonstration"""
    
    print("🎯 Setting up prompt management system...")
    
    # Create prompt service (auto-detects storage backend)
    prompt_service = create_prompt_service("auto")
    
    # 1. Document Analysis Prompt
    doc_analysis_prompt = prompt_service.create_prompt(
        prompt_id="smart_document_analyzer",
        name="Smart Document Analyzer",
        content=[
            {
                "role": "system",
                "content": {
                    "template": """You are an expert {{ domain }} analyst with {{ experience_years }} years of experience.
                    
Your analysis style: {{ analysis_style }}
Current focus: {{ current_focus }}""",
                    "variables": {
                        "domain": "document",
                        "experience_years": "10+",
                        "analysis_style": "thorough and methodical",
                        "current_focus": "extracting actionable insights"
                    }
                }
            },
            {
                "role": "user",
                "content": {
                    "template": """Please analyze this {{ document_type }}:

{{ document_content }}

Analysis Requirements:
{% for requirement in analysis_requirements %}
- {{ requirement }}
{% endfor %}

{% if include_recommendations %}
Please include specific recommendations and next steps.
{% endif %}

Format: Provide a structured analysis with clear sections.""",
                    "required_variables": ["document_type", "document_content"],
                    "variables": {
                        "analysis_requirements": [
                            "Key themes and topics",
                            "Important facts and figures", 
                            "Notable patterns or trends"
                        ],
                        "include_recommendations": True
                    }
                }
            }
        ],
        version="1.0.0",
        metadata={
            "author": "AI Team",
            "description": "Advanced document analyzer with customizable focus",
            "category": "analysis",
            "tags": ["document", "analysis", "smart", "customizable"],
            "model_recommendations": ["gemini-1.5-pro", "claude-3-sonnet"],
            "use_case": "Comprehensive document analysis with domain expertise"
        },
        status=PromptStatus.ACTIVE
    )
    
    print(f"✅ Created prompt: {doc_analysis_prompt.id} v{doc_analysis_prompt.version.version}")
    
    # 2. Q&A Assistant Prompt  
    qa_prompt = prompt_service.create_prompt(
        prompt_id="contextual_qa_assistant",
        name="Contextual Q&A Assistant",
        content=[
            {
                "role": "system",
                "content": {
                    "template": """You are {{ assistant_personality }}, designed to answer questions using provided context.

Expertise Level: {{ expertise_level }}
Communication Style: {{ communication_style }}

Guidelines:
{% for guideline in guidelines %}
- {{ guideline }}
{% endfor %}""",
                    "variables": {
                        "assistant_personality": "a knowledgeable and helpful AI assistant",
                        "expertise_level": "Expert-level knowledge across domains",
                        "communication_style": "Clear, concise, and engaging",
                        "guidelines": [
                            "Always cite specific parts of the context when possible",
                            "If context doesn't contain the answer, say so clearly",
                            "Provide practical, actionable information",
                            "Use examples to clarify complex concepts"
                        ]
                    }
                }
            },
            {
                "role": "user",
                "content": {
                    "template": """Context Information:
{{ context_information }}

Question: {{ user_question }}

{% if answer_format %}
Please format your answer as: {{ answer_format }}
{% endif %}""",
                    "required_variables": ["context_information", "user_question"]
                }
            }
        ],
        version="1.0.0",
        metadata={
            "author": "AI Team",
            "description": "Context-aware Q&A assistant with citation capabilities",
            "category": "question-answering",
            "tags": ["qa", "context", "assistant", "citation"],
            "model_recommendations": ["gemini-1.5-flash", "claude-3-haiku"]
        },
        status=PromptStatus.ACTIVE
    )
    
    print(f"✅ Created prompt: {qa_prompt.id} v{qa_prompt.version.version}")
    
    return prompt_service


def create_workflow_with_managed_prompts():
    """Create a workflow that uses managed prompts"""
    
    workflow_config = {
        "name": "intelligent_document_qa_system",
        "description": "Advanced document Q&A system using managed prompts for easy optimization",
        "config": {
            "memory_backend": "auto",
            "default_llm_provider": "gemini",
            "default_model": "gemini-1.5-pro"
        },
        "steps": [
            {
                "id": "process_document",
                "type": "document_processing",
                "description": "Extract and enhance text from document",
                "config": {
                    "enhance_with_llm": True,
                    "semantic_analysis": True
                },
                "inputs": {
                    "file_path": "$document_path"
                },
                "outputs": ["text", "enhanced_text", "key_points"]
            },
            {
                "id": "analyze_document", 
                "type": "llm_structured",
                "description": "Analyze document using smart analyzer prompt",
                "prompt_ref": {
                    "prompt_id": "smart_document_analyzer",
                    "version": "1.0.0",
                    "context_variables": {
                        "domain": "business intelligence",
                        "analysis_style": "strategic and actionable",
                        "current_focus": "identifying opportunities and risks"
                    }
                },
                "config": {
                    "provider": "gemini",
                    "response_schema": {
                        "name": "DocumentAnalysis",
                        "fields": {
                            "executive_summary": {"type": "string", "description": "Brief executive summary"},
                            "key_themes": {"type": "list", "description": "Main themes identified"},
                            "important_metrics": {"type": "list", "description": "Key numbers and metrics"},
                            "recommendations": {"type": "list", "description": "Actionable recommendations"},
                            "risk_factors": {"type": "list", "description": "Potential risks identified"},
                            "confidence_score": {"type": "float", "description": "Analysis confidence 0-1"}
                        }
                    }
                },
                "inputs": {
                    "document_type": "business report",
                    "document_content": {
                        "from_step": "process_document",
                        "field": "enhanced_text"
                    },
                    "analysis_requirements": [
                        "Strategic implications",
                        "Financial impact assessment",
                        "Operational considerations",
                        "Market dynamics"
                    ]
                },
                "outputs": ["structured_response", "prompt_id", "execution_time_ms"]
            },
            {
                "id": "store_analysis",
                "type": "memory_store", 
                "description": "Store analysis in memory for retrieval",
                "config": {
                    "metadata": {
                        "document_type": "analyzed_business_report",
                        "analysis_version": "1.0",
                        "prompt_used": {
                            "from_step": "analyze_document",
                            "field": "prompt_id"
                        }
                    }
                },
                "inputs": {
                    "content": {
                        "from_step": "analyze_document",
                        "field": "executive_summary"
                    },
                    "metadata": {
                        "full_analysis": {
                            "from_step": "analyze_document", 
                            "field": "structured_response"
                        },
                        "source_document": "$document_path"
                    }
                },
                "outputs": ["memory_id"]
            },
            {
                "id": "get_user_question",
                "type": "input",
                "description": "Get user's question about the analyzed document",
                "config": {
                    "prompt": "What would you like to know about this document analysis?"
                },
                "outputs": ["question"]
            },
            {
                "id": "retrieve_relevant_context",
                "type": "memory_retrieve",
                "description": "Find relevant context for the question",
                "config": {
                    "limit": 5,
                    "threshold": 0.75
                },
                "inputs": {
                    "query": {
                        "from_step": "get_user_question",
                        "field": "question"
                    }
                },
                "outputs": ["memories"]
            },
            {
                "id": "answer_question",
                "type": "llm_chat",
                "description": "Answer question using contextual Q&A prompt",
                "prompt_ref": {
                    "prompt_id": "contextual_qa_assistant", 
                    "version": "1.0.0",
                    "context_variables": {
                        "assistant_personality": "a business analysis expert",
                        "expertise_level": "Senior business analyst with strategic insight",
                        "communication_style": "Professional, clear, and strategic"
                    }
                },
                "config": {
                    "provider": "gemini",
                    "model": "gemini-1.5-pro"
                },
                "inputs": {
                    "context_information": {
                        "from_step": "analyze_document",
                        "field": "structured_response" 
                    },
                    "user_question": {
                        "from_step": "get_user_question",
                        "field": "question"
                    },
                    "answer_format": "structured response with clear reasoning"
                },
                "outputs": ["response", "prompt_id", "execution_time_ms"]
            },
            {
                "id": "output_final_result",
                "type": "output",
                "description": "Return comprehensive results with prompt tracking",
                "config": {
                    "format": "json"
                },
                "inputs": {
                    "document_analysis": {
                        "from_step": "analyze_document",
                        "field": "structured_response"
                    },
                    "user_question": {
                        "from_step": "get_user_question", 
                        "field": "question"
                    },
                    "answer": {
                        "from_step": "answer_question",
                        "field": "response"
                    },
                    "performance_metrics": {
                        "analysis_time_ms": {
                            "from_step": "analyze_document",
                            "field": "execution_time_ms"
                        },
                        "qa_time_ms": {
                            "from_step": "answer_question", 
                            "field": "execution_time_ms"
                        }
                    },
                    "prompts_used": {
                        "analysis_prompt": {
                            "from_step": "analyze_document",
                            "field": "prompt_id"
                        },
                        "qa_prompt": {
                            "from_step": "answer_question",
                            "field": "prompt_id"
                        }
                    }
                }
            }
        ]
    }
    
    return workflow_config


def demonstrate_workflow_execution():
    """Demonstrate executing the workflow with managed prompts"""
    
    print("\n🤖 Creating and executing workflow with managed prompts...")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Load the workflow
    workflow_config = create_workflow_with_managed_prompts()
    workflow = orchestrator.load_workflow_from_dict(workflow_config)
    
    print("\n📋 Workflow Configuration:")
    print(f"  Name: {workflow.name}")
    print(f"  Steps: {len(workflow.steps)}")
    print(f"  Managed Prompt Steps: {sum(1 for step in workflow.steps if step.prompt_ref)}")
    
    # Show the workflow structure
    print("\n🔄 Workflow Steps:")
    for i, step in enumerate(workflow.steps, 1):
        print(f"  {i}. {step.id} ({step.type})")
        if step.prompt_ref:
            print(f"     📝 Prompt: {step.prompt_ref.prompt_id} v{step.prompt_ref.version}")
    
    print("\n✅ Workflow ready for execution!")
    print("\n💡 To execute this workflow:")
    print("   result = orchestrator.execute_workflow(workflow, {")
    print("       'document_path': '/path/to/your/document.pdf'")
    print("   })")


def demonstrate_prompt_evaluation():
    """Show prompt evaluation capabilities"""
    
    print("\n📊 Demonstrating Prompt Evaluation...")
    
    prompt_service = create_prompt_service("auto")
    
    # Simulate some usage data
    print("📝 Simulating prompt usage for evaluation...")
    
    for i in range(10):
        success = i < 8  # 80% success rate
        execution_time = 1200 + (i * 50)  # Varying response times
        
        prompt_service.log_execution(
            prompt_id="smart_document_analyzer",
            prompt_version="1.0.0",
            execution_context={
                "document_type": "business report",
                "domain": "business intelligence",
                "analysis_style": "strategic and actionable"
            },
            rendered_messages=[
                {"role": "system", "content": "You are an expert..."},
                {"role": "user", "content": f"Analyze document {i}..."}
            ],
            llm_provider="gemini",
            llm_model="gemini-1.5-pro",
            llm_response=f"Analysis result for document {i}" if success else None,
            execution_time_ms=execution_time,
            token_usage={"input": 450 + i*10, "output": 320 + i*15, "total": 770 + i*25},
            success=success,
            error_message=None if success else f"Processing error for document {i}"
        )
    
    # Generate evaluation report
    from prompt.evaluation_service import EvaluationService
    
    eval_service = EvaluationService(prompt_service.storage)
    report = eval_service.generate_evaluation_report("smart_document_analyzer", days_back=1)
    
    print("\n📈 Evaluation Report:")
    print(f"  Prompt: {report['prompt_id']}")
    print(f"  Total Executions: {report['metrics']['total_executions']}")
    print(f"  Success Rate: {report['metrics']['success_rate']:.1%}")
    print(f"  Avg Response Time: {report['metrics']['average_response_time_ms']:.0f}ms")
    print(f"  Performance Grade: {report['grade']}")
    print(f"  Recommendations:")
    for rec in report['recommendations'][:2]:
        print(f"    • {rec}")


def demonstrate_a_b_testing():
    """Show A/B testing capabilities"""
    
    print("\n🔬 Demonstrating A/B Testing...")
    
    prompt_service = create_prompt_service("auto")
    
    # Create version 1.1.0 of the document analyzer
    improved_prompt = prompt_service.create_prompt_version(
        "smart_document_analyzer",
        "1.1.0",
        changelog="Improved analysis depth and added industry-specific insights"
    )
    
    if improved_prompt:
        print(f"✅ Created improved version: v{improved_prompt.version.version}")
        
        # Simulate some usage of the new version
        for i in range(5):
            prompt_service.log_execution(
                prompt_id="smart_document_analyzer",
                prompt_version="1.1.0",
                execution_context={"document_type": "business report"},
                rendered_messages=[{"role": "user", "content": f"Analyze v1.1 document {i}"}],
                llm_provider="gemini",
                llm_response=f"Improved analysis for document {i}",
                execution_time_ms=1100 + i*30,  # Slightly faster
                success=True
            )
        
        # Compare versions
        from prompt.evaluation_service import EvaluationService
        eval_service = EvaluationService(prompt_service.storage)
        
        comparison = eval_service.compare_prompt_versions(
            "smart_document_analyzer", "1.0.0",
            "smart_document_analyzer", "1.1.0"
        )
        
        print(f"\n🆚 A/B Test Results:")
        print(f"  Version 1.0.0 vs 1.1.0")
        print(f"  Winner: {comparison.winner}")
        print(f"  Confidence: {comparison.confidence_level:.1%}")
        print(f"  Sample Size: {comparison.sample_size}")


def main():
    """Run the complete prompt management demonstration"""
    
    print("🧱 AI Lego Bricks - Prompt Management Integration Demo")
    print("=" * 60)
    
    # Setup sample prompts
    prompt_service = setup_sample_prompts()
    
    # Create and show workflow
    demonstrate_workflow_execution()
    
    # Show evaluation capabilities
    demonstrate_prompt_evaluation()
    
    # Demonstrate A/B testing
    demonstrate_a_b_testing()
    
    # Show service statistics
    print("\n📊 System Statistics:")
    stats = prompt_service.get_stats()
    print(f"  Storage Backend: {stats['storage_backend']}")
    print(f"  Active Prompts: {stats['active_prompts_count']}")
    print(f"  Cache Size: {stats['cache_size']}")
    
    print("\n🎯 Benefits of Managed Prompts:")
    print("  • Version control and rollback capability")
    print("  • A/B testing for prompt optimization")
    print("  • Performance tracking and analytics")
    print("  • Team collaboration on prompt engineering")
    print("  • Training data collection for model improvement")
    print("  • Separation of prompts from code logic")
    
    print("\n✅ Prompt Management Demo Complete!")
    print("\n💡 Next Steps:")
    print("  1. Create your own prompts using the prompt service")
    print("  2. Reference them in your workflows with prompt_ref")
    print("  3. Monitor performance and iterate based on data")
    print("  4. Use A/B testing to validate improvements")


if __name__ == "__main__":
    main()