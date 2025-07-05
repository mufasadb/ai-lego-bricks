#!/usr/bin/env python3
"""
Comprehensive example of the concept-based prompt evaluation system

This example demonstrates:
1. Creating evaluations using the builder pattern
2. Running evaluations and analyzing results
3. Different types of concept checks
4. Importing/exporting evaluations
5. Integration with storage backends

Usage:
    python concept_evaluation_example.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prompt.eval_builder import EvaluationBuilder, QuickEvaluationBuilder, EvaluationTemplates
from prompt.concept_eval_storage import create_concept_eval_storage
from prompt.concept_evaluation_service import ConceptEvaluationService
from prompt.concept_eval_models import ConceptCheckType


def create_sample_document_summarizer_eval():
    """Create a sample evaluation for document summarization"""
    print("\n📝 Creating Document Summarizer Evaluation...")
    
    # Use the builder pattern
    builder = EvaluationBuilder(
        name="Document Summarizer Evaluation",
        description="Evaluates document summarization quality and style"
    )
    
    # Set the prompt template
    builder.with_prompt_template(
        "You are an expert document summarizer. "
        "Summarize the following {{document_type}} in {{style}} style:\n\n{{content}}"
    )
    
    # Add metadata
    builder.with_metadata(
        author="AI Evaluation Team",
        tags=["summarization", "documents", "quality"],
        version="1.0.0"
    )
    
    # Add concept checks
    builder.add_concept_check(
        check_type="must_contain",
        description="Summary contains key facts",
        concept="specific facts, data, or concrete information from the source",
        check_id="key_facts"
    )
    
    builder.add_concept_check(
        check_type="must_not_contain",
        description="Summary avoids personal opinions",
        concept="personal opinions, subjective judgments, or unsupported claims",
        check_id="objective"
    )
    
    builder.add_concept_check(
        check_type="must_contain",
        description="Summary is concise",
        concept="concise, well-structured, and appropriately length summary",
        check_id="concise"
    )
    
    # Add test cases
    test_content = """
    The quarterly earnings report shows revenue increased 15% to $2.3 billion, 
    driven by strong demand in the cloud services division. Net profit rose 8% 
    to $450 million. The company plans to expand operations in three new markets 
    next year and hire 200 additional engineers.
    """
    
    builder.add_test_case(
        context={
            "document_type": "quarterly earnings report",
            "style": "executive",
            "content": test_content.strip()
        },
        concept_check_refs=["key_facts", "objective", "concise"],
        name="earnings_report_executive",
        expected_output="Executive summary highlighting 15% revenue growth to $2.3B, 8% profit increase to $450M, expansion plans, and hiring goals."
    )
    
    builder.add_test_case(
        context={
            "document_type": "quarterly earnings report", 
            "style": "technical",
            "content": test_content.strip()
        },
        concept_check_refs=["key_facts", "objective", "concise"],
        name="earnings_report_technical"
    )
    
    return builder.build("doc_summarizer_eval")


def create_sample_classification_eval():
    """Create a sample evaluation for text classification"""
    print("\n🏷️ Creating Text Classification Evaluation...")
    
    # Use the quick builder for classification
    test_cases = [
        {
            "context": {"text": "I'm really excited about this new product launch!"},
            "expected_category": "positive"
        },
        {
            "context": {"text": "The service was disappointing and overpriced."},
            "expected_category": "negative"
        },
        {
            "context": {"text": "The product works as described, nothing special."},
            "expected_category": "neutral"
        }
    ]
    
    return EvaluationTemplates.get_classification_eval(
        categories=["positive", "negative", "neutral"],
        test_cases=test_cases
    )


def create_sample_binary_decision_eval():
    """Create a sample evaluation for binary decisions"""
    print("\n✅ Creating Binary Decision Evaluation...")
    
    decision_cases = [
        {
            "context": {"email": "URGENT: Click here to claim your prize now!"},
            "expected_decision": "yes"  # This should be classified as spam
        },
        {
            "context": {"email": "Meeting scheduled for tomorrow at 2 PM in conference room A."},
            "expected_decision": "no"   # This is not spam
        },
        {
            "context": {"email": "Free money! No strings attached! Act now!"},
            "expected_decision": "yes"  # This should be classified as spam
        }
    ]
    
    return QuickEvaluationBuilder.create_binary_decision_evaluation(
        name="Spam Detection Evaluation",
        prompt_template="Is this email spam? Answer only 'yes' or 'no'.\n\nEmail: {{email}}",
        decision_cases=decision_cases,
        eval_id="spam_detection_eval"
    )


def demonstrate_evaluation_execution():
    """Demonstrate running evaluations and analyzing results"""
    print("\n🚀 Demonstrating Evaluation Execution...")
    
    # Create storage backend (file-based for this example)
    storage = create_concept_eval_storage("file", storage_path="./example_evaluations")
    
    # Create evaluation service
    eval_service = ConceptEvaluationService(
        storage_backend=storage,
        default_llm_provider="gemini",  # Use Gemini for generation
        default_judge_model="gemini"    # Use Gemini as judge
    )
    
    # Create and save evaluation definitions
    doc_eval = create_sample_document_summarizer_eval()
    classification_eval = create_sample_classification_eval()
    binary_eval = create_sample_binary_decision_eval()
    
    # Save evaluations
    print("\n💾 Saving evaluation definitions...")
    storage.save_evaluation_definition(doc_eval)
    storage.save_evaluation_definition(classification_eval)
    storage.save_evaluation_definition(binary_eval)
    
    # List saved evaluations
    print("\n📋 Saved evaluations:")
    saved_evals = storage.list_evaluation_definitions()
    for eval_def in saved_evals:
        print(f"  - {eval_def.name} (ID: {eval_def.eval_id})")
    
    # Run the document summarization evaluation
    print("\n🔄 Running Document Summarizer Evaluation...")
    try:
        results = eval_service.run_evaluation_by_id("doc_summarizer_eval", llm_provider="gemini")
        
        if results:
            print(f"\n📊 Evaluation Results:")
            print(f"Overall Score: {results.overall_score:.1%}")
            print(f"Test Cases: {results.passed_test_cases}/{results.total_test_cases} passed")
            print(f"Execution Time: {results.total_execution_time_ms:.0f}ms")
            print(f"Grade: {results.summary}")
            
            print(f"\n📈 Concept Breakdown:")
            for check_type, breakdown in results.concept_breakdown.items():
                print(f"  {check_type}: {breakdown['pass_rate']:.1%} pass rate")
            
            print(f"\n💡 Recommendations:")
            for rec in results.recommendations:
                print(f"  - {rec}")
            
            # Show detailed test case results
            print(f"\n🔍 Detailed Test Case Results:")
            for i, test_result in enumerate(results.test_case_results):
                print(f"  Test Case {i+1} ({test_result.test_case_name}): {'✅ PASS' if test_result.overall_passed else '❌ FAIL'}")
                print(f"    Score: {test_result.overall_score:.1%}")
                print(f"    Output: {test_result.llm_output[:100]}...")
                
                for check in test_result.check_results:
                    status = "✅" if check.passed else "❌"
                    print(f"    {status} {check.description}: {check.judge_reasoning[:80]}...")
        
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        print("Note: This example requires LLM services to be configured.")
    
    return eval_service


def demonstrate_comparison_and_history():
    """Demonstrate evaluation comparison and history tracking"""
    print("\n📈 Demonstrating Evaluation History and Comparison...")
    
    storage = create_concept_eval_storage("file", storage_path="./example_evaluations")
    eval_service = ConceptEvaluationService(storage)
    
    # Get evaluation history
    history = eval_service.get_evaluation_history("doc_summarizer_eval", limit=5)
    print(f"Found {len(history)} previous evaluation runs")
    
    for i, result in enumerate(history):
        print(f"  Run {i+1}: {result.overall_score:.1%} score at {result.started_at}")
    
    # Compare evaluations if we have multiple runs
    if len(history) >= 2:
        comparison = eval_service.compare_evaluations("doc_summarizer_eval")
        if comparison:
            print(f"\n📊 Comparison with Previous Run:")
            print(f"  Current Score: {comparison['current_score']:.1%}")
            print(f"  Previous Score: {comparison['previous_score']:.1%}")
            print(f"  Change: {comparison['score_change']:+.1%}")
            print(f"  Improved: {'✅' if comparison['score_improved'] else '❌'}")


def demonstrate_advanced_features():
    """Demonstrate advanced features and patterns"""
    print("\n🔧 Demonstrating Advanced Features...")
    
    # Create a complex evaluation with multiple check types
    builder = EvaluationBuilder(
        name="Advanced Multi-Modal Evaluation",
        description="Demonstrates all concept check types"
    )
    
    builder.with_prompt_template(
        "Analyze this {{content_type}} and provide a {{analysis_type}} analysis: {{content}}"
    )
    
    # Add various types of checks
    builder.add_concept_check(
        check_type="must_contain",
        description="Contains specific analysis",
        concept="detailed analysis with specific examples",
        weight=2.0,  # Higher weight
        check_id="detailed_analysis"
    )
    
    builder.add_concept_check(
        check_type="must_not_contain",
        description="Avoids speculation",
        concept="speculation, assumptions, or unverified claims",
        weight=1.5,
        check_id="no_speculation"
    )
    
    # Add test case with inline checks for specific scenarios
    builder.add_test_case_with_inline_checks(
        context={
            "content_type": "financial report",
            "analysis_type": "risk assessment", 
            "content": "The company reported volatile earnings over the past quarter."
        },
        inline_checks=[
            {
                "type": "binary_decision",
                "description": "Identifies risks correctly",
                "concept": "Does the analysis correctly identify financial risks?",
                "expected_value": "yes",
                "weight": 2.0
            }
        ],
        name="risk_assessment_case"
    )
    
    # Also add a case using referenced checks
    builder.add_test_case(
        context={
            "content_type": "market report",
            "analysis_type": "trend analysis",
            "content": "Market shows upward trend in technology sector."
        },
        concept_check_refs=["detailed_analysis", "no_speculation"],
        name="trend_analysis_case"
    )
    
    advanced_eval = builder.build("advanced_eval")
    
    print(f"Created advanced evaluation with {len(advanced_eval.test_cases)} test cases")
    print(f"Total concept checks: {len(advanced_eval.concept_checks)}")
    
    # Save it
    storage = create_concept_eval_storage("file", storage_path="./example_evaluations")
    storage.save_evaluation_definition(advanced_eval)
    print("Advanced evaluation saved!")


def main():
    """Main example function"""
    print("🧪 Concept-Based Prompt Evaluation System Example")
    print("=" * 50)
    
    # Create example evaluations
    create_sample_document_summarizer_eval()
    create_sample_classification_eval()
    create_sample_binary_decision_eval()
    
    # Demonstrate execution
    eval_service = demonstrate_evaluation_execution()
    
    # Demonstrate history and comparison
    demonstrate_comparison_and_history()
    
    # Demonstrate advanced features
    demonstrate_advanced_features()
    
    print("\n✨ Example completed!")
    print("\nNext steps:")
    print("1. Configure your LLM providers (Gemini, Anthropic, Ollama)")
    print("2. Set up Supabase for production storage (optional)")
    print("3. Create your own evaluations using the builder patterns")
    print("4. Integrate with agent orchestration workflows")
    print("5. Set up continuous evaluation for prompt quality monitoring")
    
    # Show file locations
    print(f"\n📁 Example files created in:")
    print(f"  - ./example_evaluations/ (evaluation definitions and results)")
    
    return eval_service


if __name__ == "__main__":
    # Set up basic error handling
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        print("\nNote: This example requires:")
        print("- LLM services to be configured (Gemini API key, etc.)")
        print("- Required dependencies to be installed")
        print("- Project structure to be set up correctly")