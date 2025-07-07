#!/usr/bin/env python3
"""
Example script to run coordinator evaluations using the standard evaluation framework

This demonstrates how to use the existing prompt evaluation system to test:
1. Routing accuracy - does the coordinator choose the right expert?
2. Expert response quality - do the experts provide good answers in their domains?
"""

import json
import sys
from concept_evaluation_service import ConceptEvaluationService
from concept_eval_storage import create_concept_eval_storage


def run_routing_evaluation():
    """Test routing accuracy"""
    print("🎯 Running Coordinator Routing Evaluation...")
    print("=" * 50)
    
    # Load routing evaluation definition
    with open('coordinator_routing_evaluation.json', 'r') as f:
        routing_eval = json.load(f)
    
    # Create evaluation service
    storage = create_concept_eval_storage('auto')
    service = ConceptEvaluationService(storage)
    
    try:
        # Run the evaluation
        results = service.run_evaluation(routing_eval)
        
        print(f"📊 Overall Routing Accuracy: {results.overall_score:.1%}")
        print(f"   Passed: {results.passed_test_cases}/{results.total_test_cases} test cases")
        
        # Show concept breakdown
        if hasattr(results, 'concept_breakdown'):
            print(f"\n📈 Concept Breakdown:")
            for concept, stats in results.concept_breakdown.items():
                print(f"   {concept}: {stats.get('pass_rate', 0):.1%}")
        
        # Show recommendations
        if hasattr(results, 'recommendations') and results.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in results.recommendations:
                print(f"   • {rec}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running routing evaluation: {e}")
        return None


def run_expert_quality_evaluations():
    """Test expert response quality"""
    print("\n🔬 Running Expert Quality Evaluations...")
    print("=" * 50)
    
    # Load expert evaluation definitions
    with open('coordinator_expert_evaluation.json', 'r') as f:
        expert_evals = json.load(f)
    
    # Create evaluation service
    storage = create_concept_eval_storage('auto')
    service = ConceptEvaluationService(storage)
    
    expert_results = {}
    
    for expert_name, expert_eval in expert_evals['expert_evaluations'].items():
        print(f"\n📝 Testing {expert_name.replace('_', ' ').title()}...")
        
        try:
            results = service.run_evaluation(expert_eval)
            expert_results[expert_name] = results
            
            print(f"   Score: {results.overall_score:.1%}")
            print(f"   Passed: {results.passed_test_cases}/{results.total_test_cases} test cases")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            expert_results[expert_name] = None
    
    return expert_results


def print_summary(routing_results, expert_results):
    """Print a summary of all evaluation results"""
    print("\n📋 Evaluation Summary")
    print("=" * 50)
    
    if routing_results:
        print(f"🎯 Routing Accuracy: {routing_results.overall_score:.1%}")
    else:
        print("🎯 Routing Accuracy: Failed to evaluate")
    
    print(f"\n🔬 Expert Quality Scores:")
    for expert_name, results in expert_results.items():
        expert_display = expert_name.replace('_', ' ').title()
        if results:
            print(f"   {expert_display}: {results.overall_score:.1%}")
        else:
            print(f"   {expert_display}: Failed to evaluate")
    
    # Calculate overall coordinator system score
    valid_scores = []
    if routing_results:
        valid_scores.append(routing_results.overall_score)
    
    for results in expert_results.values():
        if results:
            valid_scores.append(results.overall_score)
    
    if valid_scores:
        overall_score = sum(valid_scores) / len(valid_scores)
        print(f"\n🎉 Overall Coordinator System Score: {overall_score:.1%}")
        
        if overall_score >= 0.85:
            print("   ✅ Excellent performance!")
        elif overall_score >= 0.75:
            print("   ✅ Good performance")
        elif overall_score >= 0.65:
            print("   ⚠️  Needs improvement")
        else:
            print("   ❌ Significant issues - review prompts")


def main():
    """Main evaluation runner"""
    print("🤖 AI Coordinator Evaluation Suite")
    print("Using standard prompt evaluation framework")
    print("=" * 60)
    
    # Run evaluations
    routing_results = run_routing_evaluation()
    expert_results = run_expert_quality_evaluations()
    
    # Print summary
    print_summary(routing_results, expert_results)
    
    print(f"\n✨ Evaluation complete!")
    print("\n💡 To improve performance:")
    print("   1. Review failed test cases in detailed results")
    print("   2. Refine expert system prompts based on feedback")
    print("   3. Add more specific routing criteria for edge cases")
    print("   4. Test with additional edge cases and scenarios")


if __name__ == "__main__":
    main()