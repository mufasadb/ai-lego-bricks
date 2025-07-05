"""
Main service for executing concept-based prompt evaluations
"""

import time
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import jinja2

from .concept_eval_models import (
    PromptEvaluation, EvalTestCase, ConceptCheck, 
    TestCaseResult, CheckResult, EvalExecutionResult,
    ConceptEvalDefinition
)
from .concept_eval_storage import ConceptEvalStorageBackend
from .concept_judge import ConceptJudgeService, create_concept_judge

# Import LLM services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm'))
from llm_factory import LLMClientFactory
from llm_types import LLMProvider


class ConceptEvaluationService:
    """Main service for running concept-based prompt evaluations"""
    
    def __init__(self, 
                 storage_backend: ConceptEvalStorageBackend,
                 default_llm_provider: str = "gemini",
                 default_judge_model: str = "gemini"):
        """
        Initialize the evaluation service
        
        Args:
            storage_backend: Storage backend for evaluations and results
            default_llm_provider: Default LLM for generating outputs
            default_judge_model: Default model for judging outputs
        """
        self.storage = storage_backend
        self.default_llm_provider = default_llm_provider
        self.default_judge_model = default_judge_model
        
        # Create judge service
        self.judge_service = create_concept_judge(default_judge_model)
        
        # LLM provider mapping
        self.provider_map = {
            "gemini": LLMProvider.GEMINI,
            "anthropic": LLMProvider.ANTHROPIC,
            "ollama": LLMProvider.OLLAMA
        }
    
    def run_evaluation(self, 
                      evaluation: PromptEvaluation,
                      llm_provider: Optional[str] = None,
                      save_results: bool = True) -> EvalExecutionResult:
        """
        Execute a complete prompt evaluation
        
        Args:
            evaluation: The evaluation definition to run
            llm_provider: LLM provider for generating outputs (overrides default)
            save_results: Whether to save results to storage
            
        Returns:
            Complete evaluation results
        """
        start_time = datetime.now()
        provider = llm_provider or self.default_llm_provider
        
        print(f"🚀 Starting evaluation: {evaluation.name}")
        print(f"📊 Test cases: {len(evaluation.test_cases)}")
        print(f"🤖 LLM Provider: {provider}")
        print(f"⚖️ Judge Model: {evaluation.judge_model}")
        
        # Initialize LLM client for generating outputs
        llm_client = self._create_llm_client(provider)
        
        # Create judge service for this evaluation
        judge_service = create_concept_judge(
            judge_model=evaluation.judge_model,
            temperature=evaluation.judge_temperature,
            max_tokens=evaluation.judge_max_tokens
        )
        
        # Run each test case
        test_case_results = []
        for i, test_case in enumerate(evaluation.test_cases):
            print(f"🧪 Running test case {i+1}/{len(evaluation.test_cases)}")
            
            result = self._run_test_case(
                evaluation.prompt_template,
                test_case,
                llm_client,
                judge_service
            )
            test_case_results.append(result)
        
        # Calculate summary metrics
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        summary_metrics = self._calculate_summary_metrics(test_case_results)
        concept_breakdown = self._calculate_concept_breakdown(test_case_results)
        recommendations = self._generate_recommendations(test_case_results, summary_metrics)
        
        # Create execution result
        execution_result = EvalExecutionResult(
            evaluation_name=evaluation.name,
            evaluation_version=evaluation.version,
            test_case_results=test_case_results,
            overall_score=summary_metrics['overall_score'],
            total_test_cases=len(test_case_results),
            passed_test_cases=summary_metrics['passed_test_cases'],
            failed_test_cases=summary_metrics['failed_test_cases'],
            concept_breakdown=concept_breakdown,
            started_at=start_time,
            completed_at=end_time,
            total_execution_time_ms=execution_time,
            judge_model_used=evaluation.judge_model,
            llm_provider_used=provider,
            summary=self._generate_summary(summary_metrics),
            recommendations=recommendations
        )
        
        # Save results if requested
        if save_results:
            success = self.storage.save_execution_result(execution_result)
            if success:
                print(f"✅ Results saved to storage")
            else:
                print(f"⚠️ Failed to save results to storage")
        
        print(f"🏁 Evaluation completed in {execution_time:.0f}ms")
        print(f"📈 Overall score: {summary_metrics['overall_score']:.1%}")
        
        return execution_result
    
    def run_evaluation_by_id(self, eval_id: str, **kwargs) -> Optional[EvalExecutionResult]:
        """
        Load and run an evaluation by ID
        
        Args:
            eval_id: ID of the evaluation to run
            **kwargs: Additional arguments for run_evaluation
            
        Returns:
            Execution results or None if evaluation not found
        """
        # Load evaluation definition
        eval_def = self.storage.get_evaluation_definition(eval_id)
        if not eval_def:
            print(f"❌ Evaluation '{eval_id}' not found")
            return None
        
        # Convert to full evaluation object
        evaluation = eval_def.to_prompt_evaluation()
        
        # Run evaluation
        return self.run_evaluation(evaluation, **kwargs)
    
    def _run_test_case(self, 
                      prompt_template: str,
                      test_case: EvalTestCase,
                      llm_client,
                      judge_service: ConceptJudgeService) -> TestCaseResult:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Render prompt template with context
            rendered_prompt = self._render_template(prompt_template, test_case.context)
            
            # Generate output using LLM
            llm_output = llm_client.generate(rendered_prompt)
            
            # Evaluate concept checks
            check_results = judge_service.evaluate_multiple_checks(llm_output, test_case.concept_checks)
            
            # Calculate overall results
            overall_passed = all(result.passed for result in check_results)
            overall_score = self._calculate_test_case_score(check_results, test_case.concept_checks)
            
            execution_time = (time.time() - start_time) * 1000
            
            return TestCaseResult(
                test_case_name=test_case.name,
                context=test_case.context,
                llm_output=llm_output,
                check_results=check_results,
                overall_passed=overall_passed,
                overall_score=overall_score,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            print(f"⚠️ Test case failed: {str(e)}")
            
            return TestCaseResult(
                test_case_name=test_case.name,
                context=test_case.context,
                llm_output="",
                check_results=[],
                overall_passed=False,
                overall_score=0.0,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Render Jinja2 template with context"""
        try:
            jinja_template = jinja2.Template(template)
            return jinja_template.render(**context)
        except Exception as e:
            raise ValueError(f"Failed to render template: {str(e)}")
    
    def _create_llm_client(self, provider: str):
        """Create LLM client for the specified provider"""
        provider_enum = self.provider_map.get(provider, LLMProvider.GEMINI)
        return LLMClientFactory.create_text_client(
            provider=provider_enum,
            temperature=0.7,  # Default for generation
            max_tokens=2000
        )
    
    def _calculate_test_case_score(self, check_results: List[CheckResult], concept_checks: List[ConceptCheck]) -> float:
        """Calculate weighted score for a test case"""
        if not check_results:
            return 0.0
        
        total_weight = sum(check.weight for check in concept_checks)
        if total_weight == 0:
            return 0.0
        
        weighted_score = 0.0
        for i, result in enumerate(check_results):
            weight = concept_checks[i].weight if i < len(concept_checks) else 1.0
            score = result.confidence if result.passed else 0.0
            weighted_score += score * weight
        
        return weighted_score / total_weight
    
    def _calculate_summary_metrics(self, test_case_results: List[TestCaseResult]) -> Dict[str, Any]:
        """Calculate overall summary metrics"""
        if not test_case_results:
            return {
                'overall_score': 0.0,
                'passed_test_cases': 0,
                'failed_test_cases': 0,
                'average_execution_time': 0.0
            }
        
        passed_cases = sum(1 for result in test_case_results if result.overall_passed)
        failed_cases = len(test_case_results) - passed_cases
        
        scores = [result.overall_score for result in test_case_results if result.overall_score is not None]
        overall_score = statistics.mean(scores) if scores else 0.0
        
        execution_times = [result.execution_time_ms for result in test_case_results if result.execution_time_ms is not None]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
        
        return {
            'overall_score': overall_score,
            'passed_test_cases': passed_cases,
            'failed_test_cases': failed_cases,
            'average_execution_time': avg_execution_time
        }
    
    def _calculate_concept_breakdown(self, test_case_results: List[TestCaseResult]) -> Dict[str, Dict[str, Any]]:
        """Calculate pass rates and metrics by concept type"""
        concept_stats = defaultdict(lambda: {
            'total_checks': 0,
            'passed_checks': 0,
            'average_confidence': 0.0,
            'concepts': defaultdict(lambda: {'passed': 0, 'total': 0})
        })
        
        for result in test_case_results:
            for check_result in result.check_results:
                check_type = check_result.check_type.value
                concept = check_result.concept
                
                # Update type-level stats
                concept_stats[check_type]['total_checks'] += 1
                if check_result.passed:
                    concept_stats[check_type]['passed_checks'] += 1
                
                # Update concept-level stats
                concept_stats[check_type]['concepts'][concept]['total'] += 1
                if check_result.passed:
                    concept_stats[check_type]['concepts'][concept]['passed'] += 1
        
        # Calculate rates and averages
        breakdown = {}
        for check_type, stats in concept_stats.items():
            total = stats['total_checks']
            passed = stats['passed_checks']
            pass_rate = passed / total if total > 0 else 0.0
            
            concept_details = {}
            for concept, concept_stats in stats['concepts'].items():
                concept_total = concept_stats['total']
                concept_passed = concept_stats['passed']
                concept_rate = concept_passed / concept_total if concept_total > 0 else 0.0
                concept_details[concept] = {
                    'pass_rate': concept_rate,
                    'passed': concept_passed,
                    'total': concept_total
                }
            
            breakdown[check_type] = {
                'pass_rate': pass_rate,
                'passed_checks': passed,
                'total_checks': total,
                'concepts': dict(concept_details)
            }
        
        return breakdown
    
    def _generate_recommendations(self, test_case_results: List[TestCaseResult], summary_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        overall_score = summary_metrics['overall_score']
        failed_cases = summary_metrics['failed_test_cases']
        total_cases = len(test_case_results)
        
        # Overall performance recommendations
        if overall_score < 0.7:
            recommendations.append(f"Overall score is {overall_score:.1%}. Consider revising the prompt template for better performance.")
        
        if failed_cases > total_cases * 0.3:
            recommendations.append(f"{failed_cases}/{total_cases} test cases failed. Review failed cases for common patterns.")
        
        # Analyze common failure patterns
        failure_patterns = defaultdict(int)
        for result in test_case_results:
            if not result.overall_passed:
                for check in result.check_results:
                    if not check.passed:
                        failure_patterns[check.concept] += 1
        
        if failure_patterns:
            most_common_failure = max(failure_patterns.items(), key=lambda x: x[1])
            recommendations.append(f"Most common failure: '{most_common_failure[0]}' failed {most_common_failure[1]} times.")
        
        # Performance recommendations
        avg_time = summary_metrics.get('average_execution_time', 0)
        if avg_time > 5000:  # 5 seconds
            recommendations.append(f"Average execution time is {avg_time:.0f}ms. Consider optimizing prompt length.")
        
        if not recommendations:
            recommendations.append("✨ Evaluation performing well! All metrics are within acceptable ranges.")
        
        return recommendations
    
    def _generate_summary(self, summary_metrics: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        score = summary_metrics['overall_score']
        passed = summary_metrics['passed_test_cases']
        failed = summary_metrics['failed_test_cases']
        total = passed + failed
        
        grade = self._get_grade(score)
        
        return f"Evaluation completed with {score:.1%} overall score (Grade: {grade}). {passed}/{total} test cases passed."
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
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
    
    def get_evaluation_history(self, eval_id: str, limit: int = 10) -> List[EvalExecutionResult]:
        """Get execution history for an evaluation"""
        return self.storage.get_execution_results(eval_id, limit)
    
    def compare_evaluations(self, eval_id: str, limit: int = 2) -> Optional[Dict[str, Any]]:
        """Compare recent evaluation runs"""
        results = self.get_evaluation_history(eval_id, limit)
        
        if len(results) < 2:
            return None
        
        current = results[0]
        previous = results[1]
        
        score_change = current.overall_score - previous.overall_score
        time_change = current.total_execution_time_ms - previous.total_execution_time_ms
        
        return {
            'current_score': current.overall_score,
            'previous_score': previous.overall_score,
            'score_change': score_change,
            'score_improved': score_change > 0,
            'execution_time_change_ms': time_change,
            'time_improved': time_change < 0,
            'current_date': current.started_at,
            'previous_date': previous.started_at
        }