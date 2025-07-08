"""
Evaluation service for prompt performance metrics and A/B testing
"""

import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .prompt_models import PromptEvaluation, PromptComparisonResult
from .prompt_storage import PromptStorageBackend


class EvaluationService:
    """Service for evaluating prompt performance and conducting A/B tests"""
    
    def __init__(self, storage_backend: PromptStorageBackend):
        self.storage = storage_backend
    
    def calculate_prompt_metrics(self, prompt_id: str, version: Optional[str] = None,
                                days_back: int = 7) -> Optional[PromptEvaluation]:
        """
        Calculate performance metrics for a prompt
        
        Args:
            prompt_id: ID of the prompt to evaluate
            version: Optional specific version
            days_back: Number of days to look back for data
            
        Returns:
            PromptEvaluation object or None if no data
        """
        # Get execution data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        executions = self.storage.get_executions(prompt_id, version, limit=10000)
        
        # Filter by time period
        period_executions = [
            e for e in executions 
            if start_time <= e.created_at <= end_time
        ]
        
        if not period_executions:
            return None
        
        # Calculate metrics
        total_executions = len(period_executions)
        successful_executions = [e for e in period_executions if e.success]
        failed_executions = [e for e in period_executions if not e.success]
        
        success_rate = len(successful_executions) / total_executions if total_executions > 0 else 0
        error_rate = len(failed_executions) / total_executions if total_executions > 0 else 0
        
        # Response time metrics
        execution_times = [e.execution_time_ms for e in period_executions if e.execution_time_ms is not None]
        avg_response_time = statistics.mean(execution_times) if execution_times else 0
        
        # Token usage metrics
        token_usage_data = []
        for execution in period_executions:
            if execution.token_usage:
                token_usage_data.append(execution.token_usage)
        
        avg_tokens = {}
        if token_usage_data:
            # Calculate average for each token type
            token_types = set()
            for usage in token_usage_data:
                token_types.update(usage.keys())
            
            for token_type in token_types:
                values = [usage.get(token_type, 0) for usage in token_usage_data]
                avg_tokens[token_type] = statistics.mean(values)
        
        # Common errors
        error_messages = [e.error_message for e in failed_executions if e.error_message]
        error_counts = defaultdict(int)
        for error in error_messages:
            error_counts[error] += 1
        
        common_errors = sorted(error_counts.keys(), key=lambda x: error_counts[x], reverse=True)[:5]
        
        # Calculate performance score (simple heuristic)
        performance_score = self._calculate_performance_score(
            success_rate, avg_response_time, avg_tokens.get('total', 0)
        )
        
        return PromptEvaluation(
            prompt_id=prompt_id,
            prompt_version=version or "latest",
            evaluation_period_start=start_time,
            evaluation_period_end=end_time,
            total_executions=total_executions,
            success_rate=success_rate,
            average_response_time_ms=avg_response_time,
            average_tokens_used=avg_tokens,
            error_rate=error_rate,
            common_errors=common_errors,
            performance_score=performance_score
        )
    
    def _calculate_performance_score(self, success_rate: float, 
                                   avg_response_time: float, 
                                   avg_tokens: float) -> float:
        """
        Calculate a composite performance score
        
        Args:
            success_rate: Success rate (0-1)
            avg_response_time: Average response time in ms
            avg_tokens: Average token usage
            
        Returns:
            Performance score (0-100)
        """
        # Normalize response time (assuming 5000ms is very slow)
        time_score = max(0, 1 - (avg_response_time / 5000))
        
        # Normalize token usage (assuming 4000 tokens is high)
        token_score = max(0, 1 - (avg_tokens / 4000))
        
        # Weighted combination
        score = (success_rate * 0.5) + (time_score * 0.3) + (token_score * 0.2)
        
        return score * 100
    
    def compare_prompt_versions(self, prompt_a_id: str, prompt_a_version: str,
                               prompt_b_id: str, prompt_b_version: str,
                               days_back: int = 7) -> PromptComparisonResult:
        """
        Compare performance between two prompt versions
        
        Args:
            prompt_a_id: ID of first prompt
            prompt_a_version: Version of first prompt
            prompt_b_id: ID of second prompt
            prompt_b_version: Version of second prompt
            days_back: Number of days to look back for data
            
        Returns:
            Comparison result
        """
        # Get metrics for both prompts
        metrics_a = self.calculate_prompt_metrics(prompt_a_id, prompt_a_version, days_back)
        metrics_b = self.calculate_prompt_metrics(prompt_b_id, prompt_b_version, days_back)
        
        if not metrics_a or not metrics_b:
            return PromptComparisonResult(
                test_id=f"{prompt_a_id}_{prompt_a_version}_vs_{prompt_b_id}_{prompt_b_version}",
                prompt_a_id=prompt_a_id,
                prompt_a_version=prompt_a_version,
                prompt_b_id=prompt_b_id,
                prompt_b_version=prompt_b_version,
                sample_size=0,
                winner=None,
                metrics_comparison={"error": "Insufficient data for comparison"}
            )
        
        # Calculate differences
        success_rate_diff = metrics_b.success_rate - metrics_a.success_rate
        response_time_diff = metrics_a.average_response_time_ms - metrics_b.average_response_time_ms  # Lower is better
        performance_diff = metrics_b.performance_score - metrics_a.performance_score
        
        # Determine winner (simple heuristic)
        winner = None
        confidence = 0.0
        
        # Count positive indicators for each prompt
        a_wins = 0
        b_wins = 0
        
        if success_rate_diff > 0.05:  # B has >5% better success rate
            b_wins += 1
        elif success_rate_diff < -0.05:  # A has >5% better success rate
            a_wins += 1
        
        if response_time_diff > 500:  # B is >500ms faster
            b_wins += 1
        elif response_time_diff < -500:  # A is >500ms faster
            a_wins += 1
        
        if performance_diff > 5:  # B has >5 point better performance score
            b_wins += 1
        elif performance_diff < -5:  # A has >5 point better performance score
            a_wins += 1
        
        if b_wins > a_wins:
            winner = "prompt_b"
            confidence = min(0.95, 0.5 + (b_wins - a_wins) * 0.15)
        elif a_wins > b_wins:
            winner = "prompt_a"
            confidence = min(0.95, 0.5 + (a_wins - b_wins) * 0.15)
        else:
            winner = "no_significant_difference"
            confidence = 0.5
        
        metrics_comparison = {
            "prompt_a_metrics": {
                "success_rate": metrics_a.success_rate,
                "avg_response_time_ms": metrics_a.average_response_time_ms,
                "performance_score": metrics_a.performance_score,
                "total_executions": metrics_a.total_executions
            },
            "prompt_b_metrics": {
                "success_rate": metrics_b.success_rate,
                "avg_response_time_ms": metrics_b.average_response_time_ms,
                "performance_score": metrics_b.performance_score,
                "total_executions": metrics_b.total_executions
            },
            "differences": {
                "success_rate_diff": success_rate_diff,
                "response_time_diff": response_time_diff,
                "performance_score_diff": performance_diff
            }
        }
        
        return PromptComparisonResult(
            test_id=f"{prompt_a_id}_{prompt_a_version}_vs_{prompt_b_id}_{prompt_b_version}",
            prompt_a_id=prompt_a_id,
            prompt_a_version=prompt_a_version,
            prompt_b_id=prompt_b_id,
            prompt_b_version=prompt_b_version,
            sample_size=metrics_a.total_executions + metrics_b.total_executions,
            winner=winner,
            confidence_level=confidence,
            metrics_comparison=metrics_comparison,
            test_duration_hours=days_back * 24
        )
    
    def get_top_performing_prompts(self, limit: int = 10, 
                                  days_back: int = 7) -> List[Tuple[str, str, float]]:
        """
        Get top performing prompts by performance score
        
        Args:
            limit: Maximum number of prompts to return
            days_back: Number of days to look back for data
            
        Returns:
            List of (prompt_id, version, performance_score) tuples
        """
        # This would need to be implemented based on your storage backend
        # For now, returning empty list as placeholder
        return []
    
    def identify_performance_regressions(self, threshold: float = 10.0,
                                       days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Identify prompts with performance regressions
        
        Args:
            threshold: Performance score threshold for regression detection
            days_back: Number of days to look back for comparison
            
        Returns:
            List of regression reports
        """
        # This would compare current performance vs historical baseline
        # Placeholder implementation
        return []
    
    def generate_evaluation_report(self, prompt_id: str, version: Optional[str] = None,
                                 days_back: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for a prompt
        
        Args:
            prompt_id: ID of the prompt to evaluate
            version: Optional specific version
            days_back: Number of days to look back for data
            
        Returns:
            Comprehensive evaluation report
        """
        metrics = self.calculate_prompt_metrics(prompt_id, version, days_back)
        
        if not metrics:
            return {
                "prompt_id": prompt_id,
                "version": version or "latest",
                "error": "No execution data found",
                "recommendations": ["Ensure prompt is being used in production"]
            }
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if metrics.success_rate < 0.95:
            recommendations.append(f"Success rate is {metrics.success_rate:.1%}. Consider reviewing error patterns.")
        
        if metrics.average_response_time_ms > 3000:
            recommendations.append(f"Average response time is {metrics.average_response_time_ms:.0f}ms. Consider optimizing prompt length.")
        
        if metrics.error_rate > 0.1:
            recommendations.append(f"Error rate is {metrics.error_rate:.1%}. Review common errors: {', '.join(metrics.common_errors[:3])}")
        
        # Token efficiency
        avg_tokens = metrics.average_tokens_used.get('total', 0)
        if avg_tokens > 3000:
            recommendations.append(f"High token usage ({avg_tokens:.0f} avg). Consider making prompt more concise.")
        
        if not recommendations:
            recommendations.append("Prompt is performing well across all metrics.")
        
        return {
            "prompt_id": prompt_id,
            "version": version or "latest",
            "evaluation_period": {
                "start": metrics.evaluation_period_start.isoformat(),
                "end": metrics.evaluation_period_end.isoformat(),
                "days": days_back
            },
            "metrics": {
                "total_executions": metrics.total_executions,
                "success_rate": metrics.success_rate,
                "error_rate": metrics.error_rate,
                "average_response_time_ms": metrics.average_response_time_ms,
                "average_tokens_used": metrics.average_tokens_used,
                "performance_score": metrics.performance_score
            },
            "common_errors": metrics.common_errors,
            "recommendations": recommendations,
            "grade": self._get_performance_grade(metrics.performance_score)
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"