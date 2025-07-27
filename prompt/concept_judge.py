"""
LLM-as-judge service for concept-based evaluation
"""

import time
from typing import List

from .concept_eval_models import ConceptCheck, ConceptCheckType, CheckResult

# Import LLM factory for judge model access
from llm.llm_factory import LLMClientFactory
from llm.llm_types import LLMProvider


class ConceptJudgeService:
    """Service for using LLMs to evaluate outputs against concept criteria"""

    def __init__(
        self,
        judge_model: str = "gemini",
        temperature: float = 0.1,
        max_tokens: int = 500,
    ):
        """
        Initialize the judge service

        Args:
            judge_model: Model to use as judge (gemini, anthropic, ollama)
            temperature: Temperature for judge responses (low for consistency)
            max_tokens: Max tokens for judge responses
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Map model names to providers
        self.provider_map = {
            "gemini": LLMProvider.GEMINI,
            "anthropic": LLMProvider.ANTHROPIC,
            "ollama": LLMProvider.OLLAMA,
        }

        # Initialize LLM client
        provider = self.provider_map.get(judge_model, LLMProvider.GEMINI)
        self.llm_client = LLMClientFactory.create_text_client(
            provider=provider, temperature=temperature, max_tokens=max_tokens
        )

    def evaluate_concept_check(
        self, output: str, concept_check: ConceptCheck
    ) -> CheckResult:
        """
        Evaluate a single concept check against an output

        Args:
            output: The text output to evaluate
            concept_check: The concept check to perform

        Returns:
            CheckResult with pass/fail and reasoning
        """
        start_time = time.time()

        try:
            # Get the appropriate judge prompt
            judge_prompt = self._get_judge_prompt(output, concept_check)

            # Call the LLM judge
            response = self.llm_client.generate(judge_prompt)

            # Parse the response
            passed, reasoning, confidence = self._parse_judge_response(
                response, concept_check.type
            )

            execution_time = (time.time() - start_time) * 1000

            return CheckResult(
                check_type=concept_check.type,
                concept=concept_check.concept,
                description=concept_check.description,
                passed=passed,
                judge_reasoning=reasoning,
                confidence=confidence,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return CheckResult(
                check_type=concept_check.type,
                concept=concept_check.concept,
                description=concept_check.description,
                passed=False,
                judge_reasoning=f"Evaluation failed: {str(e)}",
                confidence=0.0,
                execution_time_ms=execution_time,
                error_message=str(e),
            )

    def evaluate_multiple_checks(
        self, output: str, concept_checks: List[ConceptCheck]
    ) -> List[CheckResult]:
        """
        Evaluate multiple concept checks against an output

        Args:
            output: The text output to evaluate
            concept_checks: List of concept checks to perform

        Returns:
            List of CheckResult objects
        """
        results = []
        for check in concept_checks:
            result = self.evaluate_concept_check(output, check)
            results.append(result)

        return results

    def _get_judge_prompt(self, output: str, concept_check: ConceptCheck) -> str:
        """Generate the appropriate judge prompt based on check type"""

        if concept_check.type == ConceptCheckType.MUST_CONTAIN:
            return self._get_must_contain_prompt(output, concept_check)
        elif concept_check.type == ConceptCheckType.MUST_NOT_CONTAIN:
            return self._get_must_not_contain_prompt(output, concept_check)
        elif concept_check.type == ConceptCheckType.BINARY_DECISION:
            return self._get_binary_decision_prompt(output, concept_check)
        else:
            raise ValueError(f"Unknown concept check type: {concept_check.type}")

    def _get_must_contain_prompt(self, output: str, concept_check: ConceptCheck) -> str:
        """Generate prompt for 'must contain' checks"""
        return f"""You are an expert evaluator. Your task is to determine if a given text contains a specific concept.

EVALUATION CRITERIA:
- Concept to check for: {concept_check.concept}
- Description: {concept_check.description}

TEXT TO EVALUATE:
```
{output}
```

INSTRUCTIONS:
1. Carefully read the text
2. Determine if the text contains the specified concept
3. Think step-by-step about your reasoning
4. Provide your final answer

Please respond in this exact format:

REASONING: [Explain your step-by-step analysis of whether the concept is present]

ANSWER: [YES or NO]

CONFIDENCE: [A number from 0.0 to 1.0 indicating how confident you are in your assessment]"""

    def _get_must_not_contain_prompt(
        self, output: str, concept_check: ConceptCheck
    ) -> str:
        """Generate prompt for 'must not contain' checks"""
        return f"""You are an expert evaluator. Your task is to determine if a given text avoids a specific concept.

EVALUATION CRITERIA:
- Concept to avoid: {concept_check.concept}
- Description: {concept_check.description}

TEXT TO EVALUATE:
```
{output}
```

INSTRUCTIONS:
1. Carefully read the text
2. Determine if the text successfully avoids the specified concept
3. Think step-by-step about your reasoning
4. Provide your final answer

Please respond in this exact format:

REASONING: [Explain your step-by-step analysis of whether the concept is avoided]

ANSWER: [YES if the concept is successfully avoided, NO if the concept is present]

CONFIDENCE: [A number from 0.0 to 1.0 indicating how confident you are in your assessment]"""

    def _get_binary_decision_prompt(
        self, output: str, concept_check: ConceptCheck
    ) -> str:
        """Generate prompt for binary decision checks"""
        expected = concept_check.expected_value or "yes"
        return f"""You are an expert evaluator. Your task is to determine if a response correctly answers a yes/no question.

EVALUATION CRITERIA:
- Question/Criteria: {concept_check.concept}
- Expected answer: {expected.upper()}
- Description: {concept_check.description}

RESPONSE TO EVALUATE:
```
{output}
```

INSTRUCTIONS:
1. Carefully read the response
2. Determine what the actual answer is (yes or no)
3. Compare with the expected answer: {expected.upper()}
4. Think step-by-step about your reasoning
5. Provide your final assessment

Please respond in this exact format:

REASONING: [Explain what answer the response gives and whether it matches the expected answer]

ANSWER: [YES if the response matches the expected answer, NO if it doesn't]

CONFIDENCE: [A number from 0.0 to 1.0 indicating how confident you are in your assessment]"""

    def _parse_judge_response(
        self, response: str, check_type: ConceptCheckType
    ) -> tuple[bool, str, float]:
        """
        Parse the judge's response to extract pass/fail, reasoning, and confidence

        Returns:
            (passed: bool, reasoning: str, confidence: float)
        """
        try:
            # Extract sections from the response
            reasoning = ""
            answer = ""
            confidence = 0.5

            lines = response.strip().split("\n")
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith("REASONING:"):
                    current_section = "reasoning"
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("ANSWER:"):
                    current_section = "answer"
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    current_section = "confidence"
                    conf_text = line.replace("CONFIDENCE:", "").strip()
                    try:
                        confidence = float(conf_text)
                    except ValueError:
                        confidence = 0.5
                elif current_section == "reasoning" and line:
                    reasoning += " " + line
                elif current_section == "answer" and line:
                    answer += " " + line

            # Determine if the check passed
            answer_upper = answer.upper()
            passed = "YES" in answer_upper and "NO" not in answer_upper.replace(
                "YES", ""
            )

            # Clean up reasoning
            if not reasoning:
                reasoning = "No reasoning provided"

            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

            return passed, reasoning, confidence

        except Exception:
            # Fallback parsing
            response_upper = response.upper()
            passed = "YES" in response_upper and (
                "NO" not in response_upper
                or response_upper.find("YES") < response_upper.find("NO")
            )
            reasoning = (
                f"Parsed from response: {response[:200]}..."
                if len(response) > 200
                else response
            )
            confidence = 0.5

            return passed, reasoning, confidence


class BatchConceptJudge:
    """Optimized judge for batch evaluation of multiple outputs"""

    def __init__(self, judge_service: ConceptJudgeService):
        self.judge_service = judge_service

    def evaluate_batch(
        self, outputs: List[str], concept_checks: List[ConceptCheck]
    ) -> List[List[CheckResult]]:
        """
        Evaluate multiple outputs against the same set of concept checks

        Args:
            outputs: List of text outputs to evaluate
            concept_checks: List of concept checks to apply to each output

        Returns:
            List of CheckResult lists (one per output)
        """
        batch_results = []

        for output in outputs:
            output_results = self.judge_service.evaluate_multiple_checks(
                output, concept_checks
            )
            batch_results.append(output_results)

        return batch_results

    def evaluate_single_concept_batch(
        self, outputs: List[str], concept_check: ConceptCheck
    ) -> List[CheckResult]:
        """
        Evaluate multiple outputs against a single concept check

        Args:
            outputs: List of text outputs to evaluate
            concept_check: Single concept check to apply

        Returns:
            List of CheckResult objects
        """
        results = []

        for output in outputs:
            result = self.judge_service.evaluate_concept_check(output, concept_check)
            results.append(result)

        return results


def create_concept_judge(judge_model: str = "gemini", **kwargs) -> ConceptJudgeService:
    """
    Factory function to create a concept judge service

    Args:
        judge_model: Model to use as judge
        **kwargs: Additional configuration options

    Returns:
        ConceptJudgeService instance
    """
    return ConceptJudgeService(
        judge_model=judge_model,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 500),
    )
