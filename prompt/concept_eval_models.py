"""
Pydantic models for concept-based prompt evaluation
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class ConceptCheckType(str, Enum):
    """Types of concept checks available"""
    MUST_CONTAIN = "must_contain"
    MUST_NOT_CONTAIN = "must_not_contain"
    BINARY_DECISION = "binary_decision"


class ConceptCheck(BaseModel):
    """Definition of a single concept check"""
    type: ConceptCheckType
    description: str = Field(..., description="Human-readable description of what to check")
    concept: str = Field(..., description="The concept, question, or criteria to evaluate")
    expected_value: Optional[str] = Field(None, description="For binary decisions, the expected answer (yes/no)")
    weight: float = Field(1.0, description="Weight of this check in overall scoring")
    
    
class EvalTestCase(BaseModel):
    """A single test case within an evaluation"""
    name: Optional[str] = Field(None, description="Optional name for this test case")
    context: Dict[str, Any] = Field(..., description="Variables to fill the prompt template")
    concept_checks: List[ConceptCheck] = Field(..., description="Checks to perform on the output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    expected_output: Optional[str] = Field(None, description="Optional expected output for reference")


class PromptEvaluation(BaseModel):
    """Complete definition of a prompt evaluation"""
    name: str = Field(..., description="Name of this evaluation")
    description: Optional[str] = Field(None, description="Description of what this evaluation tests")
    prompt_template: str = Field(..., description="Jinja2 template for the prompt")
    test_cases: List[EvalTestCase] = Field(..., description="Test cases to run")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    
    # Configuration
    judge_model: str = Field("gemini", description="LLM model to use as judge")
    judge_temperature: float = Field(0.1, description="Temperature for judge model")
    judge_max_tokens: Optional[int] = Field(None, description="Max tokens for judge responses")


# Results Models

class CheckResult(BaseModel):
    """Result of a single concept check"""
    check_type: ConceptCheckType
    concept: str
    description: str
    passed: bool
    judge_reasoning: str = Field(..., description="LLM judge's reasoning")
    confidence: float = Field(..., description="Confidence score from 0-1")
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class TestCaseResult(BaseModel):
    """Result of running a single test case"""
    test_case_name: Optional[str]
    context: Dict[str, Any]
    llm_output: str = Field(..., description="Output generated from template + context")
    check_results: List[CheckResult] = Field(..., description="Results of all concept checks")
    overall_passed: bool = Field(..., description="True if all checks passed")
    overall_score: float = Field(..., description="Weighted score from 0-1")
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None


class EvalExecutionResult(BaseModel):
    """Complete results of running an evaluation"""
    evaluation_name: str
    evaluation_version: str
    test_case_results: List[TestCaseResult] = Field(..., description="Results for each test case")
    
    # Summary metrics
    overall_score: float = Field(..., description="Overall score from 0-1")
    total_test_cases: int
    passed_test_cases: int
    failed_test_cases: int
    
    # Concept breakdown
    concept_breakdown: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Pass rate and details for each concept type"
    )
    
    # Execution metadata
    started_at: datetime
    completed_at: datetime
    total_execution_time_ms: float
    judge_model_used: str
    llm_provider_used: str
    
    # Summary
    summary: str = Field(..., description="Human-readable summary of results")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


class ConceptEvalDefinition(BaseModel):
    """Lightweight definition for storing/loading evaluations"""
    eval_id: str = Field(..., description="Unique identifier for this evaluation")
    name: str
    description: Optional[str] = None
    prompt_template: str
    test_cases: List[Dict[str, Any]] = Field(..., description="Simplified test case definitions")
    concept_checks: List[Dict[str, Any]] = Field(..., description="Reusable concept check definitions")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_prompt_evaluation(self) -> PromptEvaluation:
        """Convert to full PromptEvaluation object"""
        # Build test cases with concept checks
        full_test_cases = []
        for test_case_def in self.test_cases:
            checks = []
            for check_ref in test_case_def.get('concept_checks', []):
                if isinstance(check_ref, str):
                    # Reference to a predefined check
                    check_def = next((c for c in self.concept_checks if c.get('id') == check_ref), None)
                    if check_def:
                        checks.append(ConceptCheck(**{k: v for k, v in check_def.items() if k != 'id'}))
                else:
                    # Inline check definition
                    checks.append(ConceptCheck(**check_ref))
            
            full_test_cases.append(EvalTestCase(
                name=test_case_def.get('name'),
                context=test_case_def['context'],
                concept_checks=checks,
                metadata=test_case_def.get('metadata', {}),
                expected_output=test_case_def.get('expected_output')
            ))
        
        return PromptEvaluation(
            name=self.name,
            description=self.description,
            prompt_template=self.prompt_template,
            test_cases=full_test_cases,
            author=self.metadata.get('author'),
            tags=self.metadata.get('tags', []),
            version=self.metadata.get('version', '1.0.0'),
            judge_model=self.metadata.get('judge_model', 'gemini'),
            judge_temperature=self.metadata.get('judge_temperature', 0.1),
            judge_max_tokens=self.metadata.get('judge_max_tokens')
        )


# Predefined concept check templates for common scenarios
COMMON_CONCEPT_CHECKS = {
    "contains_factual_info": ConceptCheck(
        type=ConceptCheckType.MUST_CONTAIN,
        description="Output contains factual information",
        concept="specific facts, data, or concrete information"
    ),
    "avoids_opinions": ConceptCheck(
        type=ConceptCheckType.MUST_NOT_CONTAIN,
        description="Output avoids personal opinions",
        concept="subjective opinions, personal beliefs, or unsupported claims"
    ),
    "professional_tone": ConceptCheck(
        type=ConceptCheckType.MUST_CONTAIN,
        description="Output maintains professional tone",
        concept="professional, formal, and appropriate language"
    ),
    "answers_question": ConceptCheck(
        type=ConceptCheckType.BINARY_DECISION,
        description="Output directly answers the question",
        concept="Does the output answer the question asked?",
        expected_value="yes"
    ),
    "correct_format": ConceptCheck(
        type=ConceptCheckType.MUST_CONTAIN,
        description="Output follows correct format",
        concept="proper structure, formatting, and organization"
    ),
    "no_hallucination": ConceptCheck(
        type=ConceptCheckType.MUST_NOT_CONTAIN,
        description="Output avoids hallucination",
        concept="made-up facts, fictional information, or unsupported claims"
    )
}