"""
Builder tools and helpers for creating concept-based evaluations
"""

import json
import csv
from typing import List, Dict, Any, Optional

from .concept_eval_models import (
    ConceptEvalDefinition, COMMON_CONCEPT_CHECKS
)


class EvaluationBuilder:
    """Builder class for creating evaluations step by step"""
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initialize builder with evaluation name
        
        Args:
            name: Name of the evaluation
            description: Optional description
        """
        self.name = name
        self.description = description
        self.prompt_template = ""
        self.test_cases = []
        self.concept_checks = []
        self.metadata = {}
    
    def with_prompt_template(self, template: str) -> 'EvaluationBuilder':
        """Set the prompt template"""
        self.prompt_template = template
        return self
    
    def with_metadata(self, **kwargs) -> 'EvaluationBuilder':
        """Add metadata fields"""
        self.metadata.update(kwargs)
        return self
    
    def add_concept_check(self, 
                         check_type: str,
                         description: str,
                         concept: str,
                         expected_value: Optional[str] = None,
                         weight: float = 1.0,
                         check_id: Optional[str] = None) -> 'EvaluationBuilder':
        """
        Add a reusable concept check definition
        
        Args:
            check_type: "must_contain", "must_not_contain", or "binary_decision"
            description: Human-readable description
            concept: The concept or criteria to check
            expected_value: For binary decisions, the expected answer
            weight: Weight for scoring (default 1.0)
            check_id: Optional ID for referencing this check
        """
        check_def = {
            "type": check_type,
            "description": description,
            "concept": concept,
            "weight": weight
        }
        
        if expected_value:
            check_def["expected_value"] = expected_value
        
        if check_id:
            check_def["id"] = check_id
        
        self.concept_checks.append(check_def)
        return self
    
    def add_common_check(self, check_name: str, check_id: Optional[str] = None) -> 'EvaluationBuilder':
        """
        Add a predefined common concept check
        
        Args:
            check_name: Name of the common check (see COMMON_CONCEPT_CHECKS)
            check_id: Optional ID for referencing this check
        """
        if check_name not in COMMON_CONCEPT_CHECKS:
            raise ValueError(f"Unknown common check: {check_name}. Available: {list(COMMON_CONCEPT_CHECKS.keys())}")
        
        common_check = COMMON_CONCEPT_CHECKS[check_name]
        check_def = {
            "type": common_check.type.value,
            "description": common_check.description,
            "concept": common_check.concept,
            "weight": common_check.weight
        }
        
        if common_check.expected_value:
            check_def["expected_value"] = common_check.expected_value
        
        if check_id:
            check_def["id"] = check_id
        else:
            check_def["id"] = check_name
        
        self.concept_checks.append(check_def)
        return self
    
    def add_test_case(self,
                     context: Dict[str, Any],
                     concept_check_refs: List[str],
                     name: Optional[str] = None,
                     expected_output: Optional[str] = None,
                     **metadata) -> 'EvaluationBuilder':
        """
        Add a test case using concept check references
        
        Args:
            context: Variables for template rendering
            concept_check_refs: List of concept check IDs to apply
            name: Optional name for this test case
            expected_output: Optional expected output for reference
            **metadata: Additional metadata for the test case
        """
        test_case = {
            "context": context,
            "concept_checks": concept_check_refs
        }
        
        if name:
            test_case["name"] = name
        if expected_output:
            test_case["expected_output"] = expected_output
        if metadata:
            test_case["metadata"] = metadata
        
        self.test_cases.append(test_case)
        return self
    
    def add_test_case_with_inline_checks(self,
                                       context: Dict[str, Any],
                                       inline_checks: List[Dict[str, Any]],
                                       name: Optional[str] = None,
                                       expected_output: Optional[str] = None,
                                       **metadata) -> 'EvaluationBuilder':
        """
        Add a test case with inline concept check definitions
        
        Args:
            context: Variables for template rendering
            inline_checks: List of inline concept check definitions
            name: Optional name for this test case
            expected_output: Optional expected output for reference
            **metadata: Additional metadata for the test case
        """
        test_case = {
            "context": context,
            "concept_checks": inline_checks
        }
        
        if name:
            test_case["name"] = name
        if expected_output:
            test_case["expected_output"] = expected_output
        if metadata:
            test_case["metadata"] = metadata
        
        self.test_cases.append(test_case)
        return self
    
    def build(self, eval_id: Optional[str] = None) -> ConceptEvalDefinition:
        """
        Build the final evaluation definition
        
        Args:
            eval_id: Optional evaluation ID (generated if not provided)
            
        Returns:
            ConceptEvalDefinition object
        """
        if not self.prompt_template:
            raise ValueError("Prompt template is required")
        
        if not self.test_cases:
            raise ValueError("At least one test case is required")
        
        if not eval_id:
            eval_id = self.name.lower().replace(" ", "_")
        
        return ConceptEvalDefinition(
            eval_id=eval_id,
            name=self.name,
            description=self.description,
            prompt_template=self.prompt_template,
            test_cases=self.test_cases,
            concept_checks=self.concept_checks,
            metadata=self.metadata
        )


class QuickEvaluationBuilder:
    """Simplified builder for common evaluation patterns"""
    
    @staticmethod
    def create_accuracy_evaluation(name: str,
                                 prompt_template: str,
                                 test_cases: List[Dict[str, Any]],
                                 eval_id: Optional[str] = None) -> ConceptEvalDefinition:
        """
        Create an evaluation focused on accuracy and correctness
        
        Args:
            name: Evaluation name
            prompt_template: Jinja2 template
            test_cases: List of test cases with context and expected_output
            eval_id: Optional evaluation ID
            
        Returns:
            ConceptEvalDefinition
        """
        builder = EvaluationBuilder(name, "Accuracy-focused evaluation")
        builder.with_prompt_template(prompt_template)
        
        # Add common accuracy checks
        builder.add_common_check("contains_factual_info", "factual")
        builder.add_common_check("answers_question", "answers")
        builder.add_common_check("no_hallucination", "no_hallucination")
        
        # Add test cases
        for i, test_case in enumerate(test_cases):
            builder.add_test_case(
                context=test_case["context"],
                concept_check_refs=["factual", "answers", "no_hallucination"],
                name=f"test_case_{i+1}",
                expected_output=test_case.get("expected_output")
            )
        
        return builder.build(eval_id)
    
    @staticmethod
    def create_style_evaluation(name: str,
                              prompt_template: str,
                              test_cases: List[Dict[str, Any]],
                              eval_id: Optional[str] = None) -> ConceptEvalDefinition:
        """
        Create an evaluation focused on style and tone
        
        Args:
            name: Evaluation name
            prompt_template: Jinja2 template
            test_cases: List of test cases with context
            eval_id: Optional evaluation ID
            
        Returns:
            ConceptEvalDefinition
        """
        builder = EvaluationBuilder(name, "Style and tone evaluation")
        builder.with_prompt_template(prompt_template)
        
        # Add style checks
        builder.add_common_check("professional_tone", "professional")
        builder.add_common_check("correct_format", "format")
        builder.add_common_check("avoids_opinions", "objective")
        
        # Add test cases
        for i, test_case in enumerate(test_cases):
            builder.add_test_case(
                context=test_case["context"],
                concept_check_refs=["professional", "format", "objective"],
                name=f"style_test_{i+1}"
            )
        
        return builder.build(eval_id)
    
    @staticmethod
    def create_binary_decision_evaluation(name: str,
                                        prompt_template: str,
                                        decision_cases: List[Dict[str, Any]],
                                        eval_id: Optional[str] = None) -> ConceptEvalDefinition:
        """
        Create an evaluation for binary decision prompts
        
        Args:
            name: Evaluation name
            prompt_template: Jinja2 template
            decision_cases: List of cases with context and expected_decision
            eval_id: Optional evaluation ID
            
        Returns:
            ConceptEvalDefinition
        """
        builder = EvaluationBuilder(name, "Binary decision evaluation")
        builder.with_prompt_template(prompt_template)
        
        # Add test cases with custom binary checks
        for i, case in enumerate(decision_cases):
            expected = case.get("expected_decision", "yes")
            
            builder.add_test_case_with_inline_checks(
                context=case["context"],
                inline_checks=[{
                    "type": "binary_decision",
                    "description": f"Correct decision: {expected}",
                    "concept": "Is the decision correct?",
                    "expected_value": expected,
                    "weight": 1.0
                }],
                name=f"decision_case_{i+1}"
            )
        
        return builder.build(eval_id)


class EvaluationImporter:
    """Tools for importing evaluations from various formats"""
    
    @staticmethod
    def from_csv(csv_path: str,
                name: str,
                prompt_template: str,
                context_columns: List[str],
                concept_checks: List[Dict[str, Any]],
                expected_output_column: Optional[str] = None) -> ConceptEvalDefinition:
        """
        Import test cases from CSV file
        
        Args:
            csv_path: Path to CSV file
            name: Evaluation name
            prompt_template: Jinja2 template
            context_columns: Column names to use as template context
            concept_checks: List of concept check definitions
            expected_output_column: Optional column for expected outputs
            
        Returns:
            ConceptEvalDefinition
        """
        builder = EvaluationBuilder(name, f"Imported from {csv_path}")
        builder.with_prompt_template(prompt_template)
        
        # Add concept checks
        for i, check in enumerate(concept_checks):
            check_id = check.get("id", f"check_{i}")
            builder.add_concept_check(
                check_type=check["type"],
                description=check["description"],
                concept=check["concept"],
                expected_value=check.get("expected_value"),
                weight=check.get("weight", 1.0),
                check_id=check_id
            )
        
        # Read CSV and create test cases
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            check_refs = [check.get("id", f"check_{i}") for i, check in enumerate(concept_checks)]
            
            for i, row in enumerate(reader):
                context = {col: row[col] for col in context_columns if col in row}
                expected_output = row.get(expected_output_column) if expected_output_column else None
                
                builder.add_test_case(
                    context=context,
                    concept_check_refs=check_refs,
                    name=f"csv_case_{i+1}",
                    expected_output=expected_output
                )
        
        return builder.build()
    
    @staticmethod
    def from_json(json_path: str) -> ConceptEvalDefinition:
        """
        Import evaluation definition from JSON file
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ConceptEvalDefinition
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ConceptEvalDefinition(**data)
    
    @staticmethod
    def export_to_json(evaluation: ConceptEvalDefinition, output_path: str):
        """
        Export evaluation definition to JSON file
        
        Args:
            evaluation: Evaluation to export
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation.model_dump(), f, indent=2, default=str)


class EvaluationTemplates:
    """Pre-built evaluation templates for common use cases"""
    
    @staticmethod
    def get_summarization_eval(test_cases: List[Dict[str, Any]]) -> ConceptEvalDefinition:
        """Template for evaluating summarization prompts"""
        return QuickEvaluationBuilder.create_accuracy_evaluation(
            name="Document Summarization Evaluation",
            prompt_template="Summarize the following {{document_type}}: {{content}}",
            test_cases=test_cases,
            eval_id="summarization_eval"
        )
    
    @staticmethod
    def get_classification_eval(categories: List[str], test_cases: List[Dict[str, Any]]) -> ConceptEvalDefinition:
        """Template for evaluating classification prompts"""
        builder = EvaluationBuilder("Classification Evaluation", "Evaluates classification accuracy")
        
        template = f"Classify the following text into one of these categories: {', '.join(categories)}.\n\nText: {{text}}"
        builder.with_prompt_template(template)
        
        # Add classification-specific checks
        builder.add_concept_check(
            check_type="must_contain",
            description="Contains valid category",
            concept=f"one of the valid categories: {', '.join(categories)}",
            check_id="valid_category"
        )
        
        builder.add_common_check("answers_question", "answers")
        builder.add_common_check("no_hallucination", "no_hallucination")
        
        # Add test cases
        for i, test_case in enumerate(test_cases):
            builder.add_test_case(
                context=test_case["context"],
                concept_check_refs=["valid_category", "answers", "no_hallucination"],
                name=f"classification_test_{i+1}",
                expected_output=test_case.get("expected_category")
            )
        
        return builder.build("classification_eval")
    
    @staticmethod
    def get_qa_eval(test_cases: List[Dict[str, Any]]) -> ConceptEvalDefinition:
        """Template for evaluating Q&A prompts"""
        return QuickEvaluationBuilder.create_accuracy_evaluation(
            name="Question Answering Evaluation",
            prompt_template="Answer the following question based on the provided context.\n\nContext: {{context}}\n\nQuestion: {{question}}",
            test_cases=test_cases,
            eval_id="qa_eval"
        )