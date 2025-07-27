"""
Storage backends for concept-based prompt evaluation
"""

import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import uuid

from .concept_eval_models import ConceptEvalDefinition, EvalExecutionResult


class ConceptEvalStorageBackend(ABC):
    """Abstract base class for concept evaluation storage backends"""

    @abstractmethod
    def save_evaluation_definition(self, eval_def: ConceptEvalDefinition) -> bool:
        """Save an evaluation definition to storage"""
        pass

    @abstractmethod
    def get_evaluation_definition(
        self, eval_id: str
    ) -> Optional[ConceptEvalDefinition]:
        """Retrieve an evaluation definition by ID"""
        pass

    @abstractmethod
    def list_evaluation_definitions(
        self, tags: Optional[List[str]] = None
    ) -> List[ConceptEvalDefinition]:
        """List evaluation definitions with optional tag filters"""
        pass

    @abstractmethod
    def delete_evaluation_definition(self, eval_id: str) -> bool:
        """Delete an evaluation definition"""
        pass

    @abstractmethod
    def save_execution_result(self, result: EvalExecutionResult) -> bool:
        """Save execution result to storage"""
        pass

    @abstractmethod
    def get_execution_results(
        self, eval_id: str, limit: int = 50
    ) -> List[EvalExecutionResult]:
        """Get execution results for an evaluation"""
        pass

    @abstractmethod
    def get_latest_execution_result(
        self, eval_id: str
    ) -> Optional[EvalExecutionResult]:
        """Get the most recent execution result for an evaluation"""
        pass


class FileConceptEvalStorage(ConceptEvalStorageBackend):
    """File-based concept evaluation storage backend"""

    def __init__(self, storage_path: str = "./concept_evaluations"):
        self.storage_path = Path(storage_path)
        self.definitions_path = self.storage_path / "definitions"
        self.results_path = self.storage_path / "results"

        # Create directories if they don't exist
        self.definitions_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

    def save_evaluation_definition(self, eval_def: ConceptEvalDefinition) -> bool:
        """Save evaluation definition to JSON file"""
        try:
            file_path = self.definitions_path / f"{eval_def.eval_id}.json"
            with open(file_path, "w") as f:
                json.dump(eval_def.model_dump(), f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving evaluation definition: {e}")
            return False

    def get_evaluation_definition(
        self, eval_id: str
    ) -> Optional[ConceptEvalDefinition]:
        """Load evaluation definition from JSON file"""
        try:
            file_path = self.definitions_path / f"{eval_id}.json"
            if not file_path.exists():
                return None

            with open(file_path, "r") as f:
                data = json.load(f)
            return ConceptEvalDefinition(**data)
        except Exception as e:
            print(f"Error loading evaluation definition: {e}")
            return None

    def list_evaluation_definitions(
        self, tags: Optional[List[str]] = None
    ) -> List[ConceptEvalDefinition]:
        """List all evaluation definitions, optionally filtered by tags"""
        definitions = []

        for file_path in self.definitions_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                eval_def = ConceptEvalDefinition(**data)

                # Filter by tags if specified
                if tags:
                    eval_tags = eval_def.metadata.get("tags", [])
                    if not any(tag in eval_tags for tag in tags):
                        continue

                definitions.append(eval_def)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return sorted(definitions, key=lambda x: x.name)

    def delete_evaluation_definition(self, eval_id: str) -> bool:
        """Delete evaluation definition file"""
        try:
            file_path = self.definitions_path / f"{eval_id}.json"
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting evaluation definition: {e}")
            return False

    def save_execution_result(self, result: EvalExecutionResult) -> bool:
        """Save execution result to JSON file"""
        try:
            # Create directory for this evaluation if it doesn't exist
            eval_results_path = self.results_path / result.evaluation_name
            eval_results_path.mkdir(exist_ok=True)

            # Use timestamp for unique filename
            timestamp = result.started_at.strftime("%Y%m%d_%H%M%S")
            file_path = eval_results_path / f"result_{timestamp}.json"

            with open(file_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving execution result: {e}")
            return False

    def get_execution_results(
        self, eval_id: str, limit: int = 50
    ) -> List[EvalExecutionResult]:
        """Get execution results for an evaluation"""
        results = []

        # Look for results directory
        eval_results_path = self.results_path / eval_id
        if not eval_results_path.exists():
            return results

        # Load all result files
        result_files = sorted(eval_results_path.glob("result_*.json"), reverse=True)

        for file_path in result_files[:limit]:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Convert datetime strings back to datetime objects
                if "started_at" in data:
                    data["started_at"] = datetime.fromisoformat(
                        data["started_at"].replace("Z", "+00:00")
                    )
                if "completed_at" in data:
                    data["completed_at"] = datetime.fromisoformat(
                        data["completed_at"].replace("Z", "+00:00")
                    )

                result = EvalExecutionResult(**data)
                results.append(result)
            except Exception as e:
                print(f"Error loading result file {file_path}: {e}")
                continue

        return results

    def get_latest_execution_result(
        self, eval_id: str
    ) -> Optional[EvalExecutionResult]:
        """Get the most recent execution result"""
        results = self.get_execution_results(eval_id, limit=1)
        return results[0] if results else None


class SupabaseConceptEvalStorage(ConceptEvalStorageBackend):
    """Supabase-based concept evaluation storage backend"""

    def __init__(self, supabase_client):
        self.client = supabase_client
        self.definitions_table = "concept_eval_definitions"
        self.results_table = "concept_eval_results"

    def save_evaluation_definition(self, eval_def: ConceptEvalDefinition) -> bool:
        """Save evaluation definition to Supabase"""
        try:
            data = eval_def.model_dump()
            # Convert lists/dicts to JSON strings for storage
            data["test_cases"] = json.dumps(data["test_cases"])
            data["concept_checks"] = json.dumps(data["concept_checks"])
            data["metadata"] = json.dumps(data["metadata"])

            self.client.table(self.definitions_table).upsert(data).execute()
            return True
        except Exception as e:
            print(f"Error saving evaluation definition to Supabase: {e}")
            return False

    def get_evaluation_definition(
        self, eval_id: str
    ) -> Optional[ConceptEvalDefinition]:
        """Get evaluation definition from Supabase"""
        try:
            result = (
                self.client.table(self.definitions_table)
                .select("*")
                .eq("eval_id", eval_id)
                .execute()
            )

            if not result.data:
                return None

            data = result.data[0]
            # Parse JSON fields back to Python objects
            data["test_cases"] = json.loads(data["test_cases"])
            data["concept_checks"] = json.loads(data["concept_checks"])
            data["metadata"] = json.loads(data["metadata"])

            return ConceptEvalDefinition(**data)
        except Exception as e:
            print(f"Error getting evaluation definition from Supabase: {e}")
            return None

    def list_evaluation_definitions(
        self, tags: Optional[List[str]] = None
    ) -> List[ConceptEvalDefinition]:
        """List evaluation definitions from Supabase"""
        try:
            query = self.client.table(self.definitions_table).select("*")

            # Note: Tag filtering would need to be done in Python since Supabase JSON filtering is complex
            result = query.execute()

            definitions = []
            for data in result.data:
                # Parse JSON fields
                data["test_cases"] = json.loads(data["test_cases"])
                data["concept_checks"] = json.loads(data["concept_checks"])
                data["metadata"] = json.loads(data["metadata"])

                eval_def = ConceptEvalDefinition(**data)

                # Filter by tags if specified
                if tags:
                    eval_tags = eval_def.metadata.get("tags", [])
                    if not any(tag in eval_tags for tag in tags):
                        continue

                definitions.append(eval_def)

            return sorted(definitions, key=lambda x: x.name)
        except Exception as e:
            print(f"Error listing evaluation definitions from Supabase: {e}")
            return []

    def delete_evaluation_definition(self, eval_id: str) -> bool:
        """Delete evaluation definition from Supabase"""
        try:
            self.client.table(self.definitions_table).delete().eq(
                "eval_id", eval_id
            ).execute()
            return True
        except Exception as e:
            print(f"Error deleting evaluation definition from Supabase: {e}")
            return False

    def save_execution_result(self, result: EvalExecutionResult) -> bool:
        """Save execution result to Supabase"""
        try:
            data = result.model_dump()

            # Add unique ID for the result
            data["id"] = str(uuid.uuid4())

            # Convert complex objects to JSON strings
            data["test_case_results"] = json.dumps(
                data["test_case_results"], default=str
            )
            data["concept_breakdown"] = json.dumps(data["concept_breakdown"])
            data["recommendations"] = json.dumps(data["recommendations"])

            self.client.table(self.results_table).insert(data).execute()
            return True
        except Exception as e:
            print(f"Error saving execution result to Supabase: {e}")
            return False

    def get_execution_results(
        self, eval_id: str, limit: int = 50
    ) -> List[EvalExecutionResult]:
        """Get execution results from Supabase"""
        try:
            result = (
                self.client.table(self.results_table)
                .select("*")
                .eq("evaluation_name", eval_id)
                .order("started_at", desc=True)
                .limit(limit)
                .execute()
            )

            results = []
            for data in result.data:
                # Parse JSON fields back to Python objects
                data["test_case_results"] = json.loads(data["test_case_results"])
                data["concept_breakdown"] = json.loads(data["concept_breakdown"])
                data["recommendations"] = json.loads(data["recommendations"])

                # Convert datetime strings back to datetime objects
                if isinstance(data["started_at"], str):
                    data["started_at"] = datetime.fromisoformat(
                        data["started_at"].replace("Z", "+00:00")
                    )
                if isinstance(data["completed_at"], str):
                    data["completed_at"] = datetime.fromisoformat(
                        data["completed_at"].replace("Z", "+00:00")
                    )

                # Remove the database ID before creating the model
                data.pop("id", None)

                exec_result = EvalExecutionResult(**data)
                results.append(exec_result)

            return results
        except Exception as e:
            print(f"Error getting execution results from Supabase: {e}")
            return []

    def get_latest_execution_result(
        self, eval_id: str
    ) -> Optional[EvalExecutionResult]:
        """Get the most recent execution result"""
        results = self.get_execution_results(eval_id, limit=1)
        return results[0] if results else None


def create_concept_eval_storage(
    backend_type: str = "auto", **kwargs
) -> ConceptEvalStorageBackend:
    """
    Factory function to create concept evaluation storage backend

    Args:
        backend_type: "file", "supabase", or "auto"
        **kwargs: Additional arguments for backend initialization

    Returns:
        ConceptEvalStorageBackend instance
    """
    if backend_type == "file":
        return FileConceptEvalStorage(
            kwargs.get("storage_path", "./concept_evaluations")
        )

    elif backend_type == "supabase":
        # Try to import and use supabase client
        try:
            from supabase import create_client

            url = kwargs.get("supabase_url") or os.getenv("SUPABASE_URL")
            key = kwargs.get("supabase_key") or os.getenv("SUPABASE_ANON_KEY")

            if not url or not key:
                raise ValueError("Supabase URL and key required")

            client = create_client(url, key)
            return SupabaseConceptEvalStorage(client)
        except ImportError:
            print("Supabase not available, falling back to file storage")
            return FileConceptEvalStorage()
        except Exception as e:
            print(f"Failed to initialize Supabase: {e}, falling back to file storage")
            return FileConceptEvalStorage()

    elif backend_type == "auto":
        # Try Supabase first, fall back to file
        try:
            return create_concept_eval_storage("supabase", **kwargs)
        except Exception:
            return create_concept_eval_storage("file", **kwargs)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
