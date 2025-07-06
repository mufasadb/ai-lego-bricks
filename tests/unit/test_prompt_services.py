"""
Unit tests for prompt management services.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from prompt.prompt_service import PromptService, create_prompt_service
from prompt.prompt_registry import PromptRegistry
from prompt.prompt_models import PromptTemplate, PromptVersion, PromptExecution
from prompt.prompt_storage import FilePromptStorage, SupabasePromptStorage
from prompt.evaluation_service import EvaluationService
from prompt.concept_evaluation_service import ConceptEvaluationService


class TestPromptService:
    """Test suite for PromptService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_storage = Mock()
        self.mock_evaluation = Mock()
        
        self.service = PromptService(
            storage=self.mock_storage,
            evaluation_service=self.mock_evaluation
        )
        
        self.sample_template = PromptTemplate(
            id="test_prompt_1",
            name="Test Prompt",
            description="A test prompt template",
            template="Hello {name}, how are you feeling about {topic}?",
            variables=["name", "topic"],
            category="greeting",
            version="1.0.0"
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.storage == self.mock_storage
        assert self.service.evaluation_service == self.mock_evaluation
    
    def test_create_prompt_template(self):
        """Test creating a prompt template."""
        self.mock_storage.save_template.return_value = "template_id_123"
        
        template_id = self.service.create_template(
            name="Test Prompt",
            template="Hello {name}!",
            description="A greeting prompt",
            variables=["name"],
            category="greeting"
        )
        
        assert template_id == "template_id_123"
        self.mock_storage.save_template.assert_called_once()
        
        # Verify template structure
        call_args = self.mock_storage.save_template.call_args[0]
        saved_template = call_args[0]
        assert saved_template.name == "Test Prompt"
        assert saved_template.template == "Hello {name}!"
        assert saved_template.variables == ["name"]
    
    def test_get_prompt_template(self):
        """Test retrieving a prompt template."""
        self.mock_storage.get_template.return_value = self.sample_template
        
        template = self.service.get_template("test_prompt_1")
        
        assert template == self.sample_template
        self.mock_storage.get_template.assert_called_once_with("test_prompt_1")
    
    def test_list_prompt_templates(self):
        """Test listing prompt templates."""
        mock_templates = [
            self.sample_template,
            PromptTemplate(
                id="test_prompt_2",
                name="Another Prompt",
                template="Goodbye {name}!",
                variables=["name"],
                category="farewell"
            )
        ]
        self.mock_storage.list_templates.return_value = mock_templates
        
        templates = self.service.list_templates()
        
        assert len(templates) == 2
        assert templates[0].name == "Test Prompt"
        assert templates[1].name == "Another Prompt"
        self.mock_storage.list_templates.assert_called_once()
    
    def test_list_templates_by_category(self):
        """Test listing templates by category."""
        mock_templates = [self.sample_template]
        self.mock_storage.list_templates.return_value = mock_templates
        
        templates = self.service.list_templates(category="greeting")
        
        assert len(templates) == 1
        assert templates[0].category == "greeting"
        self.mock_storage.list_templates.assert_called_once_with(category="greeting")
    
    def test_render_prompt(self):
        """Test rendering a prompt with variables."""
        self.mock_storage.get_template.return_value = self.sample_template
        
        rendered = self.service.render_prompt(
            "test_prompt_1",
            variables={"name": "Alice", "topic": "AI"}
        )
        
        assert rendered == "Hello Alice, how are you feeling about AI?"
        self.mock_storage.get_template.assert_called_once_with("test_prompt_1")
    
    def test_render_prompt_missing_variables(self):
        """Test rendering prompt with missing variables."""
        self.mock_storage.get_template.return_value = self.sample_template
        
        with pytest.raises(ValueError, match="Missing required variable"):
            self.service.render_prompt(
                "test_prompt_1",
                variables={"name": "Alice"}  # Missing 'topic'
            )
    
    def test_render_prompt_extra_variables(self):
        """Test rendering prompt with extra variables."""
        self.mock_storage.get_template.return_value = self.sample_template
        
        # Should work fine with extra variables
        rendered = self.service.render_prompt(
            "test_prompt_1",
            variables={"name": "Alice", "topic": "AI", "extra": "ignored"}
        )
        
        assert rendered == "Hello Alice, how are you feeling about AI?"
    
    def test_update_prompt_template(self):
        """Test updating a prompt template."""
        updated_template = PromptTemplate(
            id="test_prompt_1",
            name="Updated Test Prompt",
            template="Hi {name}, what do you think about {topic}?",
            variables=["name", "topic"],
            category="greeting",
            version="1.1.0"
        )
        
        self.mock_storage.update_template.return_value = updated_template
        
        result = self.service.update_template(
            "test_prompt_1",
            name="Updated Test Prompt",
            template="Hi {name}, what do you think about {topic}?"
        )
        
        assert result.name == "Updated Test Prompt"
        assert result.version == "1.1.0"
        self.mock_storage.update_template.assert_called_once()
    
    def test_delete_prompt_template(self):
        """Test deleting a prompt template."""
        self.mock_storage.delete_template.return_value = True
        
        result = self.service.delete_template("test_prompt_1")
        
        assert result is True
        self.mock_storage.delete_template.assert_called_once_with("test_prompt_1")
    
    def test_log_prompt_execution(self):
        """Test logging prompt execution."""
        execution = PromptExecution(
            template_id="test_prompt_1",
            variables={"name": "Alice", "topic": "AI"},
            rendered_prompt="Hello Alice, how are you feeling about AI?",
            response="I'm excited about AI developments!",
            execution_time=1.5,
            timestamp=datetime.utcnow()
        )
        
        self.mock_storage.log_execution.return_value = "execution_id_123"
        
        execution_id = self.service.log_execution(execution)
        
        assert execution_id == "execution_id_123"
        self.mock_storage.log_execution.assert_called_once_with(execution)
    
    def test_get_execution_history(self):
        """Test getting execution history."""
        mock_executions = [
            PromptExecution(
                template_id="test_prompt_1",
                variables={"name": "Alice"},
                rendered_prompt="Hello Alice!",
                response="Hi there!",
                execution_time=1.0
            )
        ]
        self.mock_storage.get_executions.return_value = mock_executions
        
        executions = self.service.get_execution_history("test_prompt_1", limit=10)
        
        assert len(executions) == 1
        assert executions[0].variables["name"] == "Alice"
        self.mock_storage.get_executions.assert_called_once_with("test_prompt_1", limit=10)
    
    def test_evaluate_prompt_performance(self):
        """Test evaluating prompt performance."""
        mock_metrics = {
            "avg_execution_time": 1.2,
            "success_rate": 0.95,
            "total_executions": 100
        }
        self.mock_evaluation.evaluate_prompt.return_value = mock_metrics
        
        metrics = self.service.evaluate_prompt("test_prompt_1")
        
        assert metrics["avg_execution_time"] == 1.2
        assert metrics["success_rate"] == 0.95
        self.mock_evaluation.evaluate_prompt.assert_called_once_with("test_prompt_1")


class TestPromptRegistry:
    """Test suite for PromptRegistry."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.registry = PromptRegistry()
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        assert self.registry is not None
        assert hasattr(self.registry, 'register')
        assert hasattr(self.registry, 'get')
        assert hasattr(self.registry, 'list_all')
    
    def test_register_prompt(self):
        """Test registering a prompt."""
        template = self.registry.register(
            name="test_prompt",
            template="Hello {name}!",
            variables=["name"],
            category="greeting"
        )
        
        assert template.name == "test_prompt"
        assert template.template == "Hello {name}!"
        assert template.variables == ["name"]
        assert template.category == "greeting"
    
    def test_get_registered_prompt(self):
        """Test getting a registered prompt."""
        # Register a prompt first
        self.registry.register(
            name="test_prompt",
            template="Hello {name}!",
            variables=["name"]
        )
        
        # Retrieve it
        template = self.registry.get("test_prompt")
        
        assert template is not None
        assert template.name == "test_prompt"
        assert template.template == "Hello {name}!"
    
    def test_get_nonexistent_prompt(self):
        """Test getting a non-existent prompt."""
        template = self.registry.get("nonexistent_prompt")
        assert template is None
    
    def test_list_all_prompts(self):
        """Test listing all registered prompts."""
        # Register multiple prompts
        self.registry.register("prompt1", "Template 1", [])
        self.registry.register("prompt2", "Template 2", [])
        
        all_prompts = self.registry.list_all()
        
        assert len(all_prompts) == 2
        prompt_names = [p.name for p in all_prompts]
        assert "prompt1" in prompt_names
        assert "prompt2" in prompt_names
    
    def test_register_duplicate_prompt(self):
        """Test registering a prompt with duplicate name."""
        self.registry.register("duplicate", "Template 1", [])
        
        # Should update the existing prompt
        updated = self.registry.register("duplicate", "Template 2", [])
        
        assert updated.template == "Template 2"
        
        # Should only have one prompt with this name
        all_prompts = self.registry.list_all()
        duplicate_prompts = [p for p in all_prompts if p.name == "duplicate"]
        assert len(duplicate_prompts) == 1
    
    def test_filter_prompts_by_category(self):
        """Test filtering prompts by category."""
        self.registry.register("greeting1", "Hello!", [], category="greeting")
        self.registry.register("greeting2", "Hi there!", [], category="greeting")
        self.registry.register("farewell1", "Goodbye!", [], category="farewell")
        
        greeting_prompts = self.registry.list_by_category("greeting")
        
        assert len(greeting_prompts) == 2
        for prompt in greeting_prompts:
            assert prompt.category == "greeting"


class TestPromptStorage:
    """Test suite for prompt storage implementations."""
    
    def test_file_storage_initialization(self):
        """Test file storage initialization."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            storage = FilePromptStorage("/tmp/test_prompts")
            
            assert storage is not None
            assert storage.storage_path.name == "test_prompts"
            mock_mkdir.assert_called_once()
    
    def test_file_storage_save_template(self):
        """Test saving template to file storage."""
        template = PromptTemplate(
            id="test_id",
            name="Test Template",
            template="Hello {name}!",
            variables=["name"]
        )
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.dump') as mock_json_dump:
                storage = FilePromptStorage("/tmp/test")
                template_id = storage.save_template(template)
                
                assert template_id == "test_id"
                mock_open.assert_called_once()
                mock_json_dump.assert_called_once()
    
    def test_file_storage_get_template(self):
        """Test getting template from file storage."""
        template_data = {
            "id": "test_id",
            "name": "Test Template",
            "template": "Hello {name}!",
            "variables": ["name"],
            "category": "greeting"
        }
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.load', return_value=template_data) as mock_json_load:
                with patch('pathlib.Path.exists', return_value=True):
                    storage = FilePromptStorage("/tmp/test")
                    template = storage.get_template("test_id")
                    
                    assert template.name == "Test Template"
                    assert template.template == "Hello {name}!"
                    mock_open.assert_called_once()
                    mock_json_load.assert_called_once()
    
    def test_supabase_storage_initialization(self):
        """Test Supabase storage initialization."""
        with patch('supabase.create_client') as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client
            
            storage = SupabasePromptStorage("https://test.supabase.co", "test_key")
            
            assert storage is not None
            assert storage.client == mock_client
            mock_create_client.assert_called_once_with("https://test.supabase.co", "test_key")
    
    def test_supabase_storage_save_template(self):
        """Test saving template to Supabase storage."""
        template = PromptTemplate(
            id="test_id",
            name="Test Template",
            template="Hello {name}!",
            variables=["name"]
        )
        
        with patch('supabase.create_client') as mock_create_client:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [{"id": "test_id"}]
            mock_client.table.return_value.insert.return_value.execute.return_value = mock_response
            mock_create_client.return_value = mock_client
            
            storage = SupabasePromptStorage("https://test.supabase.co", "test_key")
            template_id = storage.save_template(template)
            
            assert template_id == "test_id"
            mock_client.table.assert_called_once_with("prompt_templates")


class TestEvaluationService:
    """Test suite for EvaluationService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_storage = Mock()
        self.service = EvaluationService(self.mock_storage)
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.storage == self.mock_storage
    
    def test_evaluate_prompt_performance(self):
        """Test evaluating prompt performance."""
        mock_executions = [
            PromptExecution(
                template_id="test_prompt",
                execution_time=1.0,
                response="Good response",
                success=True
            ),
            PromptExecution(
                template_id="test_prompt",
                execution_time=1.5,
                response="Another response",
                success=True
            ),
            PromptExecution(
                template_id="test_prompt",
                execution_time=2.0,
                response="Failed response",
                success=False
            )
        ]
        self.mock_storage.get_executions.return_value = mock_executions
        
        metrics = self.service.evaluate_prompt("test_prompt")
        
        assert metrics["total_executions"] == 3
        assert metrics["success_rate"] == 2/3  # 2 successful out of 3
        assert metrics["avg_execution_time"] == 1.5  # (1.0 + 1.5 + 2.0) / 3
        assert "avg_response_length" in metrics
    
    def test_compare_prompt_versions(self):
        """Test comparing different prompt versions."""
        version1_executions = [
            PromptExecution(template_id="v1", execution_time=1.0, success=True),
            PromptExecution(template_id="v1", execution_time=1.2, success=True)
        ]
        version2_executions = [
            PromptExecution(template_id="v2", execution_time=0.8, success=True),
            PromptExecution(template_id="v2", execution_time=0.9, success=True)
        ]
        
        self.mock_storage.get_executions.side_effect = [version1_executions, version2_executions]
        
        comparison = self.service.compare_prompts("v1", "v2")
        
        assert comparison["v1"]["avg_execution_time"] == 1.1
        assert comparison["v2"]["avg_execution_time"] == 0.85
        assert comparison["v2"]["avg_execution_time"] < comparison["v1"]["avg_execution_time"]
    
    def test_get_performance_trends(self):
        """Test getting performance trends over time."""
        mock_executions = [
            PromptExecution(
                template_id="test_prompt",
                execution_time=1.0,
                timestamp=datetime(2025, 1, 1, 10, 0),
                success=True
            ),
            PromptExecution(
                template_id="test_prompt", 
                execution_time=0.9,
                timestamp=datetime(2025, 1, 1, 11, 0),
                success=True
            ),
            PromptExecution(
                template_id="test_prompt",
                execution_time=0.8,
                timestamp=datetime(2025, 1, 1, 12, 0),
                success=True
            )
        ]
        self.mock_storage.get_executions.return_value = mock_executions
        
        trends = self.service.get_performance_trends("test_prompt", days=1)
        
        assert len(trends["hourly_metrics"]) > 0
        assert "execution_time_trend" in trends
        assert trends["execution_time_trend"] == "improving"  # Times are decreasing


class TestCreatePromptService:
    """Test suite for create_prompt_service factory function."""
    
    @patch('prompt.prompt_service.FilePromptStorage')
    @patch('prompt.prompt_service.EvaluationService')
    def test_create_service_file_storage(self, mock_eval, mock_storage):
        """Test creating service with file storage."""
        mock_storage_instance = Mock()
        mock_eval_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        mock_eval.return_value = mock_eval_instance
        
        service = create_prompt_service("file", storage_path="/tmp/prompts")
        
        assert isinstance(service, PromptService)
        mock_storage.assert_called_once_with("/tmp/prompts")
        mock_eval.assert_called_once_with(mock_storage_instance)
    
    @patch('prompt.prompt_service.SupabasePromptStorage')
    @patch('prompt.prompt_service.EvaluationService')
    @patch('os.getenv')
    def test_create_service_supabase_storage(self, mock_getenv, mock_eval, mock_storage):
        """Test creating service with Supabase storage."""
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_ANON_KEY': 'test_key'
        }.get(key)
        
        mock_storage_instance = Mock()
        mock_eval_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        mock_eval.return_value = mock_eval_instance
        
        service = create_prompt_service("supabase")
        
        assert isinstance(service, PromptService)
        mock_storage.assert_called_once_with('https://test.supabase.co', 'test_key')
        mock_eval.assert_called_once_with(mock_storage_instance)
    
    @patch('prompt.prompt_service.FilePromptStorage')
    @patch('os.getenv')
    def test_create_service_auto_detection(self, mock_getenv, mock_storage):
        """Test creating service with auto detection."""
        # No Supabase credentials available
        mock_getenv.return_value = None
        
        mock_storage_instance = Mock()
        mock_storage.return_value = mock_storage_instance
        
        service = create_prompt_service("auto")
        
        assert isinstance(service, PromptService)
        # Should fall back to file storage
        mock_storage.assert_called_once()
    
    def test_create_service_invalid_provider(self):
        """Test creating service with invalid provider."""
        with pytest.raises(ValueError, match="Invalid storage provider"):
            create_prompt_service("invalid_provider")


class TestConceptEvaluationService:
    """Test suite for ConceptEvaluationService."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm_service = Mock()
        self.mock_storage = Mock()
        
        self.service = ConceptEvaluationService(
            llm_service=self.mock_llm_service,
            storage=self.mock_storage
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert self.service.llm_service == self.mock_llm_service
        assert self.service.storage == self.mock_storage
    
    def test_evaluate_concept(self):
        """Test evaluating a concept."""
        self.mock_llm_service.generate.return_value = "This concept is clear and well-defined."
        self.mock_storage.store_evaluation.return_value = "eval_id_123"
        
        result = self.service.evaluate_concept(
            concept="Machine learning algorithms",
            criteria=["clarity", "completeness", "accuracy"]
        )
        
        assert result is not None
        assert "evaluation_id" in result
        assert result["evaluation_id"] == "eval_id_123"
        
        self.mock_llm_service.generate.assert_called_once()
        self.mock_storage.store_evaluation.assert_called_once()
    
    def test_compare_concepts(self):
        """Test comparing multiple concepts."""
        self.mock_llm_service.generate.return_value = "Concept A is more comprehensive than Concept B."
        
        comparison = self.service.compare_concepts([
            "Supervised learning", 
            "Unsupervised learning"
        ])
        
        assert comparison is not None
        assert "comparison_result" in comparison
        self.mock_llm_service.generate.assert_called_once()
    
    def test_get_evaluation_history(self):
        """Test getting evaluation history."""
        mock_evaluations = [
            {"concept": "ML", "score": 8.5, "timestamp": "2025-01-01"},
            {"concept": "AI", "score": 9.0, "timestamp": "2025-01-02"}
        ]
        self.mock_storage.get_evaluations.return_value = mock_evaluations
        
        history = self.service.get_evaluation_history(limit=10)
        
        assert len(history) == 2
        assert history[0]["concept"] == "ML"
        assert history[1]["score"] == 9.0
        self.mock_storage.get_evaluations.assert_called_once_with(limit=10)