from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from llm.generation_service import GenerationService
from prompt.prompt_service import PromptService

logger = logging.getLogger(__name__)


class GraphEntity(BaseModel):
    """Represents an entity in the knowledge graph"""

    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """Represents a relationship between entities"""

    source_entity: str
    target_entity: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphMemoryFormat(BaseModel):
    """Structured representation of memory content as a graph"""

    original_content: str
    entities: List[GraphEntity]
    relationships: List[GraphRelationship]
    summary: str
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class GraphFormatterService:
    """
    Service for converting unstructured memory content into graph-formatted data
    using LLM-powered entity and relationship extraction
    """

    def __init__(
        self, generation_service: GenerationService, prompt_service: PromptService
    ):
        self.generation_service = generation_service
        self.prompt_service = prompt_service

    def format_memory_as_graph(
        self,
        content: str,
        context: Optional[str] = None,
        extraction_mode: str = "comprehensive",
    ) -> GraphMemoryFormat:
        """
        Convert memory content into graph format using LLM extraction

        Args:
            content: The memory content to process
            context: Optional context to help with extraction
            extraction_mode: 'comprehensive', 'entities_only', 'relationships_only'

        Returns:
            GraphMemoryFormat with extracted entities and relationships
        """
        try:
            # Get the appropriate prompt based on extraction mode
            prompt_name = f"graph_memory_extraction_{extraction_mode}"

            # Prepare variables for prompt template
            prompt_variables = {
                "content": content,
                "context": context or "",
                "timestamp": datetime.now().isoformat(),
            }

            # Execute the prompt with fallback
            extraction_result = self._execute_extraction_prompt(
                prompt_name, prompt_variables
            )

            if not extraction_result:
                # Fallback to simple extraction
                return self._fallback_extraction(content)

            # Parse the structured response
            return self._parse_extraction_result(content, extraction_result)

        except Exception as e:
            logger.error(f"Error in graph formatting: {e}")
            return self._fallback_extraction(content)

    def _execute_extraction_prompt(
        self, prompt_name: str, variables: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute extraction prompt with fallback"""
        try:
            # Try to get the specific prompt
            prompt = self.prompt_service.get_prompt(prompt_name)
            if not prompt:
                # Fall back to default graph extraction prompt
                prompt = self.prompt_service.get_prompt(
                    "graph_memory_extraction_default"
                )

            if not prompt:
                logger.warning("No graph extraction prompt found, using fallback")
                return None

            # Render the prompt with variables
            rendered_prompt = self.prompt_service.render_prompt(prompt.id, variables)

            # Execute with structured response
            response = self.generation_service.generate_structured_response(
                prompt=rendered_prompt,
                response_format={
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "properties": {"type": "object"},
                                },
                                "required": ["name", "type"],
                            },
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_entity": {"type": "string"},
                                    "target_entity": {"type": "string"},
                                    "relationship_type": {"type": "string"},
                                    "properties": {"type": "object"},
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": [
                                    "source_entity",
                                    "target_entity",
                                    "relationship_type",
                                ],
                            },
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["entities", "relationships", "summary"],
                },
            )

            return response

        except Exception as e:
            logger.error(f"Error executing extraction prompt: {e}")
            return None

    def _parse_extraction_result(
        self, original_content: str, extraction_result: Dict[str, Any]
    ) -> GraphMemoryFormat:
        """Parse LLM extraction result into GraphMemoryFormat"""
        entities = []
        relationships = []

        # Parse entities
        for entity_data in extraction_result.get("entities", []):
            entity = GraphEntity(
                name=entity_data["name"],
                type=entity_data["type"],
                properties=entity_data.get("properties", {}),
            )
            entities.append(entity)

        # Parse relationships
        for rel_data in extraction_result.get("relationships", []):
            relationship = GraphRelationship(
                source_entity=rel_data["source_entity"],
                target_entity=rel_data["target_entity"],
                relationship_type=rel_data["relationship_type"],
                properties=rel_data.get("properties", {}),
                confidence=rel_data.get("confidence", 1.0),
            )
            relationships.append(relationship)

        return GraphMemoryFormat(
            original_content=original_content,
            entities=entities,
            relationships=relationships,
            summary=extraction_result.get("summary", ""),
            extraction_metadata={
                "extraction_method": "llm_structured",
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        )

    def _fallback_extraction(self, content: str) -> GraphMemoryFormat:
        """Simple fallback extraction when LLM extraction fails"""
        # Basic entity extraction from simple patterns
        entities = []
        relationships = []

        # Simple keyword-based entity extraction as fallback
        words = content.split()
        potential_entities = [
            word.strip(".,!?") for word in words if len(word) > 3 and word.isalpha()
        ]

        for i, word in enumerate(potential_entities[:5]):  # Limit to first 5
            entities.append(
                GraphEntity(
                    name=word,
                    type="concept",
                    properties={"extraction_method": "fallback"},
                )
            )

        return GraphMemoryFormat(
            original_content=content,
            entities=entities,
            relationships=relationships,
            summary=content[:200] + "..." if len(content) > 200 else content,
            extraction_metadata={
                "extraction_method": "fallback",
                "entity_count": len(entities),
                "relationship_count": 0,
            },
        )

    def format_multiple_memories(
        self, memory_contents: List[str], batch_size: int = 5
    ) -> List[GraphMemoryFormat]:
        """
        Process multiple memories in batches

        Args:
            memory_contents: List of memory content strings
            batch_size: Number of memories to process at once

        Returns:
            List of GraphMemoryFormat objects
        """
        results = []

        for i in range(0, len(memory_contents), batch_size):
            batch = memory_contents[i : i + batch_size]

            for content in batch:
                formatted_memory = self.format_memory_as_graph(content)
                results.append(formatted_memory)

        return results
