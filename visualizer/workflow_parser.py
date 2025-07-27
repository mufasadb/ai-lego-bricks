"""
Workflow Parser for JSON Agent Configurations

Parses JSON agent workflow files and extracts the flow structure,
step relationships, and metadata needed for visualization.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class StepInfo:
    """Information about a workflow step"""

    id: str
    type: str
    description: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: List[str]
    routes: Optional[Dict[str, str]] = None
    max_iterations: Optional[int] = None
    preserve_previous_results: bool = False


@dataclass
class WorkflowStructure:
    """Parsed workflow structure"""

    name: str
    description: str
    config: Dict[str, Any]
    steps: List[StepInfo]
    connections: List[Dict[str, str]]
    entry_points: List[str]
    exit_points: List[str]


class WorkflowParser:
    """Parses JSON workflow files and extracts structure for visualization"""

    def __init__(self):
        self.step_types = {
            "input",
            "document_processing",
            "memory_store",
            "memory_retrieve",
            "llm_chat",
            "llm_structured",
            "llm_vision",
            "chunk_text",
            "condition",
            "loop",
            "output",
            "file_output",
            "human_approval",
            "concept_evaluation",
            "prompt_management",
            "streaming_chat",
            "tts_generation",
            "conversation_management",
        }

    def parse_workflow_file(self, file_path: str) -> WorkflowStructure:
        """Parse a workflow JSON file"""
        with open(file_path, "r") as f:
            workflow_data = json.load(f)
        return self.parse_workflow_dict(workflow_data)

    def parse_workflow_dict(self, workflow_data: Dict[str, Any]) -> WorkflowStructure:
        """Parse a workflow dictionary"""
        # Extract basic info
        name = workflow_data.get("name", "Unnamed Workflow")
        description = workflow_data.get("description", "")
        config = workflow_data.get("config", {})

        # Parse steps
        steps = []
        step_dict = {}

        for step_data in workflow_data.get("steps", []):
            step_info = self._parse_step(step_data)
            steps.append(step_info)
            step_dict[step_info.id] = step_info

        # Analyze connections
        connections = self._analyze_connections(steps)

        # Find entry and exit points
        entry_points = self._find_entry_points(steps, connections)
        exit_points = self._find_exit_points(steps, connections)

        return WorkflowStructure(
            name=name,
            description=description,
            config=config,
            steps=steps,
            connections=connections,
            entry_points=entry_points,
            exit_points=exit_points,
        )

    def _parse_step(self, step_data: Dict[str, Any]) -> StepInfo:
        """Parse a single step"""
        return StepInfo(
            id=step_data.get("id", ""),
            type=step_data.get("type", "unknown"),
            description=step_data.get("description", ""),
            config=step_data.get("config", {}),
            inputs=step_data.get("inputs", {}),
            outputs=step_data.get("outputs", []),
            routes=step_data.get("routes", {}),
            max_iterations=step_data.get("max_iterations"),
            preserve_previous_results=step_data.get("preserve_previous_results", False),
        )

    def _analyze_connections(self, steps: List[StepInfo]) -> List[Dict[str, str]]:
        """Analyze step connections and data flow"""
        connections = []
        step_ids = {step.id for step in steps}

        for step in steps:
            # Input connections
            if isinstance(step.inputs, dict):
                for input_name, input_config in step.inputs.items():
                    if isinstance(input_config, dict) and "from_step" in input_config:
                        source_step = input_config["from_step"]
                        if source_step in step_ids:
                            connections.append(
                                {
                                    "from": source_step,
                                    "to": step.id,
                                    "type": "data",
                                    "label": f"{input_config.get('field', input_name)}",
                                }
                            )

            # Route connections (for conditions and loops)
            if step.routes:
                for condition, target_step in step.routes.items():
                    if target_step in step_ids:
                        connections.append(
                            {
                                "from": step.id,
                                "to": target_step,
                                "type": "route",
                                "label": condition,
                            }
                        )

        # Add sequential connections (steps that don't have explicit inputs)
        self._add_sequential_connections(steps, connections)

        return connections

    def _add_sequential_connections(
        self, steps: List[StepInfo], connections: List[Dict[str, str]]
    ):
        """Add implicit sequential connections between steps"""
        connected_steps = set()

        # Track which steps are already connected
        for conn in connections:
            connected_steps.add(conn["from"])
            connected_steps.add(conn["to"])

        # Add sequential connections for unconnected steps
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            # If the next step isn't already connected and doesn't have explicit inputs
            if (
                next_step.id not in connected_steps
                and not next_step.inputs
                and current_step.type not in ["condition", "loop"]
            ):

                connections.append(
                    {
                        "from": current_step.id,
                        "to": next_step.id,
                        "type": "sequential",
                        "label": "",
                    }
                )

    def _find_entry_points(
        self, steps: List[StepInfo], connections: List[Dict[str, str]]
    ) -> List[str]:
        """Find workflow entry points (steps with no incoming connections)"""
        all_targets = {conn["to"] for conn in connections}
        entry_points = []

        for step in steps:
            if step.id not in all_targets:
                entry_points.append(step.id)

        # If no entry points found, use the first step
        if not entry_points and steps:
            entry_points.append(steps[0].id)

        return entry_points

    def _find_exit_points(
        self, steps: List[StepInfo], connections: List[Dict[str, str]]
    ) -> List[str]:
        """Find workflow exit points (steps with no outgoing connections)"""
        all_sources = {conn["from"] for conn in connections}
        exit_points = []

        for step in steps:
            if step.id not in all_sources:
                exit_points.append(step.id)

        # Output steps are always exit points
        for step in steps:
            if step.type in ["output", "file_output"] and step.id not in exit_points:
                exit_points.append(step.id)

        return exit_points

    def get_step_statistics(self, workflow: WorkflowStructure) -> Dict[str, Any]:
        """Get statistics about the workflow"""
        step_type_counts = {}
        for step in workflow.steps:
            step_type_counts[step.type] = step_type_counts.get(step.type, 0) + 1

        return {
            "total_steps": len(workflow.steps),
            "step_types": step_type_counts,
            "connections": len(workflow.connections),
            "entry_points": len(workflow.entry_points),
            "exit_points": len(workflow.exit_points),
            "has_loops": any(
                step.type == "loop" or step.routes for step in workflow.steps
            ),
            "has_conditions": any(step.type == "condition" for step in workflow.steps),
        }
