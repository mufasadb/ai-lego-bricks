"""
Mermaid Diagram Generator for Agent Workflows

Converts parsed workflow structures into Mermaid flowchart syntax
for visualization in web browsers.
"""

from typing import Dict, List, Any
try:
    from .workflow_parser import WorkflowStructure, StepInfo
except ImportError:
    from workflow_parser import WorkflowStructure, StepInfo


class MermaidGenerator:
    """Generates Mermaid flowchart diagrams from workflow structures"""
    
    def __init__(self):
        # Define node styles for different step types
        self.step_styles = {
            "input": {"shape": "([", "color": "#e1f5fe", "border": "#0277bd"},
            "output": {"shape": "([", "color": "#f3e5f5", "border": "#7b1fa2"},
            "file_output": {"shape": "([", "color": "#f3e5f5", "border": "#7b1fa2"},
            "llm_chat": {"shape": "[", "color": "#e8f5e8", "border": "#2e7d32"},
            "llm_structured": {"shape": "[", "color": "#e8f5e8", "border": "#2e7d32"},
            "llm_vision": {"shape": "[", "color": "#e8f5e8", "border": "#2e7d32"},
            "condition": {"shape": "{", "color": "#fff3e0", "border": "#f57c00"},
            "loop": {"shape": "((", "color": "#fce4ec", "border": "#c2185b"},
            "memory_store": {"shape": "[(", "color": "#f1f8e9", "border": "#558b2f"},
            "memory_retrieve": {"shape": "[(", "color": "#f1f8e9", "border": "#558b2f"},
            "document_processing": {"shape": "[", "color": "#e3f2fd", "border": "#1976d2"},
            "chunk_text": {"shape": "[", "color": "#e3f2fd", "border": "#1976d2"},
            "human_approval": {"shape": "([", "color": "#ffebee", "border": "#d32f2f"},
            "concept_evaluation": {"shape": "[", "color": "#f9fbe7", "border": "#827717"},
            "prompt_management": {"shape": "[", "color": "#f9fbe7", "border": "#827717"},
            "streaming_chat": {"shape": "[", "color": "#e8f5e8", "border": "#2e7d32"},
            "tts_generation": {"shape": "[", "color": "#e0f2f1", "border": "#00695c"},
            "conversation_management": {"shape": "[", "color": "#e0f2f1", "border": "#00695c"}
        }
        
        # Default style for unknown step types
        self.default_style = {"shape": "[", "color": "#f5f5f5", "border": "#616161"}
        
        # Connection styles
        self.connection_styles = {
            "data": {"style": "-->", "color": "#2196f3"},
            "route": {"style": "-.->", "color": "#ff9800"},
            "sequential": {"style": "-->", "color": "#9e9e9e"},
            "loop_back": {"style": "-.->", "color": "#e91e63"}
        }
    
    def generate_mermaid(self, workflow: WorkflowStructure) -> str:
        """Generate complete Mermaid flowchart from workflow structure"""
        lines = []
        
        # Start flowchart
        lines.append("flowchart TD")
        lines.append("")
        
        # Add title
        lines.append(f"    %% {workflow.name}")
        if workflow.description:
            lines.append(f"    %% {workflow.description}")
        lines.append("")
        
        # Generate nodes
        node_lines = self._generate_nodes(workflow.steps)
        lines.extend(node_lines)
        lines.append("")
        
        # Generate connections
        connection_lines = self._generate_connections(workflow.connections)
        lines.extend(connection_lines)
        lines.append("")
        
        # Generate styles
        style_lines = self._generate_styles(workflow.steps)
        lines.extend(style_lines)
        
        return "\n".join(lines)
    
    def _generate_nodes(self, steps: List[StepInfo]) -> List[str]:
        """Generate Mermaid node definitions"""
        lines = []
        lines.append("    %% Workflow Steps")
        
        for step in steps:
            style = self.step_styles.get(step.type, self.default_style)
            shape_start, shape_end = self._get_shape_tokens(style["shape"])
            
            # Clean step description for display
            display_text = self._clean_text_for_mermaid(step.description or step.id)
            
            # Add step type indicator
            type_indicator = f"[{step.type.upper()}]"
            
            # Create node definition
            node_def = f"    {step.id}{shape_start}\"{type_indicator}\\n{display_text}\"{shape_end}"
            lines.append(node_def)
            
            # Add iteration info if applicable
            if step.max_iterations:
                lines.append(f"    %% {step.id} - Max iterations: {step.max_iterations}")
        
        return lines
    
    def _generate_connections(self, connections: List[Dict[str, str]]) -> List[str]:
        """Generate Mermaid connection definitions"""
        lines = []
        lines.append("    %% Connections")
        
        # Group connections by type
        connection_groups = {}
        for conn in connections:
            conn_type = conn.get("type", "data")
            if conn_type not in connection_groups:
                connection_groups[conn_type] = []
            connection_groups[conn_type].append(conn)
        
        # Generate connections by type
        for conn_type, conns in connection_groups.items():
            if conns:
                lines.append(f"    %% {conn_type.title()} connections")
                
                for conn in conns:
                    style = self.connection_styles.get(conn_type, self.connection_styles["data"])
                    label = conn.get("label", "")
                    
                    if label:
                        # Clean label for display
                        clean_label = self._clean_text_for_mermaid(label)
                        conn_def = f"    {conn['from']} {style['style']}|{clean_label}| {conn['to']}"
                    else:
                        conn_def = f"    {conn['from']} {style['style']} {conn['to']}"
                    
                    lines.append(conn_def)
                
                lines.append("")
        
        return lines
    
    def _generate_styles(self, steps: List[StepInfo]) -> List[str]:
        """Generate Mermaid style definitions"""
        lines = []
        lines.append("    %% Styles")
        
        # Group steps by type for styling
        step_types_used = set(step.type for step in steps)
        
        for step_type in step_types_used:
            style = self.step_styles.get(step_type, self.default_style)
            style_name = f"style_{step_type}"
            
            # Find all steps of this type
            steps_of_type = [step.id for step in steps if step.type == step_type]
            
            # Apply style to each step
            for step_id in steps_of_type:
                style_def = f"    style {step_id} fill:{style['color']},stroke:{style['border']},stroke-width:2px"
                lines.append(style_def)
        
        # Add special highlighting for entry and exit points
        lines.append("")
        lines.append("    %% Special highlighting")
        lines.append("    %% Entry points have thicker borders")
        lines.append("    %% Exit points have dashed borders")
        
        return lines
    
    def _get_shape_tokens(self, shape: str) -> tuple:
        """Get the start and end tokens for a Mermaid shape"""
        shape_mapping = {
            "[": ("[", "]"),        # Rectangle
            "([": ("([", "])"),     # Stadium
            "{": ("{", "}"),        # Diamond
            "((": ("((", "))"),     # Circle
            "[(": ("[(", ")]"),     # Subroutine
            ">": (">", "]"),        # Asymmetric
        }
        
        if shape in shape_mapping:
            return shape_mapping[shape]
        else:
            return ("[", "]")  # Default rectangle
    
    def _clean_text_for_mermaid(self, text: str) -> str:
        """Clean text for use in Mermaid diagrams"""
        # Remove or escape special characters
        text = text.replace("\"", "'")
        text = text.replace("\n", " ")
        text = text.replace("|", "\\|")
        
        # Limit length
        if len(text) > 50:
            text = text[:47] + "..."
        
        return text
    
    def generate_legend(self) -> str:
        """Generate a legend explaining the diagram symbols"""
        lines = []
        lines.append("flowchart LR")
        lines.append("    %% Legend")
        lines.append("")
        
        # Create legend nodes
        legend_items = [
            ("input", "Input/Data Collection"),
            ("llm_chat", "LLM Processing"),
            ("condition", "Conditional Logic"),
            ("loop", "Loop/Iteration"),
            ("memory_store", "Memory Operations"),
            ("output", "Output/Results")
        ]
        
        for i, (step_type, description) in enumerate(legend_items):
            style = self.step_styles.get(step_type, self.default_style)
            shape_start, shape_end = self._get_shape_tokens(style["shape"])
            
            node_id = f"L{i+1}"
            node_def = f"    {node_id}{shape_start}\"{step_type.upper()}\\n{description}\"{shape_end}"
            lines.append(node_def)
            
            # Add styling
            style_def = f"    style {node_id} fill:{style['color']},stroke:{style['border']},stroke-width:2px"
            lines.append(style_def)
        
        return "\n".join(lines)
    
    def generate_with_statistics(self, workflow: WorkflowStructure) -> Dict[str, Any]:
        """Generate Mermaid diagram with additional statistics"""
        try:
            from .workflow_parser import WorkflowParser
        except ImportError:
            from workflow_parser import WorkflowParser
        
        parser = WorkflowParser()
        stats = parser.get_step_statistics(workflow)
        
        return {
            "diagram": self.generate_mermaid(workflow),
            "legend": self.generate_legend(),
            "statistics": stats,
            "complexity_score": self._calculate_complexity_score(workflow, stats)
        }
    
    def _calculate_complexity_score(self, workflow: WorkflowStructure, stats: Dict[str, Any]) -> int:
        """Calculate a complexity score for the workflow"""
        score = 0
        
        # Base score from number of steps
        score += stats["total_steps"]
        
        # Add complexity for conditions and loops
        score += stats["step_types"].get("condition", 0) * 2
        score += stats["step_types"].get("loop", 0) * 3
        
        # Add complexity for multiple entry/exit points
        if stats["entry_points"] > 1:
            score += stats["entry_points"] - 1
        if stats["exit_points"] > 1:
            score += stats["exit_points"] - 1
        
        # Add complexity for connections
        score += len(workflow.connections) * 0.5
        
        return int(score)