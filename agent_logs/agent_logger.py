"""
Simple logging utility for agent examples
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class AgentLogger:
    """Simple file-based logger for agent workflow tracking"""
    
    def __init__(self, agent_name: str, log_dir: str = "agent_logs"):
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{agent_name}_{self.session_id}.log")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_stage("INIT", "Agent logging initialized", {
            "agent_name": agent_name,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_stage(self, stage: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a stage of the agent workflow"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "agent": self.agent_name,
            "session": self.session_id,
            "stage": stage,
            "message": message,
            "data": data or {}
        }
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also print to console for immediate feedback
        print(f"[{timestamp}] {self.agent_name} - {stage}: {message}")
        if data:
            print(f"  Data: {json.dumps(data, indent=2)}")
    
    def log_model_switch(self, from_model: str, to_model: str, purpose: str):
        """Log a model switch event"""
        self.log_stage("MODEL_SWITCH", f"Switching from {from_model} to {to_model}", {
            "from_model": from_model,
            "to_model": to_model,
            "purpose": purpose
        })
    
    def log_processing_start(self, stage: str, input_info: Dict[str, Any]):
        """Log the start of a processing stage"""
        self.log_stage(f"{stage}_START", f"Starting {stage} processing", input_info)
    
    def log_processing_end(self, stage: str, output_info: Dict[str, Any]):
        """Log the end of a processing stage"""
        self.log_stage(f"{stage}_END", f"Completed {stage} processing", output_info)
    
    def log_model_response(self, model: str, prompt: str, response: str, stage: str):
        """Log the actual model response for verification"""
        self.log_stage(f"{stage}_MODEL_RESPONSE", f"Model {model} response", {
            "model": model,
            "prompt_preview": prompt[:300] + "..." if len(prompt) > 300 else prompt,
            "response": response,
            "response_length": len(response),
            "stage": stage
        })
    
    def log_error(self, stage: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error event"""
        self.log_stage(f"{stage}_ERROR", f"Error in {stage}: {str(error)}", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        })
    
    def get_log_file_path(self) -> str:
        """Get the path to the current log file"""
        return self.log_file