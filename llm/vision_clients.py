import os
from typing import List
import httpx
import time
import random
from .llm_types import VisionLLMClient, ChatMessage, VisionConfig


class GeminiVisionClient(VisionLLMClient):
    """Gemini Vision client implementation"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_AI_STUDIO_KEY not found in environment variables")
        
        self.api_base_url = os.getenv("GEMINI_API_BASE_URL", 
                                     "https://generativelanguage.googleapis.com/v1beta")
        self.model = config.model or "gemini-1.5-flash"
    
    def _make_request_with_retry(self, payload):
        """Make API request with retry logic for 503 errors"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.post(
                        f"{self.api_base_url}/models/{self.model}:generateContent",
                        json=payload,
                        headers={"x-goog-api-key": self.api_key}
                    )
                    
                    # Handle specific error codes
                    if response.status_code == 503:
                        print(f"503 Service Unavailable error on attempt {attempt + 1}/{max_retries}")
                        print(f"Response: {response.text}")
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            print(f"Retrying in {delay:.2f} seconds...")
                            time.sleep(delay)
                            continue
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    if "candidates" in result and result["candidates"]:
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        raise RuntimeError("No response from Gemini Vision API")
                        
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"503 error on attempt {attempt + 1}/{max_retries}, retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                elif e.response.status_code == 429:
                    # Handle quota/billing limits with loud warnings
                    error_text = e.response.text
                    print("=" * 80)
                    print("🚨 CRITICAL: GEMINI VISION API QUOTA/BILLING LIMIT EXCEEDED! 🚨")
                    print("=" * 80)
                    print(f"❌ HTTP 429 Error: {error_text}")
                    print()
                    print("💰 BILLING ALERT:")
                    print("   • You have exceeded your Gemini Vision API quota limits")
                    print("   • Vision models typically have higher costs per request")
                    print("   • This may result in unexpected charges if on paid tier")
                    print("   • Consider implementing rate limiting or usage monitoring")
                    print()
                    print("🤖 AGENT WARNING:")
                    print("   • IF THIS IS BEING CONSUMED BY AN AGENT, PLEASE INFORM THE USER")
                    print("   • Agents can rapidly consume vision API quotas without user awareness")
                    print("   • Vision processing is particularly expensive - monitor usage closely")
                    print("   • Consider pausing automated workflows until limits reset")
                    print()
                    print("📊 Next Steps:")
                    print("   1. Check your Google AI Studio billing dashboard")
                    print("   2. Implement request throttling in agent workflows")
                    print("   3. Consider upgrading quota limits if needed")
                    print("   4. Monitor usage patterns to prevent future overages")
                    print("   5. Review vision processing requirements for efficiency")
                    print("=" * 80)
                    raise
                else:
                    print(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded for Gemini Vision API request")
    
    def process_image(self, image_data: str, prompt: str, mime_type: str = "image/png") -> str:
        """Process an image with a text prompt"""
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        return self._make_request_with_retry(payload)
    
    def process_image_with_messages(self, image_data: str, messages: List[ChatMessage], 
                                   mime_type: str = "image/png") -> str:
        """Process an image with chat history"""
        # Convert messages to Gemini format and add image to the last user message
        gemini_messages = []
        
        for i, msg in enumerate(messages):
            role = "user" if msg.role == "user" else "model"
            parts = [{"text": msg.content}]
            
            # Add image to the last user message
            if i == len(messages) - 1 and msg.role == "user":
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })
            
            gemini_messages.append({
                "role": role,
                "parts": parts
            })
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        return self._make_request_with_retry(payload)


class LLaVAClient(VisionLLMClient):
    """LLaVA client implementation for local vision processing"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.base_url = os.getenv("OLLAMA_URL", os.getenv("LLAVA_URL", "http://localhost:11434"))
        self.model = config.model or os.getenv("LLAVA_DEFAULT_MODEL", "llava")
        
        # Verify LLaVA is available
        self._verify_llava_available()
    
    def _verify_llava_available(self):
        """Check if LLaVA is available via Ollama"""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                # Check if any vision model is available (LLaVA, Qwen-VL, etc.)
                vision_models = [m for m in available_models if self._is_vision_model(m)]
                if not vision_models:
                    raise RuntimeError(f"No vision models found. Available models: {available_models}")
                
                # Use the first available vision model if none specified
                if self.model not in available_models and vision_models:
                    self.model = vision_models[0]
                    print(f"Using vision model: {self.model}")
                    
        except Exception as e:
            raise RuntimeError(f"LLaVA not available: {e}")
    
    def process_image(self, image_data: str, prompt: str, mime_type: str = "image/png") -> str:
        """Process an image with a text prompt using LLaVA"""
        # LLaVA via Ollama expects the image as base64 in the message
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_data],  # Base64 image data
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **self.config.extra_params
            }
        }
        
        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            return response.json()["response"]
    
    def process_image_with_messages(self, image_data: str, messages: List[ChatMessage], 
                                   mime_type: str = "image/png") -> str:
        """Process an image with chat history using LLaVA"""
        # For LLaVA, we'll combine the chat history into a single prompt
        combined_prompt = ""
        
        for msg in messages:
            role_prefix = "Human: " if msg.role == "user" else "Assistant: "
            combined_prompt += f"{role_prefix}{msg.content}\n"
        
        # Add instruction for image processing
        combined_prompt += "\nPlease analyze the provided image and respond to the conversation above."
        
        return self.process_image(image_data, combined_prompt, mime_type)
    
    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different LLaVA model
        
        Args:
            new_model: The new model name to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if not self._validate_model(new_model):
            raise ValueError(f"Model '{new_model}' is not available in Ollama")
        
        # Check if it's a vision model
        if not self._is_vision_model(new_model):
            raise ValueError(f"Model '{new_model}' is not a vision model")
        
        old_model = self.model
        self.model = new_model
        print(f"Switched from '{old_model}' to '{new_model}'")
        return True
    
    def get_current_model(self) -> str:
        """
        Get the current model name
        
        Returns:
            Current model name
        """
        return self.model
    
    def _validate_model(self, model_name: str) -> bool:
        """
        Check if a model exists in Ollama
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return model_name in available_models
        except Exception:
            return False
    
    def _is_vision_model(self, model_name: str) -> bool:
        """
        Check if a model is a vision model
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if it's a vision model, False otherwise
        """
        vision_indicators = ["llava", "vision", "multimodal", "vl", "qwen2.5-vl"]
        return any(indicator in model_name.lower() for indicator in vision_indicators)