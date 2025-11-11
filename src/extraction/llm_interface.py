"""Unified LLM interface supporting multiple providers."""

import os
import json
import logging
import time
from typing import Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMInterface:
    """Unified interface for LLM API calls (Gemini, OpenAI, Anthropic, local models)."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_retries: int = 3,
    ):
        """
        Initialize LLM interface.
        
        Args:
            model: Model name (gemini-2.5-flash, gpt-4o, claude-sonnet-4, llama-4-maverick-17b)
            temperature: Sampling temperature
            max_retries: Maximum retry attempts on failure
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize appropriate client
        if "gemini" in model.lower():
            self.provider = "google"
            self._init_google()
        elif "gpt" in model.lower():
            self.provider = "openai"
            self._init_openai()
        elif "claude" in model.lower():
            self.provider = "anthropic"
            self._init_anthropic()
        elif "llama" in model.lower() or "maverick" in model.lower():
            self.provider = "local"
            self._init_local()
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        logger.info(f"Initialized LLMInterface: {self.provider}/{model}")
    
    def _init_google(self):
        """Initialize Google Gemini client."""
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
    
    def _init_local(self):
        """Initialize local model (e.g., LLaMA-4 via transformers)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_path = os.getenv("LOCAL_MODEL_PATH", f"meta-llama/{self.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.client = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info(f"Loaded local model from {model_path}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate completion from LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Override default temperature
            
        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        
        try:
            if self.provider == "google":
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temp,
                        "max_output_tokens": max_tokens,
                    }
                )
                return response.text
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.provider == "local":
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.client.device)
                with torch.no_grad():
                    outputs = self.client.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        do_sample=True,
                    )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        max_tokens: int = 512,
        schema: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate JSON response and parse.
        
        Args:
            prompt: Input prompt (should request JSON output)
            max_tokens: Maximum tokens
            schema: Optional JSON schema for validation
            
        Returns:
            Parsed JSON dict
        """
        response = self.generate(prompt, max_tokens=max_tokens)
        
        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            parsed = json.loads(response)
            
           
            if schema:
              
                for key in schema.get("required", []):
                    if key not in parsed:
                        logger.warning(f"Missing required field: {key}")
            
            return parsed
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response[:200]}")
            return {}
