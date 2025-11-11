"""Argument extraction (Stage 3 of AFA pipeline)."""

import logging
from typing import Dict, List
from .llm_interface import LLMInterface
from .prompts import ARGUMENT_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class ArgumentExtractor:
    """Extract claim-evidence-warrant argument structures."""
    
    def __init__(self, llm: LLMInterface, max_tokens: int = 768):
        """
        Initialize argument extractor.
        
        Args:
            llm: LLM interface
            max_tokens: Max tokens for extraction
        """
        self.llm = llm
        self.max_tokens = max_tokens
    
    def extract(self, article: Dict[str, str]) -> Dict:
        """
        Extract argument structure from article.
        
        Args:
            article: Dict with 'text' and 'headline'
            
        Returns:
            Dict with claim, evidence, warrant, impact, supporting_arguments
        """
        # Format prompt
        prompt = ARGUMENT_EXTRACTION_PROMPT.format(
            headline=article.get("headline", ""),
            text=article["text"][:4000],
        )
        
        # Generate extraction
        try:
            result = self.llm.generate_json(prompt, max_tokens=self.max_tokens)
            
            # Validate structure
            result = self._validate_argument(result)
            
            logger.debug(f"Extracted argument: {result.get('claim', 'None')[:50]}")
            return result
        
        except Exception as e:
            logger.error(f"Argument extraction failed: {e}")
            return {
                "claim": None,
                "evidence": [],
                "warrant": None,
                "impact": None,
                "supporting_arguments": [],
            }
    
    def _validate_argument(self, result: Dict) -> Dict:
        """Validate and normalize argument extraction."""
        # Ensure required fields exist
        if "claim" not in result:
            result["claim"] = None
        if "evidence" not in result:
            result["evidence"] = []
        if "warrant" not in result:
            result["warrant"] = None
        if "impact" not in result:
            result["impact"] = None
        if "supporting_arguments" not in result:
            result["supporting_arguments"] = []
        
        # Ensure evidence is a list
        if not isinstance(result["evidence"], list):
            result["evidence"] = [result["evidence"]] if result["evidence"] else []
        
        # Validate supporting arguments
        if isinstance(result["supporting_arguments"], list):
            validated_supporting = []
            for arg in result["supporting_arguments"]:
                if isinstance(arg, dict) and "claim" in arg:
                    validated_supporting.append(arg)
            result["supporting_arguments"] = validated_supporting
        
        return result
