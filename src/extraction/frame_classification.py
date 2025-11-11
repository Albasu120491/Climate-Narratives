"""Frame classification (Stage 2 of AFA pipeline)."""

import logging
from typing import Dict
from .llm_interface import LLMInterface
from .prompts import FRAME_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class FrameClassifier:
    """Classify article frames using predefined typology."""
    
    VALID_FRAMES = {
        "economic_opportunity",
        "economic_risk",
        "regulatory_compliance",
        "technological_solution",
        "environmental_urgency",
        "social_responsibility",
        "market_dynamics",
        "uncertainty_skepticism",
    }
    
    def __init__(self, llm: LLMInterface, max_tokens: int = 512):
        """
        Initialize frame classifier.
        
        Args:
            llm: LLM interface
            max_tokens: Max tokens for classification
        """
        self.llm = llm
        self.max_tokens = max_tokens
    
    def classify(self, article: Dict[str, str]) -> Dict:
        """
        Classify article frames.
        
        Args:
            article: Dict with 'text' and 'headline'
            
        Returns:
            Dict with primary_frame, secondary_frame, justification
        """
        # Format prompt
        prompt = FRAME_CLASSIFICATION_PROMPT.format(
            headline=article.get("headline", ""),
            text=article["text"][:4000],
        )
        
        # Generate classification
        try:
            result = self.llm.generate_json(prompt, max_tokens=self.max_tokens)
            
            # Validate frames
            result = self._validate_frames(result)
            
            logger.debug(f"Classified as: {result.get('primary_frame')}")
            return result
        
        except Exception as e:
            logger.error(f"Frame classification failed: {e}")
            return {
                "primary_frame": None,
                "secondary_frame": None,
                "justification": "",
                "climate_connection": "",
            }
    
    def _validate_frames(self, result: Dict) -> Dict:
        """Validate and normalize frame classifications."""
        # Validate primary frame (required)
        primary = result.get("primary_frame", "").lower()
        if primary not in self.VALID_FRAMES:
            logger.warning(f"Invalid primary frame: {primary}")
            result["primary_frame"] = None
        else:
            result["primary_frame"] = primary
        
        # Validate secondary frame (optional)
        secondary = result.get("secondary_frame", "").lower()
        if secondary and secondary != "null" and secondary in self.VALID_FRAMES:
            result["secondary_frame"] = secondary
        else:
            result["secondary_frame"] = None
        
        return result
