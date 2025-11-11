"""Actor and stance extraction (Stage 1 of AFA pipeline)."""

import logging
from typing import Dict, List
from .llm_interface import LLMInterface
from .prompts import ACTOR_STANCE_PROMPT

logger = logging.getLogger(__name__)


class ActorStanceExtractor:
    """Extract actors and their stances on climate issues."""
    
    VALID_ACTOR_TYPES = {
        "company",
        "financial_institution",
        "government",
        "ngo",
        "individual",
    }
    
    VALID_STANCES = {
        "supportive",
        "opposing",
        "neutral",
        "mixed",
    }
    
    def __init__(self, llm: LLMInterface, max_tokens: int = 512):
        """
        Initialize actor-stance extractor.
        
        Args:
            llm: LLM interface
            max_tokens: Max tokens for extraction
        """
        self.llm = llm
        self.max_tokens = max_tokens
    
    def extract(self, article: Dict[str, str]) -> List[Dict]:
        """
        Extract actors and stances from article.
        
        Args:
            article: Dict with 'text' and 'headline'
            
        Returns:
            List of actor dicts with name, type, stance, quote, relevance
        """
        # Format prompt
        prompt = ACTOR_STANCE_PROMPT.format(
            headline=article.get("headline", ""),
            text=article["text"][:4000],  # Truncate for token limits
        )
        
        # Generate extraction
        try:
            result = self.llm.generate_json(prompt, max_tokens=self.max_tokens)
            actors = result.get("actors", [])
            
            # Validate and clean
            actors = self._validate_actors(actors)
            
            logger.debug(f"Extracted {len(actors)} actors")
            return actors
        
        except Exception as e:
            logger.error(f"Actor extraction failed: {e}")
            return []
    
    def _validate_actors(self, actors: List[Dict]) -> List[Dict]:
        """Validate and normalize actor extractions."""
        validated = []
        
        for actor in actors:
            # Check required fields
            if not all(k in actor for k in ["name", "actor_type", "stance"]):
                logger.warning(f"Skipping incomplete actor: {actor}")
                continue
            
            # Normalize actor type
            actor_type = actor["actor_type"].lower()
            if actor_type not in self.VALID_ACTOR_TYPES:
                logger.warning(f"Invalid actor type: {actor_type}, defaulting to 'company'")
                actor_type = "company"
            actor["actor_type"] = actor_type
            
            # Normalize stance
            stance = actor["stance"].lower()
            if stance not in self.VALID_STANCES:
                logger.warning(f"Invalid stance: {stance}, defaulting to 'neutral'")
                stance = "neutral"
            actor["stance"] = stance
            
            validated.append(actor)
        
        return validated
