"""Dictionary-based frame classification baseline."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DictionaryBaseline:
    """Dictionary-based frame classification (Card et al. 2015 style)."""
    
    # Frame dictionaries (simplified - actual would be more comprehensive)
    FRAME_DICTIONARIES = {
        "economic_opportunity": [
            "growth", "profit", "investment", "opportunity", "returns",
            "competitive advantage", "market leader", "innovation", "revenue",
        ],
        "economic_risk": [
            "loss", "cost", "liability", "risk", "stranded assets",
            "financial burden", "expensive", "damage", "threat",
        ],
        "regulatory_compliance": [
            "regulation", "compliance", "law", "policy", "mandate",
            "requirement", "government", "SEC", "legislation",
        ],
        "technological_solution": [
            "innovation", "technology", "solution", "R&D", "breakthrough",
            "clean tech", "renewable", "efficient", "advanced",
        ],
        "environmental_urgency": [
            "crisis", "urgent", "emergency", "damage", "catastrophe",
            "extinction", "irreversible", "tipping point", "collapse",
        ],
        "social_responsibility": [
            "responsibility", "ethical", "moral", "duty", "commitment",
            "stakeholder", "ESG", "sustainable", "corporate citizenship",
        ],
        "market_dynamics": [
            "competition", "market share", "supply", "demand", "pricing",
            "positioning", "strategy", "competitive", "market forces",
        ],
        "uncertainty_skepticism": [
            "uncertain", "doubt", "skeptical", "unclear", "disputed",
            "questionable", "unproven", "controversy", "debate",
        ],
    }
    
    def __init__(self):
        """Initialize dictionary baseline."""
        logger.info("Initialized DictionaryBaseline")
    
    def classify(self, text: str) -> str:
        """Classify frame using dictionary matching."""
        text_lower = text.lower()
        
        frame_scores = {}
        
        for frame, keywords in self.FRAME_DICTIONARIES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            frame_scores[frame] = score
        
        # Return frame with highest score
        if max(frame_scores.values()) == 0:
            return None
        
        return max(frame_scores, key=frame_scores.get)
