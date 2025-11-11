"""AFA extraction pipeline components."""

from .actor_stance import ActorStanceExtractor
from .frame_classification import FrameClassifier
from .argument_extraction import ArgumentExtractor
from .llm_interface import LLMInterface
from .prompts import (
    ACTOR_STANCE_PROMPT,
    FRAME_CLASSIFICATION_PROMPT,
    ARGUMENT_EXTRACTION_PROMPT,
)

__all__ = [
    "ActorStanceExtractor",
    "FrameClassifier",
    "ArgumentExtractor",
    "LLMInterface",
    "AFAPipeline",
]


class AFAPipeline:
    """Complete Actor-Frame-Argument extraction pipeline."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        """
        Initialize AFA pipeline.
        
        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Max tokens per response
        """
        self.llm = LLMInterface(model=model, temperature=temperature)
        
        self.actor_extractor = ActorStanceExtractor(self.llm, max_tokens=max_tokens)
        self.frame_classifier = FrameClassifier(self.llm, max_tokens=max_tokens)
        self.argument_extractor = ArgumentExtractor(self.llm, max_tokens=768)
    
    def extract(self, article: Dict[str, str]) -> Dict:
        """
        Run full AFA extraction on article.
        
        Args:
            article: Dict with 'text' and 'headline' keys
            
        Returns:
            Dict with actors, frames, arguments
        """
        # Stage 1: Actor-Stance
        actors = self.actor_extractor.extract(article)
        
        # Stage 2: Frames
        frames = self.frame_classifier.classify(article)
        
        # Stage 3: Arguments
        arguments = self.argument_extractor.extract(article)
        
        return {
            "actors": actors,
            "frames": frames,
            "arguments": arguments,
            "article_id": article.get("article_id"),
        }
