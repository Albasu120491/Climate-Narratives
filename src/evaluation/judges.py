"""Multi-judge evaluation utilities."""

import logging
from typing import List, Dict, Callable
import numpy as np

logger = logging.getLogger(__name__)


class Judge:
    """Single judge for evaluation."""
    
    def __init__(
        self,
        name: str,
        score_function: Callable,
    ):
        """
        Initialize judge.
        
        Args:
            name: Judge identifier
            score_function: Function that returns score dict
        """
        self.name = name
        self.score_function = score_function
    
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Run evaluation."""
        return self.score_function(*args, **kwargs)


class MultiJudgeEvaluator:
    """Aggregate scores from multiple judges."""
    
    def __init__(self, judges: List[Judge]):
        """
        Initialize multi-judge evaluator.
        
        Args:
            judges: List of Judge objects
        """
        self.judges = judges
        logger.info(f"Initialized MultiJudgeEvaluator with {len(judges)} judges")
    
    def evaluate(
        self,
        *args,
        aggregation: str = "mean",
        **kwargs
    ) -> Dict[str, float]:
        """
        Aggregate evaluations from all judges.
        
        Args:
            aggregation: Aggregation method ('mean', 'median', 'max', 'min')
            
        Returns:
            Aggregated scores
        """
        all_scores = defaultdict(list)
        
        # Collect scores from all judges
        for judge in self.judges:
            try:
                scores = judge.evaluate(*args, **kwargs)
                for key, value in scores.items():
                    all_scores[key].append(value)
            except Exception as e:
                logger.error(f"Judge {judge.name} failed: {e}")
        
        # Aggregate
        aggregated = {}
        for key, values in all_scores.items():
            if aggregation == "mean":
                aggregated[key] = float(np.mean(values))
            elif aggregation == "median":
                aggregated[key] = float(np.median(values))
            elif aggregation == "max":
                aggregated[key] = float(np.max(values))
            elif aggregation == "min":
                aggregated[key] = float(np.min(values))
            else:
                aggregated[key] = float(np.mean(values))
        
        return aggregated
