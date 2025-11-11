"""
Decompositional Verification Framework (DVF).
Evaluates extractions across four dimensions:
- Completeness
- Faithfulness
- Coherence
- Climate Relevance
"""

import logging
from typing import Dict, List, Optional
import numpy as np
from .judges import MultiJudgeEvaluator, Judge
from ..extraction.llm_interface import LLMInterface
from ..extraction.prompts import (
    DVF_COMPLETENESS_PROMPT,
    DVF_FAITHFULNESS_PROMPT,
    DVF_COHERENCE_PROMPT,
    DVF_RELEVANCE_PROMPT,
)

logger = logging.getLogger(__name__)


class DecompositionalVerifier:
    """Single-judge DVF verification."""
    
    def __init__(self, llm: LLMInterface):
        """
        Initialize verifier.
        
        Args:
            llm: LLM interface for judge
        """
        self.llm = llm
    
    def verify_completeness(
        self,
        article: Dict,
        extraction: Dict,
    ) -> Dict[str, float]:
        """Verify completeness of extraction."""
        prompt = DVF_COMPLETENESS_PROMPT.format(
            article_text=article["text"][:4000],
            extraction_json=str(extraction),
        )
        
        try:
            result = self.llm.generate_json(prompt, max_tokens=256)
            scores = result.get("completeness", {})
            
            # Ensure all dimensions present
            for dim in ["actors", "stance", "frames", "arguments"]:
                if dim not in scores:
                    scores[dim] = 0.0
                scores[dim] = float(scores[dim])
            
            return scores
        
        except Exception as e:
            logger.error(f"Completeness verification failed: {e}")
            return {
                "actors": 0.0,
                "stance": 0.0,
                "frames": 0.0,
                "arguments": 0.0,
            }
    
    def verify_faithfulness(
        self,
        article: Dict,
        extraction: Dict,
    ) -> Dict[str, float]:
        """Verify faithfulness to source."""
        prompt = DVF_FAITHFULNESS_PROMPT.format(
            article_text=article["text"][:4000],
            extraction_json=str(extraction),
        )
        
        try:
            result = self.llm.generate_json(prompt, max_tokens=256)
            scores = result.get("faithfulness", {})
            
            for dim in ["quote_alignment", "paraphrase_equivalence", "no_hallucination"]:
                if dim not in scores:
                    scores[dim] = 0.0
                scores[dim] = float(scores[dim])
            
            return scores
        
        except Exception as e:
            logger.error(f"Faithfulness verification failed: {e}")
            return {
                "quote_alignment": 0.0,
                "paraphrase_equivalence": 0.0,
                "no_hallucination": 0.0,
            }
    
    def verify_coherence(self, extraction: Dict) -> Dict[str, float]:
        """Verify structural coherence."""
        prompt = DVF_COHERENCE_PROMPT.format(
            extraction_json=str(extraction),
        )
        
        try:
            result = self.llm.generate_json(prompt, max_tokens=256)
            scores = result.get("coherence", {})
            
            for dim in ["schema_wellformed", "actor_frame_links", "frame_argument_links", "internal_consistency"]:
                if dim not in scores:
                    scores[dim] = 0.0
                scores[dim] = float(scores[dim])
            
            return scores
        
        except Exception as e:
            logger.error(f"Coherence verification failed: {e}")
            return {
                "schema_wellformed": 0.0,
                "actor_frame_links": 0.0,
                "frame_argument_links": 0.0,
                "internal_consistency": 0.0,
            }
    
    def verify_relevance(
        self,
        article: Dict,
        extraction: Dict,
    ) -> Dict[str, float]:
        """Verify climate relevance."""
        prompt = DVF_RELEVANCE_PROMPT.format(
            article_text=article["text"][:4000],
            extraction_json=str(extraction),
        )
        
        try:
            result = self.llm.generate_json(prompt, max_tokens=256)
            scores = result.get("relevance", {})
            
            for dim in ["climate_focus", "peripheral_excluded", "frame_appropriateness"]:
                if dim not in scores:
                    scores[dim] = 0.0
                scores[dim] = float(scores[dim])
            
            return scores
        
        except Exception as e:
            logger.error(f"Relevance verification failed: {e}")
            return {
                "climate_focus": 0.0,
                "peripheral_excluded": 0.0,
                "frame_appropriateness": 0.0,
            }


class DVFEvaluator:
    """
    Multi-judge Decompositional Verification Framework.
    Aggregates scores from multiple judges for robustness.
    """
    
    def __init__(
        self,
        judges: Optional[List[str]] = None,
        aggregation: str = "mean",
    ):
        """
        Initialize DVF evaluator.
        
        Args:
            judges: List of judge model names (default: GPT-4o, Claude, Qwen3, Mixtral)
            aggregation: Aggregation method ('mean', 'median', 'vote')
        """
        if judges is None:
            judges = [
                "gpt-4o",
                "claude-sonnet-4",
                "qwen3-30b-a3b",
                "mixtral-8x22b",
            ]
        
        self.judges = []
        for judge_name in judges:
            try:
                llm = LLMInterface(model=judge_name, temperature=0.0)
                verifier = DecompositionalVerifier(llm)
                self.judges.append({
                    "name": judge_name,
                    "verifier": verifier,
                })
                logger.info(f"Initialized DVF judge: {judge_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize judge {judge_name}: {e}")
        
        if not self.judges:
            raise ValueError("No judges successfully initialized")
        
        self.aggregation = aggregation
    
    def evaluate(
        self,
        article: Dict,
        extraction: Dict,
        gold_standard: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Run full DVF evaluation.
        
        Args:
            article: Original article
            extraction: AFA extraction to evaluate
            gold_standard: Optional gold standard for calibration
            
        Returns:
            Dict with dimension-level scores
        """
        all_scores = {
            "completeness": [],
            "faithfulness": [],
            "coherence": [],
            "relevance": [],
        }
        
        # Collect scores from all judges
        for judge_info in self.judges:
            verifier = judge_info["verifier"]
            
            try:
                # Completeness
                comp_scores = verifier.verify_completeness(article, extraction)
                comp_avg = np.mean(list(comp_scores.values()))
                all_scores["completeness"].append(comp_avg)
                
                # Faithfulness
                faith_scores = verifier.verify_faithfulness(article, extraction)
                faith_avg = np.mean(list(faith_scores.values()))
                all_scores["faithfulness"].append(faith_avg)
                
                # Coherence
                coh_scores = verifier.verify_coherence(extraction)
                coh_avg = np.mean(list(coh_scores.values()))
                all_scores["coherence"].append(coh_avg)
                
                # Relevance
                rel_scores = verifier.verify_relevance(article, extraction)
                rel_avg = np.mean(list(rel_scores.values()))
                all_scores["relevance"].append(rel_avg)
            
            except Exception as e:
                logger.error(f"Judge {judge_info['name']} failed: {e}")
        
        # Aggregate across judges
        aggregated = {}
        for dimension, scores in all_scores.items():
            if scores:
                if self.aggregation == "mean":
                    aggregated[dimension] = float(np.mean(scores))
                elif self.aggregation == "median":
                    aggregated[dimension] = float(np.median(scores))
                else:
                    aggregated[dimension] = float(np.mean(scores))
            else:
                aggregated[dimension] = 0.0
        
        # Compare to gold standard if provided
        if gold_standard:
            aggregated["gold_comparison"] = self._compare_to_gold(
                extraction, gold_standard
            )
        
        return aggregated
    
    def _compare_to_gold(
        self,
        extraction: Dict,
        gold_standard: Dict,
    ) -> Dict[str, float]:
        """Compare extraction to gold standard."""
        from .metrics import compute_metrics
        
        metrics = compute_metrics(extraction, gold_standard)
        return metrics
    
    def calibrate(
        self,
        extractions: List[Dict],
        articles: List[Dict],
        human_scores: List[Dict],
    ) -> Dict[str, float]:
        """
        Calibrate DVF against human evaluations.
        
        Args:
            extractions: List of extractions
            articles: List of articles
            human_scores: List of human DVF scores
            
        Returns:
            Calibration metrics (correlation, MAE)
        """
        dvf_scores = []
        human_avg = []
        
        for extraction, article, human in zip(extractions, articles, human_scores):
            dvf = self.evaluate(article, extraction)
            dvf_avg = np.mean([dvf["completeness"], dvf["faithfulness"], 
                               dvf["coherence"], dvf["relevance"]])
            
            human_avg_score = np.mean([human["completeness"], human["faithfulness"],
                                       human["coherence"], human["relevance"]])
            
            dvf_scores.append(dvf_avg)
            human_avg.append(human_avg_score)
        
        # Compute correlation
        from scipy.stats import pearsonr, spearmanr
        
        pearson_r, pearson_p = pearsonr(dvf_scores, human_avg)
        spearman_r, spearman_p = spearmanr(dvf_scores, human_avg)
        
        # Mean absolute error
        mae = np.mean(np.abs(np.array(dvf_scores) - np.array(human_avg)))
        
        calibration = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "mae": float(mae),
            "n_samples": len(extractions),
        }
        
        logger.info(f"DVF Calibration: r={pearson_r:.3f}, MAE={mae:.3f}")
        
        return calibration
