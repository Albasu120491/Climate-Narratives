"""Validation utilities for corpus construction."""

import logging
from typing import List, Dict, Set
import pandas as pd
import re

logger = logging.getLogger(__name__)


# Climate keyword lexicon from paper (Appendix A.3)
CLIMATE_KEYWORDS = {
    "general": [
        "climate change", "global warming", "greenhouse effect",
        "climate crisis", "climate emergency", "climate action",
    ],
    "carbon": [
        "carbon", "CO2", "carbon dioxide", "carbon tax",
        "carbon capture", "carbon neutral", "carbon footprint",
    ],
    "energy": [
        "renewables", "renewable energy", "solar", "wind",
        "fossil fuels", "biofuels", "clean energy",
    ],
    "finance": [
        "ESG", "green bonds", "carbon markets", "sustainable finance",
        "climate risk", "stranded assets",
    ],
    "policy": [
        "Paris Agreement", "Kyoto Protocol", "net zero",
        "emissions reduction", "carbon pricing",
    ],
}


def keyword_validator(article_text: str, keywords: Dict[str, List[str]] = None) -> bool:
    """
    Check if article contains climate-related keywords.
    
    Args:
        article_text: Article text
        keywords: Dict of keyword categories
        
    Returns:
        True if article contains climate keywords
    """
    if keywords is None:
        keywords = CLIMATE_KEYWORDS
    
    text_lower = article_text.lower()
    
    for category, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword.lower() in text_lower:
                return True
    
    return False


def validate_djid_filtering(
    corpus_df: pd.DataFrame,
    sample_size: int = 1000,
) -> Dict[str, float]:
    """
    Validate DJID filtering using keyword-based re-screening.
    
    Args:
        corpus_df: Corpus filtered by DJID
        sample_size: Sample size for validation
        
    Returns:
        Dict with precision/recall estimates
    """
    logger.info(f"Validating DJID filtering on {sample_size} samples")
    
    # Sample articles
    sample = corpus_df.sample(min(sample_size, len(corpus_df)), random_state=42)
    
    # Check keyword matches
    keyword_matches = sample['text'].apply(keyword_validator)
    
    # Precision: what fraction of DJID-filtered articles are truly climate-related
    precision = keyword_matches.sum() / len(sample)
    
    logger.info(f"  Keyword validation precision: {precision:.3f}")
    
    return {
        "precision": precision,
        "sample_size": len(sample),
        "keyword_matches": int(keyword_matches.sum()),
    }


def validate_sample_distribution(
    sample_df: pd.DataFrame,
    population_df: pd.DataFrame,
    columns: List[str] = None,
) -> Dict[str, float]:
    """
    Validate that sample preserves population distributions.
    
    Args:
        sample_df: Sampled corpus
        population_df: Full corpus
        columns: Columns to validate (e.g., temporal, thematic)
        
    Returns:
        Dict with validation metrics (JSD, KL divergence)
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy
    
    if columns is None:
        columns = ['stratum', 'primary_djid']
    
    metrics = {}
    
    for col in columns:
        if col not in sample_df.columns or col not in population_df.columns:
            continue
        
        # Compute distributions
        sample_dist = sample_df[col].value_counts(normalize=True).sort_index()
        pop_dist = population_df[col].value_counts(normalize=True).sort_index()
        
        # Align indices
        all_categories = sample_dist.index.union(pop_dist.index)
        sample_dist = sample_dist.reindex(all_categories, fill_value=0)
        pop_dist = pop_dist.reindex(all_categories, fill_value=0)
        
        # Jensen-Shannon divergence
        jsd = jensenshannon(sample_dist.values, pop_dist.values)
        
        metrics[f"jsd_{col}"] = float(jsd)
        logger.info(f"  {col} JSD: {jsd:.4f}")
    
    return metrics
