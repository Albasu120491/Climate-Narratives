"""
Argument analysis for RQ3: Actor-frame-argument alignment.
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from collections import Counter

logger = logging.getLogger(__name__)


class ArgumentAnalyzer:
    """Analyze argumentative strategies and actor-frame alignment."""
    
    def __init__(self, extractions: List[Dict]):
        """
        Initialize argument analyzer.
        
        Args:
            extractions: List of AFA extractions
        """
        self.extractions_df = pd.DataFrame(extractions)
        logger.info(f"Initialized ArgumentAnalyzer with {len(extractions)} extractions")
    
    def analyze_actor_frame_alignment(self) -> Dict:
        """
        RQ3: Analyze how actors use frames.
        
        Returns:
            Dict with actor-frame associations and statistics
        """
        # Extract actor-frame pairs
        pairs = []
        
        for _, row in self.extractions_df.iterrows():
            actors = row.get('actors', [])
            frames = row.get('frames', {})
            primary_frame = frames.get('primary_frame')
            
            if not primary_frame:
                continue
            
            for actor in actors:
                actor_type = actor.get('actor_type')
                if actor_type:
                    pairs.append({
                        'actor_type': actor_type,
                        'frame': primary_frame,
                        'stance': actor.get('stance'),
                    })
        
        if not pairs:
            return {}
        
        pairs_df = pd.DataFrame(pairs)
        
        # Contingency table
        contingency = pd.crosstab(pairs_df['actor_type'], pairs_df['frame'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Standardized residuals (indicates over/under-representation)
        residuals = (contingency - expected) / np.sqrt(expected)
        
        results = {
            "contingency_table": contingency.to_dict(),
            "standardized_residuals": residuals.to_dict(),
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
        }
        
        logger.info(f"Actor-frame alignment: χ²={chi2:.1f}, p={p_value:.4f}")
        
        return results
    
    def analyze_warrant_strategies(self) -> Dict:
        """Analyze warrant types by actor group."""
        warrant_by_actor = {}
        
        for _, row in self.extractions_df.iterrows():
            actors = row.get('actors', [])
            arguments = row.get('arguments', {})
            warrant = arguments.get('warrant', '')
            
            if not warrant:
                continue
            
            # Simple warrant classification (could be improved with LLM)
            warrant_type = self._classify_warrant(warrant)
            
            for actor in actors:
                actor_type = actor.get('actor_type')
                if actor_type not in warrant_by_actor:
                    warrant_by_actor[actor_type] = []
                warrant_by_actor[actor_type].append(warrant_type)
        
        # Compute distributions
        warrant_distributions = {}
        for actor_type, warrants in warrant_by_actor.items():
            warrant_counts = Counter(warrants)
            total = sum(warrant_counts.values())
            warrant_distributions[actor_type] = {
                k: v / total * 100 for k, v in warrant_counts.items()
            }
        
        return warrant_distributions
    
    def _classify_warrant(self, warrant: str) -> str:
        """Classify warrant type based on keywords."""
        warrant_lower = warrant.lower()
        
        # Simple keyword-based classification
        if any(kw in warrant_lower for kw in ['competitive', 'advantage', 'market leadership', 'growth']):
            return 'market_opportunity'
        elif any(kw in warrant_lower for kw in ['risk', 'loss', 'liability', 'stranded']):
            return 'risk_mitigation'
        elif any(kw in warrant_lower for kw in ['regulation', 'compliance', 'law', 'policy']):
            return 'regulatory_necessity'
        elif any(kw in warrant_lower for kw in ['moral', 'ethical', 'responsibility', 'duty']):
            return 'moral_imperative'
        elif any(kw in warrant_lower for kw in ['urgent', 'crisis', 'emergency', 'damage']):
            return 'environmental_urgency'
        elif any(kw in warrant_lower for kw in ['innovation', 'technology', 'solution', 'r&d']):
            return 'technological_progress'
        else:
            return 'other'
    
    def compute_argument_complexity(self) -> Dict:
        """Compute argument complexity metrics."""
        complexity_by_frame = {}
        
        for _, row in self.extractions_df.iterrows():
            frames = row.get('frames', {})
            arguments = row.get('arguments', {})
            primary_frame = frames.get('primary_frame')
            
            if not primary_frame:
                continue
            
            # Count evidence pieces as proxy for complexity
            evidence = arguments.get('evidence', [])
            n_evidence = len(evidence) if isinstance(evidence, list) else 0
            
            if primary_frame not in complexity_by_frame:
                complexity_by_frame[primary_frame] = []
            complexity_by_frame[primary_frame].append(n_evidence)
        
        # Compute statistics
        complexity_stats = {}
        for frame, complexities in complexity_by_frame.items():
            complexity_stats[frame] = {
                "mean": float(np.mean(complexities)),
                "median": float(np.median(complexities)),
                "std": float(np.std(complexities)),
            }
        
        return complexity_stats
