"""
Temporal analysis for RQ1: Actor prominence over time.
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyze actor prominence and trends over time."""
    
    TEMPORAL_STRATA = {
        "pre-crisis": ("2000-01-01", "2007-12-31"),
        "financial-crisis": ("2008-01-01", "2012-12-31"),
        "post-crisis": ("2013-01-01", "2018-12-31"),
        "climate-surge": ("2019-01-01", "2023-12-31"),
    }
    
    def __init__(self, extractions: List[Dict]):
        """
        Initialize temporal analyzer.
        
        Args:
            extractions: List of AFA extractions with article metadata
        """
        self.extractions_df = pd.DataFrame(extractions)
        self.extractions_df['publication_date'] = pd.to_datetime(
            self.extractions_df['publication_date']
        )
        
        # Add temporal stratum
        self.extractions_df['stratum'] = self.extractions_df['publication_date'].apply(
            self._assign_stratum
        )
        
        logger.info(f"Initialized TemporalAnalyzer with {len(extractions)} extractions")
    
    def _assign_stratum(self, date):
        """Assign temporal stratum to date."""
        for stratum, (start, end) in self.TEMPORAL_STRATA.items():
            if pd.to_datetime(start) <= date <= pd.to_datetime(end):
                return stratum
        return "other"
    
    def analyze_actor_prominence(self) -> Dict:
        """
        RQ1: Analyze actor prominence over time.
        
        Returns:
            Dict with actor distribution by stratum and statistics
        """
        # Extract all actors with their types and strata
        actor_records = []
        
        for _, row in self.extractions_df.iterrows():
            stratum = row['stratum']
            actors = row.get('actors', [])
            
            for actor in actors:
                actor_records.append({
                    'actor_type': actor.get('actor_type'),
                    'stratum': stratum,
                    'name': actor.get('name'),
                })
        
        actor_df = pd.DataFrame(actor_records)
        
        # Compute distribution by stratum
        actor_dist = pd.crosstab(
            actor_df['stratum'],
            actor_df['actor_type'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        # Sort strata chronologically
        stratum_order = ["pre-crisis", "financial-crisis", "post-crisis", "climate-surge"]
        actor_dist = actor_dist.reindex(stratum_order)
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(
            actor_df['stratum'],
            actor_df['actor_type']
        )
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results = {
            "actor_distribution": actor_dist.to_dict(),
            "contingency_table": contingency_table.to_dict(),
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
        }
        
        logger.info(f"Actor prominence analysis: χ²={chi2:.1f}, p={p_value:.4f}")
        
        return results
    
    def compute_actor_trends(self) -> pd.DataFrame:
        """Compute quarterly trends for actor prominence."""
        # Extract actors with dates
        actor_records = []
        
        for _, row in self.extractions_df.iterrows():
            date = row['publication_date']
            actors = row.get('actors', [])
            
            for actor in actors:
                actor_records.append({
                    'date': date,
                    'actor_type': actor.get('actor_type'),
                    'quarter': date.to_period('Q'),
                })
        
        actor_df = pd.DataFrame(actor_records)
        
        # Compute quarterly proportions
        quarterly_counts = actor_df.groupby(['quarter', 'actor_type']).size().unstack(fill_value=0)
        quarterly_props = quarterly_counts.div(quarterly_counts.sum(axis=1), axis=0) * 100
        
        return quarterly_props
    
    def get_top_actors_by_period(self, top_n: int = 10) -> Dict[str, List]:
        """Get most mentioned actors in each period."""
        top_actors = {}
        
        for stratum in self.TEMPORAL_STRATA.keys():
            stratum_df = self.extractions_df[self.extractions_df['stratum'] == stratum]
            
            actor_counts = {}
            for _, row in stratum_df.iterrows():
                actors = row.get('actors', [])
                for actor in actors:
                    name = actor.get('name')
                    actor_counts[name] = actor_counts.get(name, 0) + 1
            
            # Sort and get top N
            sorted_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)
            top_actors[stratum] = sorted_actors[:top_n]
        
        return top_actors
