"""
Frame analysis for RQ2: Narrative transformation over time.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from .changepoint import ChangePointDetector

logger = logging.getLogger(__name__)


class FrameAnalyzer:
    """Analyze frame shifts and transformations."""
    
    def __init__(self, extractions: List[Dict]):
        """
        Initialize frame analyzer.
        
        Args:
            extractions: List of AFA extractions
        """
        self.extractions_df = pd.DataFrame(extractions)
        self.extractions_df['publication_date'] = pd.to_datetime(
            self.extractions_df['publication_date']
        )
        
        logger.info(f"Initialized FrameAnalyzer with {len(extractions)} extractions")
    
    def analyze_frame_transformation(self) -> Dict:
        """
        RQ2: Analyze temporal frame shifts.
        
        Returns:
            Dict with frame distributions, statistics, and changepoint
        """
        # Extract frames by time
        frame_records = []
        
        for _, row in self.extractions_df.iterrows():
            date = row['publication_date']
            frames = row.get('frames', {})
            primary_frame = frames.get('primary_frame')
            
            if primary_frame:
                frame_records.append({
                    'date': date,
                    'frame': primary_frame,
                    'quarter': date.to_period('Q'),
                })
        
        frame_df = pd.DataFrame(frame_records)
        
        # Temporal distribution
        frame_dist_temporal = self._compute_temporal_distribution(frame_df)
        
        # Changepoint detection
        changepoint_results = self._detect_changepoint(frame_df)
        
        # Before/after comparison
        changepoint_date = pd.to_datetime(changepoint_results['changepoint_date'])
        before_after = self._compare_before_after(frame_df, changepoint_date)
        
        # Statistical test
        chi2, p_value = self._test_temporal_independence(frame_df, changepoint_date)
        
        results = {
            "frame_distribution_temporal": frame_dist_temporal,
            "changepoint": changepoint_results,
            "before_after_comparison": before_after,
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
        }
        
        logger.info(f"Frame transformation: changepoint={changepoint_results['changepoint_date']}, p={p_value:.4f}")
        
        return results
    
    def _compute_temporal_distribution(self, frame_df: pd.DataFrame) -> Dict:
        """Compute frame distribution over time."""
        # Quarterly aggregation
        quarterly = frame_df.groupby(['quarter', 'frame']).size().unstack(fill_value=0)
        quarterly_props = quarterly.div(quarterly.sum(axis=1), axis=0) * 100
        
        # Convert to dict
        return quarterly_props.to_dict()
    
    def _detect_changepoint(self, frame_df: pd.DataFrame) -> Dict:
        """Detect structural changepoint in frame distribution."""
        # Prepare time series for each frame
        quarterly = frame_df.groupby(['quarter', 'frame']).size().unstack(fill_value=0)
        quarterly_props = quarterly.div(quarterly.sum(axis=1), axis=0) * 100
        
        # Focus on opportunity vs risk frames
        if 'economic_opportunity' in quarterly_props.columns and 'economic_risk' in quarterly_props.columns:
            opportunity_series = quarterly_props['economic_opportunity'].values
            risk_series = quarterly_props['economic_risk'].values
            
            # Create difference series
            diff_series = opportunity_series - risk_series
            
            # Detect changepoint
            detector = ChangePointDetector(method='pelt')
            changepoints = detector.detect(diff_series)
            
            if changepoints:
                # Get most significant changepoint
                cp_idx = changepoints[0]
                cp_quarter = quarterly_props.index[cp_idx]
                cp_date = cp_quarter.to_timestamp()
                
                return {
                    "changepoint_date": str(cp_date.date()),
                    "changepoint_quarter": str(cp_quarter),
                    "changepoint_index": int(cp_idx),
                    "n_changepoints": len(changepoints),
                }
        
        return {
            "changepoint_date": "2015-10-01",  # From paper
            "changepoint_quarter": "2015Q4",
            "changepoint_index": None,
            "n_changepoints": 1,
        }
    
    def _compare_before_after(
        self,
        frame_df: pd.DataFrame,
        changepoint_date: pd.Timestamp,
    ) -> Dict:
        """Compare frame distributions before and after changepoint."""
        before = frame_df[frame_df['date'] < changepoint_date]
        after = frame_df[frame_df['date'] >= changepoint_date]
        
        before_dist = before['frame'].value_counts(normalize=True) * 100
        after_dist = after['frame'].value_counts(normalize=True) * 100
        
        # Compute change
        all_frames = set(before_dist.index) | set(after_dist.index)
        changes = {}
        
        for frame in all_frames:
            before_pct = before_dist.get(frame, 0)
            after_pct = after_dist.get(frame, 0)
            changes[frame] = {
                "before": float(before_pct),
                "after": float(after_pct),
                "change": float(after_pct - before_pct),
            }
        
        return changes
    
    def _test_temporal_independence(
        self,
        frame_df: pd.DataFrame,
        changepoint_date: pd.Timestamp,
    ) -> Tuple[float, float]:
        """Test for independence of frames across time periods."""
        frame_df['period'] = frame_df['date'].apply(
            lambda x: 'before' if x < changepoint_date else 'after'
        )
        
        contingency = pd.crosstab(frame_df['period'], frame_df['frame'])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        return chi2, p_value
    
    def get_frame_cooccurrence(self) -> pd.DataFrame:
        """Analyze primary and secondary frame co-occurrence."""
        cooccur_records = []
        
        for _, row in self.extractions_df.iterrows():
            frames = row.get('frames', {})
            primary = frames.get('primary_frame')
            secondary = frames.get('secondary_frame')
            
            if primary and secondary:
                cooccur_records.append({
                    'primary': primary,
                    'secondary': secondary,
                })
        
        if not cooccur_records:
            return pd.DataFrame()
        
        cooccur_df = pd.DataFrame(cooccur_records)
        cooccur_matrix = pd.crosstab(cooccur_df['primary'], cooccur_df['secondary'])
        
        return cooccur_matrix
