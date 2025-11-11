"""Changepoint detection for time series analysis."""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ChangePointDetector:
    """Detect changepoints in time series using PELT algorithm."""
    
    def __init__(
        self,
        method: str = 'pelt',
        penalty: Optional[float] = None,
    ):
        """
        Initialize changepoint detector.
        
        Args:
            method: Detection method ('pelt', 'binseg', 'window')
            penalty: Penalty value for changepoint detection
        """
        self.method = method
        self.penalty = penalty
        
        try:
            import ruptures as rpt
            self.rpt = rpt
        except ImportError:
            logger.error("ruptures library not installed")
            raise ImportError("Install ruptures: pip install ruptures")
    
    def detect(
        self,
        signal: np.ndarray,
        min_size: int = 2,
        max_changepoints: int = 10,
    ) -> List[int]:
        """
        Detect changepoints in signal.
        
        Args:
            signal: Time series signal
            min_size: Minimum segment size
            max_changepoints: Maximum number of changepoints
            
        Returns:
            List of changepoint indices
        """
        if len(signal) < min_size * 2:
            logger.warning("Signal too short for changepoint detection")
            return []
        
        # Auto-tune penalty if not provided
        if self.penalty is None:
            penalty = np.log(len(signal)) * len(signal) * np.var(signal)
        else:
            penalty = self.penalty
        
        # Run PELT
        if self.method == 'pelt':
            algo = self.rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
            changepoints = algo.predict(pen=penalty)
        elif self.method == 'binseg':
            algo = self.rpt.Binseg(model="rbf", min_size=min_size).fit(signal)
            changepoints = algo.predict(n_bkps=max_changepoints)
        else:
            algo = self.rpt.Window(width=10, model="rbf", min_size=min_size).fit(signal)
            changepoints = algo.predict(n_bkps=max_changepoints)
        
        # Remove final point (end of signal)
        if changepoints and changepoints[-1] == len(signal):
            changepoints = changepoints[:-1]
        
        logger.info(f"Detected {len(changepoints)} changepoints: {changepoints}")
        
        return changepoints
