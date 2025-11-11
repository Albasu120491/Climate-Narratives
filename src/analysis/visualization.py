"""Visualization utilities for paper figures."""

import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:

    
    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("colorblind")
        self.figsize = (10, 6)
    
    def plot_actor_prominence(
        self,
        actor_distribution: Dict,
        output_path: Optional[str] = None,
    ):
        """Generate Figure 2: Actor prominence over time."""
        df = pd.DataFrame(actor_distribution).T
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        df.plot(kind='line', marker='o', ax=ax)
        
        ax.set_xlabel('Temporal Stratum', fontsize=12)
        ax.set_ylabel('Share of Actors (%)', fontsize=12)
        ax.set_title('Actor Prominence Over Time', fontsize=14)
        ax.legend(title='Actor Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_frame_transformation(
        self,
        frame_distribution: Dict,
        changepoint_date: str,
        output_path: Optional[str] = None,
    ):
        """Generate Figure 3: Frame transformation over time."""
        df = pd.DataFrame(frame_distribution).T
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        df.plot(kind='line', marker='s', ax=ax)
        
        # Add changepoint line
        if changepoint_date:
            ax.axvline(x=changepoint_date, color='red', linestyle='--', 
                      label='Changepoint', alpha=0.7)
        
        ax.set_xlabel('Temporal Stratum', fontsize=12)
        ax.set_ylabel('Share of Frames (%)', fontsize=12)
        ax.set_title('Temporal Dynamics of Climate Frames', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_actor_frame_heatmap(
        self,
        residuals: Dict,
        output_path: Optional[str] = None,
    ):
        """Generate heatmap of actor-frame associations."""
        df = pd.DataFrame(residuals)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            df,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Standardized Residual'},
            ax=ax,
        )
        
        ax.set_title('Actor-Frame Association Matrix', fontsize=14)
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Actor Type', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
        else:
            plt.show()
        
        plt.close()
