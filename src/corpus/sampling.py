"""
4-stage hierarchical sampling strategy:
1. Temporal stratification
2. Thematic clustering
3. MMR selection
4. Active learning enrichment
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CorpusSampler:
    """4-stage hierarchical sampling for representative corpus subset."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        lambda_mmr: float = 0.7,
        min_clusters: int = 5,
        max_clusters: int = 15,
    ):
        """
        Initialize sampler.
        
        Args:
            embedding_model: SentenceTransformer model name
            lambda_mmr: MMR lambda parameter (0=diversity, 1=relevance)
            min_clusters: Minimum clusters per stratum
            max_clusters: Maximum clusters per stratum
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.lambda_mmr = lambda_mmr
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        logger.info(f"Initialized CorpusSampler with {embedding_model}")
    
    def sample(
        self,
        corpus_df: pd.DataFrame,
        target_size: int = 4143,
        temporal_strata: Dict[str, Tuple[str, str]] = None,
        use_active_learning: bool = True,
    ) -> pd.DataFrame:
        """
        Execute full 4-stage sampling pipeline.
        
        Args:
            corpus_df: Full corpus DataFrame
            target_size: Target sample size
            temporal_strata: Dict mapping stratum names to (start_date, end_date)
            use_active_learning: Whether to apply active learning enrichment
            
        Returns:
            Sampled DataFrame
        """
        logger.info(f"Starting 4-stage sampling: {len(corpus_df)} -> {target_size}")
        
        # Default temporal strata from paper
        if temporal_strata is None:
            temporal_strata = {
                "pre-crisis": ("2000-01-01", "2007-12-31"),
                "financial-crisis": ("2008-01-01", "2012-12-31"),
                "post-crisis": ("2013-01-01", "2018-12-31"),
                "climate-surge": ("2019-01-01", "2023-12-31"),
            }
        
        # Stage 1: Temporal stratification
        stratified = self._temporal_stratification(corpus_df, temporal_strata, target_size)
        
        sampled_articles = []
        
        for stratum_name, stratum_df in stratified.items():
            logger.info(f"Processing stratum: {stratum_name} ({len(stratum_df)} articles)")
            
            # Stage 2: Thematic clustering
            clusters = self._thematic_clustering(stratum_df)
            
            # Stage 3: MMR selection
            sampled = self._mmr_selection(stratum_df, clusters)
            sampled_articles.append(sampled)
        
        # Combine all strata
        sample_df = pd.concat(sampled_articles, ignore_index=True)
        
        # Stage 4: Active learning enrichment (optional)
        if use_active_learning:
            sample_df = self._active_learning_enrichment(corpus_df, sample_df, target_size)
        
        logger.info(f"Sampling complete: {len(sample_df)} articles")
        return sample_df
    
    def _temporal_stratification(
        self,
        corpus_df: pd.DataFrame,
        strata: Dict[str, Tuple[str, str]],
        target_size: int,
    ) -> Dict[str, pd.DataFrame]:
        """Stage 1: Stratify by temporal periods."""
        corpus_df['publication_date'] = pd.to_datetime(corpus_df['publication_date'])
        
        stratified = {}
        total_size = len(corpus_df)
        
        for stratum_name, (start, end) in strata.items():
            mask = (corpus_df['publication_date'] >= start) & (corpus_df['publication_date'] <= end)
            stratum_df = corpus_df[mask].copy()
            
            # Proportional allocation
            stratum_df['stratum'] = stratum_name
            stratum_df['stratum_size'] = int(target_size * len(stratum_df) / total_size)
            
            stratified[stratum_name] = stratum_df
            logger.info(f"  {stratum_name}: {len(stratum_df)} articles -> target {stratum_df['stratum_size'].iloc[0]}")
        
        return stratified
    
    def _thematic_clustering(self, stratum_df: pd.DataFrame) -> np.ndarray:
        """Stage 2: Cluster by thematic similarity."""
        # Create weighted embeddings (2:1 headline:first_para)
        texts = []
        for _, row in stratum_df.iterrows():
            text = f"{row['headline']} {row['headline']} {row.get('first_paragraph', '')}"
            texts.append(text)
        
        logger.info(f"  Encoding {len(texts)} articles...")
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        
        # Find optimal cluster count via silhouette score
        n_samples = len(stratum_df)
        max_k = min(self.max_clusters, n_samples // 20)
        
        if max_k < self.min_clusters:
            # Too few samples for clustering
            return np.zeros(n_samples, dtype=int)
        
        best_score = -1
        best_k = self.min_clusters
        
        for k in range(self.min_clusters, max_k + 1):
            clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clustering.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Final clustering with best k
        clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
        labels = clustering.fit_predict(embeddings)
        
        logger.info(f"  Optimal clusters: {best_k} (silhouette={best_score:.3f})")
        
        return labels
    
    def _mmr_selection(
        self,
        stratum_df: pd.DataFrame,
        cluster_labels: np.ndarray,
    ) -> pd.DataFrame:
        """Stage 3: Select representatives via MMR."""
        stratum_df = stratum_df.copy()
        stratum_df['cluster'] = cluster_labels
        target_size = stratum_df['stratum_size'].iloc[0]
        
        # Get embeddings
        texts = []
        for _, row in stratum_df.iterrows():
            text = f"{row['headline']} {row.get('first_paragraph', '')}"
            texts.append(text)
        
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        stratum_df['embedding'] = list(embeddings)
        
        selected_indices = []
        
        # Sample from each cluster proportionally
        for cluster_id in stratum_df['cluster'].unique():
            cluster_df = stratum_df[stratum_df['cluster'] == cluster_id]
            cluster_target = max(1, int(target_size * len(cluster_df) / len(stratum_df)))
            
            # Compute cluster centroid
            cluster_embeds = np.array(cluster_df['embedding'].tolist())
            centroid = cluster_embeds.mean(axis=0)
            
            # MMR selection
            selected = []
            candidates = list(cluster_df.index)
            
            for _ in range(min(cluster_target, len(candidates))):
                best_score = -np.inf
                best_idx = None
                
                for idx in candidates:
                    if idx in selected:
                        continue
                    
                    embed = embeddings[idx]
                    
                    # Relevance: similarity to centroid
                    relevance = 1 - cosine(embed, centroid)
                    
                    # Diversity: max similarity to already selected
                    if selected:
                        selected_embeds = embeddings[selected]
                        similarities = [1 - cosine(embed, s_emb) for s_emb in selected_embeds]
                        diversity = -max(similarities)
                    else:
                        diversity = 0
                    
                    # MMR score
                    score = self.lambda_mmr * relevance + (1 - self.lambda_mmr) * diversity
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(best_idx)
            
            selected_indices.extend(selected)
        
        sampled = stratum_df.loc[selected_indices].copy()
        logger.info(f"  Selected {len(sampled)} articles via MMR")
        
        return sampled
    
    def _active_learning_enrichment(
        self,
        corpus_df: pd.DataFrame,
        sample_df: pd.DataFrame,
        target_size: int,
    ) -> pd.DataFrame:
        """Stage 4: Enrich with high-uncertainty cases."""
        # Simplified version - in practice, would use trained classifier
        # For now, randomly add 10% more complex cases
        
        current_size = len(sample_df)
        if current_size >= target_size:
            return sample_df
        
        additional_needed = target_size - current_size
        
        # Get articles not in sample
        sampled_ids = set(sample_df['article_id'])
        remaining = corpus_df[~corpus_df['article_id'].isin(sampled_ids)]
        
        # Heuristic: select longer articles (proxy for complexity)
        remaining['text_length'] = remaining['text'].str.len()
        enrichment = remaining.nlargest(additional_needed, 'text_length')
        
        combined = pd.concat([sample_df, enrichment], ignore_index=True)
        logger.info(f"  Active learning: added {len(enrichment)} high-complexity articles")
        
        return combined


def stratified_sampling(
    corpus_df: pd.DataFrame,
    strata_column: str,
    target_size: int,
) -> pd.DataFrame:
    """Simple stratified sampling by a categorical column."""
    sampled = corpus_df.groupby(strata_column, group_keys=False).apply(
        lambda x: x.sample(frac=target_size / len(corpus_df), random_state=42)
    )
    return sampled


def mmr_selection(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    n_select: int,
    lambda_param: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance selection.
    
    Args:
        embeddings: Document embeddings (n_docs, dim)
        centroid: Cluster centroid (dim,)
        n_select: Number to select
        lambda_param: Relevance vs diversity trade-off
        
    Returns:
        List of selected indices
    """
    selected = []
    candidates = list(range(len(embeddings)))
    
    for _ in range(min(n_select, len(candidates))):
        best_score = -np.inf
        best_idx = None
        
        for idx in candidates:
            if idx in selected:
                continue
            
            # Relevance
            relevance = 1 - cosine(embeddings[idx], centroid)
            
            # Diversity
            if selected:
                similarities = [1 - cosine(embeddings[idx], embeddings[s]) for s in selected]
                diversity = -max(similarities)
            else:
                diversity = 0
            
            score = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
    
    return selected
