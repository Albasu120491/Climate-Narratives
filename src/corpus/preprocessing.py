"""Article preprocessing and deduplication."""

import re
import logging
from typing import List, Set
import hashlib
import pandas as pd
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


class ArticlePreprocessor:
    """Preprocess articles: normalization, cleaning."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.boilerplate_patterns = [
            r"Copyright \d{4}.*",
            r"All rights reserved.*",
            r"Dow Jones & Company.*",
            r"\(c\) \d{4}.*",
        ]
    
    def preprocess(self, article_text: str) -> str:
        """
        Clean and normalize article text.
        
        Args:
            article_text: Raw article text
            
        Returns:
            Cleaned text
        """
        # Remove boilerplate
        text = article_text
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Unicode normalization
        text = text.strip()
        
        return text
    
    def extract_first_paragraph(self, article_text: str) -> str:
        """Extract first paragraph (useful for embeddings)."""
        paragraphs = article_text.split('\n\n')
        return paragraphs[0] if paragraphs else article_text[:500]


def deduplicate_corpus(
    articles_df: pd.DataFrame,
    text_column: str = "text",
    similarity_threshold: float = 0.9,
) -> pd.DataFrame:
    """
    Remove near-duplicate articles using MinHash LSH.
    
    Args:
        articles_df: DataFrame with articles
        text_column: Column containing article text
        similarity_threshold: Jaccard similarity threshold for duplicates
        
    Returns:
        Deduplicated DataFrame
    """
    logger.info(f"Deduplicating {len(articles_df)} articles (threshold={similarity_threshold})")
    
    # Initialize LSH
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
    
    # Compute MinHash signatures
    def text_to_shingles(text: str, k: int = 3) -> Set[str]:
        """Convert text to character k-shingles."""
        text = text.lower()
        return set(text[i:i+k] for i in range(len(text) - k + 1))
    
    signatures = {}
    for idx, row in articles_df.iterrows():
        text = row[text_column]
        shingles = text_to_shingles(text)
        
        # Create MinHash
        minhash = MinHash(num_perm=128)
        for shingle in shingles:
            minhash.update(shingle.encode('utf8'))
        
        signatures[idx] = minhash
        lsh.insert(f"doc_{idx}", minhash)
    
    # Find duplicates
    duplicates = set()
    for idx, minhash in signatures.items():
        if idx in duplicates:
            continue
        
        # Query for similar documents
        candidates = lsh.query(minhash)
        
        # Keep first, mark rest as duplicates
        for candidate in candidates[1:]:  # Skip self
            dup_idx = int(candidate.split('_')[1])
            if dup_idx != idx:
                duplicates.add(dup_idx)
    
    # Remove duplicates
    keep_indices = [idx for idx in articles_df.index if idx not in duplicates]
    deduplicated = articles_df.loc[keep_indices].copy()
    
    removed_pct = len(duplicates) / len(articles_df) * 100
    logger.info(f"Removed {len(duplicates)} duplicates ({removed_pct:.1f}%)")
    
    return deduplicated
