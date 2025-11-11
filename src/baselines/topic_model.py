"""Topic model baseline using STM."""

import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class TopicModelBaseline:
    """Structural Topic Model baseline for frame detection."""
    
    def __init__(self, n_topics: int = 8):
        """
        Initialize topic model.
        
        Args:
            n_topics: Number of topics (matches 8 frames)
        """
        self.n_topics = n_topics
        
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import CountVectorizer
            self.LDA = LatentDirichletAllocation
            self.CountVectorizer = CountVectorizer
        except ImportError:
            raise ImportError("Install scikit-learn")
        
        self.vectorizer = None
        self.model = None
        
        logger.info(f"Initialized TopicModelBaseline with {n_topics} topics")
    
    def fit(self, documents: List[str]):
        """Fit topic model to documents."""
        # Vectorize
        self.vectorizer = self.CountVectorizer(
            max_features=5000,
            stop_words='english',
            max_df=0.95,
            min_df=2,
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA
        self.model = self.LDA(
            n_components=self.n_topics,
            random_state=42,
            max_iter=100,
        )
        
        self.model.fit(doc_term_matrix)
        logger.info("Topic model fitted")
    
    def predict(self, document: str) -> int:
        """Predict dominant topic for document."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        vec = self.vectorizer.transform([document])
        topic_dist = self.model.transform(vec)
        
        return int(np.argmax(topic_dist))
