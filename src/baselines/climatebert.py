"""ClimateBERT baseline."""

import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class ClimateBERTBaseline:
    """ClimateBERT for climate-specific tasks."""
    
    def __init__(self, model_name: str = "climatebert/distilroberta-base-climate-f"):
        """
        Initialize ClimateBERT.
        
        Args:
            model_name: ClimateBERT model variant
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        logger.info(f"Initialized ClimateBERTBaseline with {model_name}")
    
    # Similar implementation to RoBERTa baseline
    # (fine-tuning methods would be nearly identical)
