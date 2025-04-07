''' no longer needed 
from typing import Dict, Tuple, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmotionDetectionService:
    """Service for detecting emotions in text using a pre-trained transformer model."""
    
    def __init__(self):
        """Initialize the emotion detection service with a pre-trained model."""
        try:
            # Load the emotion detection model
            self.model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
            
            logger.info(f"Emotion detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize emotion detection service: {str(e)}")
            raise

    def detect_emotion(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in the given text using the pre-trained model.
        
        Args:
            text: The text to analyze for emotions
            
        Returns:
            Dictionary with emotion labels as keys and confidence scores as values
        """
        try:
            results = self.model(text)
            emotions = {result['label']: result['score'] for result in results[0]}
            
            # Sort emotions by probability (descending)
            sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
            
            logger.info(f"Emotion detection successful. Top emotion: {list(sorted_emotions.keys())[0]}")
            return sorted_emotions
            
        except Exception as e:
            logger.error(f"Error detecting emotions: {str(e)}")
            raise Exception(f"Failed to detect emotions: {str(e)}")
    
    async def get_primary_emotion(self, text: str) -> Tuple[str, float]:
        """
        Get the primary emotion with its confidence score.
        
        Args:
            text: The text to analyze
            
        Returns:
            Tuple of (emotion_label, confidence_score)
        """
        emotions = await self.detect_emotion(text)
        top_emotion = next(iter(emotions.items()))
        return top_emotion

from transformers import pipeline

# Load DistilBERT sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")

def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of the given text."""
    result = sentiment_analyzer(text)[0]
    return result["label"]
'''    