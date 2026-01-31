"""
FinBERT-based sentiment analyzer for financial text
"""
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import config


class FinBERTAnalyzer:
    """FinBERT sentiment analyzer"""
    
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize FinBERT model
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading FinBERT model: {model_name}...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT labels: positive, negative, neutral
        self.labels = ['positive', 'negative', 'neutral']
        
        print(f"Model loaded on {self.device}")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text
        
        Returns:
            Dict with positive, negative, neutral probabilities and sentiment_score
        """
        if not text or len(text.strip()) == 0:
            return {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'sentiment_score': 0.0
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        # Create result dict
        result = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2])
        }
        
        # Calculate sentiment score: positive - negative (range: -1 to 1)
        result['sentiment_score'] = result['positive'] - result['negative']
        
        return result
    
    def analyze_batch(self, texts: List[str], batch_size=32) -> pd.DataFrame:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with sentiment scores
        """
        print(f"Analyzing {len(texts)} texts...")
        
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                result = self.analyze_text(text)
                all_results.append(result)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {i}/{len(texts)} texts")
        
        df = pd.DataFrame(all_results)
        print(f"Sentiment analysis complete")
        
        return df
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column='text') -> pd.DataFrame:
        """
        Add sentiment scores to a dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text
        
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty or text_column not in df.columns:
            return df
        
        # Analyze texts
        sentiment_df = self.analyze_batch(df[text_column].tolist())
        
        # Combine with original dataframe
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        return result_df


def get_finbert_analyzer():
    """Get FinBERT analyzer instance (singleton pattern)"""
    if not hasattr(get_finbert_analyzer, 'instance'):
        get_finbert_analyzer.instance = FinBERTAnalyzer()
    return get_finbert_analyzer.instance
