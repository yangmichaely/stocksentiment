"""
Sentiment aggregator - combines FinBERT and domain-specific scores
"""
import pandas as pd
import numpy as np
from datetime import datetime

from .finbert_analyzer import get_finbert_analyzer
from .domain_lexicon import get_domain_lexicon
from ..utils.config import config


class SentimentAggregator:
    """Aggregate sentiment from multiple sources"""
    
    def __init__(self):
        """Initialize sentiment analyzers"""
        self.finbert = get_finbert_analyzer()
        self.domain_lexicon = get_domain_lexicon()
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column='text') -> pd.DataFrame:
        """
        Perform comprehensive sentiment analysis on dataframe
        
        Args:
            df: Input dataframe with text data
            text_column: Name of column containing text
        
        Returns:
            DataFrame with all sentiment scores
        """
        if df.empty:
            return df
        
        print(f"\nAnalyzing sentiment for {len(df)} texts...")
        
        # FinBERT analysis
        print("1/2 Running FinBERT analysis...")
        df = self.finbert.analyze_dataframe(df, text_column=text_column)
        
        # Domain-specific analysis
        print("2/2 Running domain-specific analysis...")
        df = self.domain_lexicon.analyze_dataframe(df, text_column=text_column)
        
        # Add combined score (weighted average)
        df['combined_sentiment'] = (
            0.5 * df['sentiment_score'] +
            0.2 * df['demand_score'] +
            0.1 * df['supply_risk_score'] +
            0.1 * df['cost_pressure_score'] +
            0.1 * df['production_sentiment']
        )
        
        print("Sentiment analysis complete!\n")
        
        return df
    
    def aggregate_by_ticker_date(self, df: pd.DataFrame, time_window='1D') -> pd.DataFrame:
        """
        Aggregate sentiment by ticker and date
        
        Args:
            df: Dataframe with sentiment scores
            time_window: Time window for aggregation (e.g., '1D', '1W')
        
        Returns:
            Aggregated dataframe
        """
        if df.empty or 'ticker' not in df.columns or 'timestamp' not in df.columns:
            return df
        
        print(f"Aggregating sentiment by ticker and {time_window} window...")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by ticker and time window
        df['date'] = df['timestamp'].dt.floor(time_window)
        
        # Aggregation functions
        agg_funcs = {
            'sentiment_score': ['mean', 'std', 'min', 'max'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'supply_risk_score': 'mean',
            'demand_score': 'mean',
            'cost_pressure_score': 'mean',
            'regulatory_risk_score': 'mean',
            'production_sentiment': 'mean',
            'combined_sentiment': 'mean',
            'text': 'count'  # Number of texts
        }
        
        # Only aggregate columns that exist
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
        
        aggregated = df.groupby(['ticker', 'date']).agg(agg_funcs).reset_index()
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in aggregated.columns.values]
        
        # Rename count column
        if 'text_count' in aggregated.columns:
            aggregated.rename(columns={'text_count': 'num_texts'}, inplace=True)
        
        # Calculate additional metrics
        if 'sentiment_score_std' in aggregated.columns:
            aggregated['sentiment_volatility'] = aggregated['sentiment_score_std']
        
        if 'negative_mean' in aggregated.columns:
            # Negative sentiment spike ratio
            aggregated['negative_ratio'] = aggregated['negative_mean']
        
        print(f"Aggregated to {len(aggregated)} ticker-date pairs")
        
        return aggregated
    
    def calculate_sentiment_momentum(self, df: pd.DataFrame, periods=[1, 5]) -> pd.DataFrame:
        """
        Calculate sentiment momentum (change over time)
        
        Args:
            df: Aggregated dataframe by ticker and date
            periods: List of periods to calculate momentum
        
        Returns:
            DataFrame with momentum features
        """
        if df.empty or 'ticker' not in df.columns or 'date' not in df.columns:
            return df
        
        print("Calculating sentiment momentum...")
        
        df = df.sort_values(['ticker', 'date'])
        
        for period in periods:
            # Calculate difference from N periods ago
            col_name = 'sentiment_score_mean' if 'sentiment_score_mean' in df.columns else 'combined_sentiment'
            
            if col_name in df.columns:
                df[f'sentiment_momentum_{period}d'] = (
                    df.groupby('ticker')[col_name].diff(periods=period)
                )
        
        return df


def analyze_all_data(df: pd.DataFrame, save=True) -> pd.DataFrame:
    """
    Convenience function to analyze all data
    
    Args:
        df: Raw dataframe with text data
        save: Whether to save results
    
    Returns:
        Analyzed and aggregated dataframe
    """
    aggregator = SentimentAggregator()
    
    # Analyze sentiment
    analyzed_df = aggregator.analyze_dataframe(df)
    
    if save:
        from ..utils.data_utils import save_dataframe
        save_path = config.data_dir / 'processed' / f'sentiment_{datetime.now().strftime("%Y%m%d")}.parquet'
        save_dataframe(analyzed_df, save_path)
        print(f"Saved sentiment analysis to {save_path}")
    
    # Aggregate by ticker and date
    aggregated_df = aggregator.aggregate_by_ticker_date(analyzed_df, time_window='1D')
    
    # Calculate momentum
    aggregated_df = aggregator.calculate_sentiment_momentum(aggregated_df)
    
    if save:
        save_path = config.data_dir / 'processed' / f'sentiment_aggregated_{datetime.now().strftime("%Y%m%d")}.parquet'
        save_dataframe(aggregated_df, save_path)
        print(f"Saved aggregated sentiment to {save_path}")
    
    return aggregated_df
