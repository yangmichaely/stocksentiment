"""
Sentiment-based feature engineering
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.config import config


class SentimentFeatureEngineer:
    """Create features from sentiment data"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.time_windows = config.get('features', 'time_windows', default=[1, 5, 21])
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Aggregated sentiment dataframe (by ticker and date)
        
        Returns:
            DataFrame with rolling features
        """
        if df.empty or 'ticker' not in df.columns or 'date' not in df.columns:
            return df
        
        print("Creating rolling window features...")
        
        df = df.sort_values(['ticker', 'date'])
        
        # Base sentiment column
        sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in df.columns else 'combined_sentiment'
        
        if sentiment_col not in df.columns:
            return df
        
        for window in self.time_windows:
            # Rolling mean
            df[f'sentiment_rolling_{window}d'] = (
                df.groupby('ticker')[sentiment_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            )
            
            # Rolling std
            df[f'sentiment_std_{window}d'] = (
                df.groupby('ticker')[sentiment_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            )
            
            # Rolling min/max
            df[f'sentiment_min_{window}d'] = (
                df.groupby('ticker')[sentiment_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
            )
            
            df[f'sentiment_max_{window}d'] = (
                df.groupby('ticker')[sentiment_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            )
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum features (rate of change)
        
        Args:
            df: Dataframe with sentiment data
        
        Returns:
            DataFrame with momentum features
        """
        if df.empty or 'ticker' not in df.columns:
            return df
        
        print("Creating momentum features...")
        
        df = df.sort_values(['ticker', 'date'])
        
        sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in df.columns else 'combined_sentiment'
        
        if sentiment_col not in df.columns:
            return df
        
        for window in self.time_windows:
            # Percentage change
            df[f'sentiment_pct_change_{window}d'] = (
                df.groupby('ticker')[sentiment_col].pct_change(periods=window).replace([np.inf, -np.inf], np.nan)
            )
            
            # Absolute change
            df[f'sentiment_change_{window}d'] = (
                df.groupby('ticker')[sentiment_col].diff(periods=window)
            )
        
        return df
    
    def create_spike_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment spike features (extreme values)
        
        Args:
            df: Dataframe with sentiment data
        
        Returns:
            DataFrame with spike features
        """
        if df.empty or 'ticker' not in df.columns:
            return df
        
        print("Creating spike features...")
        
        sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in df.columns else 'combined_sentiment'
        
        if sentiment_col not in df.columns:
            return df
        
        # Calculate z-score (standardized sentiment)
        df['sentiment_zscore'] = df.groupby('ticker')[sentiment_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Identify extreme values (spikes)
        df['sentiment_spike_positive'] = (df['sentiment_zscore'] > 2).astype(int)
        df['sentiment_spike_negative'] = (df['sentiment_zscore'] < -2).astype(int)
        
        # Negative sentiment ratio (if available)
        if 'negative_mean' in df.columns:
            df['negative_ratio'] = df['negative_mean']
            
            # High negative ratio flag
            df['high_negative_flag'] = (df['negative_ratio'] > 0.5).astype(int)
        
        return df
    
    def create_volume_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-weighted sentiment features
        
        Args:
            df: Dataframe with sentiment data and num_texts
        
        Returns:
            DataFrame with volume-weighted features
        """
        if df.empty or 'num_texts' not in df.columns:
            return df
        
        print("Creating volume-weighted features...")
        
        sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in df.columns else 'combined_sentiment'
        
        if sentiment_col not in df.columns:
            return df
        
        # Volume-weighted sentiment for each time window
        for window in self.time_windows:
            df[f'volume_weighted_sentiment_{window}d'] = (
                df.groupby('ticker').apply(
                    lambda x: (x[sentiment_col] * x['num_texts']).rolling(window=window).sum() / 
                             (x['num_texts'].rolling(window=window).sum() + 1e-8)
                ).reset_index(level=0, drop=True)
            )
        
        # Log of text volume
        df['log_num_texts'] = np.log1p(df['num_texts'])
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all sentiment features
        
        Args:
            df: Aggregated sentiment dataframe
        
        Returns:
            DataFrame with all features
        """
        print("\n=== Creating Sentiment Features ===")
        
        df = self.create_rolling_features(df)
        df = self.create_momentum_features(df)
        df = self.create_spike_features(df)
        df = self.create_volume_weighted_features(df)
        
        # Fill NaN values
        df = df.ffill().fillna(0)
        
        print("Feature engineering complete!\n")
        
        return df
