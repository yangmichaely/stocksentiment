"""
Technical and price-based features
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from ..utils.config import config
from ..utils.universe import get_all_tickers


class TechnicalFeatureEngineer:
    """Create technical and price features"""
    
    def __init__(self):
        """Initialize technical feature engineer"""
        self.tickers = get_all_tickers()
    
    def get_price_data(self, ticker, start_date, end_date):
        """
        Get historical price data
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            df['ticker'] = ticker
            df['date'] = df.index
            return df[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"  Error getting price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_all_price_data(self, start_date, end_date):
        """
        Collect price data for all tickers
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Combined DataFrame
        """
        print(f"Collecting price data for {len(self.tickers)} tickers...")
        
        all_data = []
        
        for ticker in self.tickers:
            df = self.get_price_data(ticker, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Collected price data: {len(combined_df)} rows")
        
        return combined_df
    
    def calculate_returns(self, df: pd.DataFrame, periods=[1, 5, 21]) -> pd.DataFrame:
        """
        Calculate forward returns
        
        Args:
            df: Price dataframe
            periods: List of periods for forward returns
        
        Returns:
            DataFrame with return columns
        """
        if df.empty or 'Close' not in df.columns:
            return df
        
        print("Calculating forward returns...")
        
        df = df.sort_values(['ticker', 'date'])
        
        for period in periods:
            # Forward return (what we want to predict)
            df[f'forward_return_{period}d'] = (
                df.groupby('ticker')['Close'].shift(-period) / df['Close'].replace(0, np.nan) - 1
            )
            
            # Historical return (for momentum features)
            df[f'historical_return_{period}d'] = (
                df['Close'] / df.groupby('ticker')['Close'].shift(period).replace(0, np.nan) - 1
            )
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, windows=[5, 21]) -> pd.DataFrame:
        """
        Calculate price volatility
        
        Args:
            df: Price dataframe with returns
            windows: Rolling windows
        
        Returns:
            DataFrame with volatility columns
        """
        if df.empty:
            return df
        
        print("Calculating volatility...")
        
        df = df.sort_values(['ticker', 'date'])
        
        # Daily returns
        df['daily_return'] = df.groupby('ticker')['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
        
        for window in windows:
            # Rolling volatility (std of returns)
            df[f'volatility_{window}d'] = (
                df.groupby('ticker')['daily_return'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            )
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: Price dataframe
        
        Returns:
            DataFrame with technical indicators
        """
        if df.empty or 'Close' not in df.columns:
            return df
        
        print("Calculating technical indicators...")
        
        df = df.sort_values(['ticker', 'date'])
        
        # Moving averages
        for window in [5, 21, 50]:
            df[f'sma_{window}'] = (
                df.groupby('ticker')['Close'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            )
        
        # Price relative to moving average
        df['price_to_sma_21'] = df['Close'] / df['sma_21']
        df['price_to_sma_50'] = df['Close'] / df['sma_50']
        
        # RSI (Relative Strength Index)
        df = self._calculate_rsi(df)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = df.sort_values(['ticker', 'date'])
        
        # Price changes
        df['price_change'] = df.groupby('ticker')['Close'].diff()
        
        # Separate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Average gains and losses
        df['avg_gain'] = df.groupby('ticker')['gain'].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
        df['avg_loss'] = df.groupby('ticker')['loss'].transform(
            lambda x: x.rolling(window=period, min_periods=1).mean()
        )
        
        # RS and RSI
        df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-8)
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Clean up temporary columns
        df = df.drop(columns=['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'])
        
        return df
    
    def compare_to_benchmark(self, df: pd.DataFrame, benchmark='XLB') -> pd.DataFrame:
        """
        Calculate returns relative to benchmark
        
        Args:
            df: Price dataframe
            benchmark: Benchmark ticker (e.g., XLB)
        
        Returns:
            DataFrame with relative performance
        """
        if df.empty:
            return df
        
        print(f"Calculating relative performance vs {benchmark}...")
        
        # CRITICAL: Use HISTORICAL returns, not forward returns (no look-ahead bias)
        # Calculate benchmark's historical return
        benchmark_df = df[df['ticker'] == benchmark][['date', 'Close']].copy()
        benchmark_df['benchmark_return_5d'] = benchmark_df['Close'].pct_change(5)
        benchmark_df = benchmark_df[['date', 'benchmark_return_5d']]
        
        # Merge with main dataframe
        df = df.merge(benchmark_df, on='date', how='left')
        
        # Calculate relative HISTORICAL performance (not forward)
        if 'historical_return_5d' in df.columns and 'benchmark_return_5d' in df.columns:
            df['relative_strength_5d'] = df['historical_return_5d'] - df['benchmark_return_5d']
        
        # Note: We removed 'relative_return_5d' as it was using forward returns (data leakage)
        # We removed 'outperform_benchmark' as it was also using forward returns
        
        return df
    
    def create_all_features(self, start_date, end_date) -> pd.DataFrame:
        """
        Create all technical features
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with all technical features
        """
        print("\n=== Creating Technical Features ===")
        
        # Collect price data
        df = self.collect_all_price_data(start_date, end_date)
        
        if df.empty:
            return df
        
        # Calculate features
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_technical_indicators(df)
        df = self.compare_to_benchmark(df)
        
        # Fill NaN values
        df = df.ffill().fillna(0)
        
        print("Technical feature engineering complete!\n")
        
        return df
