"""
Data utilities
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json


def save_dataframe(df, path, format='parquet'):
    """Save dataframe in specified format"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(path)
    elif format == 'csv':
        df.to_csv(path, index=False)
    elif format == 'pickle':
        df.to_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(path, format='parquet'):
    """Load dataframe from specified format"""
    path = Path(path)
    
    if not path.exists():
        return None
    
    if format == 'parquet':
        return pd.read_parquet(path)
    elif format == 'csv':
        return pd.read_csv(path)
    elif format == 'pickle':
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_date_range(days_back=30):
    """Get date range for data collection"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def align_timestamps_to_market_close(timestamps):
    """
    Align timestamps to market close (4 PM ET)
    This prevents look-ahead bias from intraday sentiment
    """
    df = pd.DataFrame({'timestamp': timestamps})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set to 4 PM ET (market close)
    df['aligned_date'] = df['timestamp'].dt.normalize() + pd.Timedelta(hours=16)
    
    # If timestamp is after 4 PM, use next day's close
    mask = df['timestamp'].dt.hour >= 16
    df.loc[mask, 'aligned_date'] = df.loc[mask, 'aligned_date'] + pd.Timedelta(days=1)
    
    return df['aligned_date'].tolist()


def remove_weekends_holidays(dates):
    """Remove weekends from date list"""
    df = pd.DataFrame({'date': dates})
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove weekends
    df = df[df['date'].dt.dayofweek < 5]
    
    return df['date'].tolist()


def cache_data(func):
    """Decorator to cache data collection results"""
    def wrapper(*args, **kwargs):
        # Create cache key from function name and args
        cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
        cache_file = Path('data/cache') / f"{cache_key}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if cached data exists and is recent (< 1 day old)
        if cache_file.exists():
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < timedelta(days=1):
                print(f"Loading cached data for {func.__name__}")
                return pd.read_pickle(cache_file)
        
        # Otherwise, call function and cache result
        result = func(*args, **kwargs)
        if result is not None and not result.empty:
            result.to_pickle(cache_file)
        
        return result
    
    return wrapper
