"""
Data manager for incremental collection and rolling window management
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from ..utils.config import config
from ..utils.data_utils import load_dataframe, save_dataframe


class DataManager:
    """Manage incremental data collection with rolling windows"""
    
    def __init__(self, window_days=90):
        """
        Initialize data manager
        
        Args:
            window_days: Size of rolling window (default 90 days)
        """
        self.window_days = window_days
        self.data_dir = config.data_dir / 'processed'
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_date(self, data_type='sentiment') -> datetime:
        """
        Get the latest date in existing data
        
        Args:
            data_type: Type of data ('sentiment', 'features', 'prices')
        
        Returns:
            Latest date or None if no data exists
        """
        file_patterns = {
            'sentiment': 'sentiment_aggregated_*.parquet',
            'features': 'features_*.parquet',
            'prices': 'price_data_*.parquet'
        }
        
        pattern = file_patterns.get(data_type, 'sentiment_aggregated_*.parquet')
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            return None
        
        # Find most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = load_dataframe(latest_file)
            if 'date' in df.columns:
                latest_date = pd.to_datetime(df['date']).max()
                print(f"  Found existing {data_type}: {latest_file.name}, latest date: {latest_date.date()}")
                return latest_date
        except Exception as e:
            print(f"Warning: Could not load {data_type} data: {e}")
        
        return None
    
    def append_and_window(self, new_data: pd.DataFrame, data_type='sentiment') -> pd.DataFrame:
        """
        Append new data and maintain rolling window
        
        Args:
            new_data: New data to append
            data_type: Type of data ('sentiment', 'features', 'prices')
        
        Returns:
            Combined data with rolling window applied
        """
        file_patterns = {
            'sentiment': 'sentiment_aggregated_*.parquet',
            'features': 'features_*.parquet',
            'prices': 'price_data_*.parquet'
        }
        
        pattern = file_patterns.get(data_type, 'sentiment_aggregated_*.parquet')
        files = list(self.data_dir.glob(pattern))
        
        # Load existing data if it exists
        if files:
            # Get most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            try:
                existing_data = load_dataframe(latest_file)
                print(f"✓ Loaded existing {data_type}: {len(existing_data)} rows from {latest_file.name}")
            except Exception as e:
                print(f"Warning: Could not load existing {data_type}: {e}")
                existing_data = pd.DataFrame()
        else:
            existing_data = pd.DataFrame()
            print(f"No existing {data_type} data found - creating new")
        
        # Append new data
        if not existing_data.empty:
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates (keep latest)
            if 'date' in combined.columns and 'ticker' in combined.columns:
                combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
        else:
            combined = new_data
        
        # Apply rolling window
        if 'date' in combined.columns:
            combined['date'] = pd.to_datetime(combined['date'])
            cutoff_date = datetime.now() - timedelta(days=self.window_days)
            
            before_count = len(combined)
            combined = combined[combined['date'] >= cutoff_date]
            after_count = len(combined)
            
            if before_count > after_count:
                print(f"✓ Rolling window: Dropped {before_count - after_count} old rows (keeping last {self.window_days} days)")
        
        # Save updated data with dated filename
        from datetime import datetime
        file_names = {
            'sentiment': f'sentiment_aggregated_{datetime.now().strftime("%Y%m%d")}.parquet',
            'features': f'features_{datetime.now().strftime("%Y%m%d")}.parquet',
            'prices': f'price_data_{datetime.now().strftime("%Y%m%d")}.parquet'
        }
        
        output_file = self.data_dir / file_names.get(data_type, f'{data_type}_{datetime.now().strftime("%Y%m%d")}.parquet')
        save_dataframe(combined, output_file)
        print(f"✓ Saved {data_type}: {len(combined)} rows to {output_file.name}")
        
        return combined
    
    def needs_update(self, data_type='sentiment', max_age_days=7) -> bool:
        """
        Check if data needs updating
        
        Args:
            data_type: Type of data to check
            max_age_days: Maximum age in days before update needed
        
        Returns:
            True if update needed
        """
        latest_date = self.get_latest_date(data_type)
        
        if latest_date is None:
            print(f"✓ {data_type} update needed: No existing data")
            return True
        
        days_old = (datetime.now() - latest_date).days
        
        if days_old >= max_age_days:
            print(f"✓ {data_type} update needed: {days_old} days old (threshold: {max_age_days})")
            return True
        else:
            print(f"✓ {data_type} up to date: {days_old} days old")
            return False
    
    def get_collection_window(self, data_type='sentiment', default_days=7) -> tuple:
        """
        Calculate optimal collection window
        
        Args:
            data_type: Type of data
            default_days: Default number of days to collect
        
        Returns:
            (start_date, end_date) tuple
        """
        latest_date = self.get_latest_date(data_type)
        
        end_date = datetime.now()
        
        if latest_date is None:
            # No existing data - collect full window
            start_date = end_date - timedelta(days=self.window_days)
            print(f"✓ First collection: {self.window_days} days")
        else:
            # Incremental collection from last data point
            start_date = latest_date + timedelta(days=1)
            days_to_collect = (end_date - start_date).days
            
            if days_to_collect <= 0:
                print(f"✓ Data is current (last: {latest_date.date()})")
                return None, None
            
            print(f"✓ Incremental collection: {days_to_collect} days (from {start_date.date()})")
        
        return start_date, end_date
