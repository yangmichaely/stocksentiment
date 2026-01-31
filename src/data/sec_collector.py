"""
SEC filings collector (8-K, 10-Q, 10-K)
"""
import pandas as pd
from datetime import datetime, timedelta
from sec_edgar_downloader import Downloader
import os

from ..utils.config import config
from ..utils.universe import get_all_tickers


class SECCollector:
    """Collect SEC filings"""
    
    def __init__(self):
        """Initialize SEC collector"""
        self.tickers = get_all_tickers()
        self.download_dir = config.data_dir / 'sec_filings'
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloader (new API)
        self.downloader = Downloader(download_folder=str(self.download_dir))
    
    def collect_8k_filings(self, ticker, after_date=None, limit=10):
        """
        Collect 8-K filings (material events)
        
        Args:
            ticker: Stock ticker
            after_date: Only get filings after this date
            limit: Maximum number of filings
        
        Returns:
            List of filing metadata
        """
        try:
            # Download 8-K filings
            self.downloader.get("8-K", ticker, limit=limit, after=after_date)
            
            # Parse downloaded files
            ticker_dir = self.download_dir / "sec-edgar-filings" / ticker / "8-K"
            
            if not ticker_dir.exists():
                return []
            
            filings = []
            for filing_dir in sorted(ticker_dir.iterdir(), reverse=True)[:limit]:
                if filing_dir.is_dir():
                    # Read filing metadata
                    full_submission_path = filing_dir / "full-submission.txt"
                    
                    if full_submission_path.exists():
                        with open(full_submission_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:5000]  # First 5000 chars
                        
                        filings.append({
                            'ticker': ticker,
                            'filing_type': '8-K',
                            'filing_date': filing_dir.name,
                            'text': content,
                            'path': str(full_submission_path)
                        })
            
            return filings
            
        except Exception as e:
            print(f"  Error collecting 8-K for {ticker}: {e}")
            return []
    
    def collect_all(self, days_back=90, save=True):
        """
        Collect SEC filings for all tickers
        
        Args:
            days_back: Number of days to look back
            save: Whether to save to disk
        
        Returns:
            DataFrame with SEC filings
        """
        print("Collecting SEC filings (8-K)...")
        
        all_filings = []
        after_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        for ticker in self.tickers:
            print(f"  Collecting SEC filings for {ticker}...")
            
            filings = self.collect_8k_filings(ticker, after_date=after_date)
            all_filings.extend(filings)
        
        if not all_filings:
            print("No SEC filings collected")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_filings)
        df['timestamp'] = pd.to_datetime(df['filing_date'])
        df['source'] = 'sec_filing'
        df['data_source'] = 'sec_edgar'
        
        print(f"Collected {len(df)} SEC filings")
        
        if save:
            from ..utils.data_utils import save_dataframe
            save_path = config.data_dir / 'raw' / f'sec_{datetime.now().strftime("%Y%m%d")}.parquet'
            save_dataframe(df, save_path)
            print(f"Saved to {save_path}")
        
        return df
