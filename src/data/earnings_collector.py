"""
Earnings call transcript collector
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup

from ..utils.config import config
from ..utils.universe import get_all_tickers


class EarningsCollector:
    """Collect earnings call transcripts and summaries"""
    
    def __init__(self):
        """Initialize earnings collector"""
        self.tickers = get_all_tickers()
        self.alpha_vantage_key = config.alpha_vantage_key
    
    def get_earnings_dates(self, ticker, quarters_back=4):
        """
        Get earnings dates for a ticker using yfinance
        
        Args:
            ticker: Stock ticker
            quarters_back: Number of quarters to look back
        
        Returns:
            List of earnings dates
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            # Get earnings dates
            earnings_dates = stock.earnings_dates
            
            if earnings_dates is None or earnings_dates.empty:
                return []
            
            # Get most recent dates
            dates = earnings_dates.head(quarters_back * 2).index.tolist()
            return [d.to_pydatetime() for d in dates]
            
        except Exception as e:
            print(f"  Error getting earnings dates for {ticker}: {e}")
            return []
    
    def collect_earnings_sentiment_proxy(self, ticker):
        """
        Collect earnings sentiment proxy from Alpha Vantage
        Uses actual earnings vs estimates as a sentiment signal
        
        Args:
            ticker: Stock ticker
        
        Returns:
            DataFrame with earnings data
        """
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key not configured")
            return pd.DataFrame()
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'EARNINGS',
            'symbol': ticker,
            'apikey': self.alpha_vantage_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'quarterlyEarnings' not in data:
                return pd.DataFrame()
            
            earnings_data = []
            for earning in data['quarterlyEarnings'][:8]:  # Last 2 years
                reported_eps = earning.get('reportedEPS')
                estimated_eps = earning.get('estimatedEPS')
                
                if reported_eps and estimated_eps:
                    try:
                        reported = float(reported_eps)
                        estimated = float(estimated_eps)
                        surprise = reported - estimated
                        surprise_pct = (surprise / abs(estimated)) * 100 if estimated != 0 else 0
                        
                        earnings_data.append({
                            'ticker': ticker,
                            'fiscal_date': earning.get('fiscalDateEnding'),
                            'reported_date': earning.get('reportedDate'),
                            'reported_eps': reported,
                            'estimated_eps': estimated,
                            'surprise': surprise,
                            'surprise_pct': surprise_pct,
                            'text': f"Earnings: Reported EPS ${reported} vs Estimate ${estimated}, surprise {surprise_pct:.1f}%",
                            'timestamp': pd.to_datetime(earning.get('reportedDate')),
                            'source': 'earnings_report',
                            'data_source': 'alpha_vantage'
                        })
                    except (ValueError, TypeError):
                        continue
            
            return pd.DataFrame(earnings_data)
            
        except Exception as e:
            print(f"  Error collecting earnings for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_all(self, save=True):
        """
        Collect earnings data for all tickers
        
        Args:
            save: Whether to save to disk
        
        Returns:
            DataFrame with earnings data
        """
        print("Collecting earnings data...")
        
        all_data = []
        
        for ticker in self.tickers:
            print(f"  Collecting earnings for {ticker}...")
            
            df = self.collect_earnings_sentiment_proxy(ticker)
            if not df.empty:
                all_data.append(df)
            
            # Rate limiting (Alpha Vantage: 5 calls/minute)
            time.sleep(12)
        
        if not all_data:
            print("No earnings data collected")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp', ascending=False)
        
        print(f"Collected {len(combined_df)} earnings reports")
        
        if save:
            from ..utils.data_utils import save_dataframe
            save_path = config.data_dir / 'raw' / f'earnings_{datetime.now().strftime("%Y%m%d")}.parquet'
            save_dataframe(combined_df, save_path)
            print(f"Saved to {save_path}")
        
        return combined_df
