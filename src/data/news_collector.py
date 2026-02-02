"""
News collector using Finnhub and GDELT
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import time
import hashlib

from ..utils.config import config
from ..utils.universe import get_all_tickers


class NewsCollector:
    """Collect news articles from Finnhub and GDELT"""
    
    def __init__(self):
        """Initialize news collectors"""
        self.tickers = get_all_tickers()
        
        self.keywords = config.get('data_collection', 'news', 'keywords', default=[
            'mining', 'metals', 'commodities', 'copper', 'gold', 'lithium', 'steel'
        ])
    
    def collect_gdelt_news(self, days_back=30):
        """
        Collect news from GDELT (Global Database of Events, Language, and Tone)
        
        Args:
            days_back: Number of days to look back
        
        Returns:
            DataFrame with news articles
        """
        print("Collecting news from GDELT...")
        
        all_articles = []
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # GDELT DOC 2.0 API (free, no key required)
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Use only keywords to avoid hitting rate limits with many tickers
        # GDELT has aggressive rate limiting
        search_terms = self.keywords + ['mining stocks', 'metal prices', 'commodity markets']
        
        for i, term in enumerate(search_terms):
            params = {
                'query': term,
                'mode': 'artlist',
                'maxrecords': 100,  # Reduced from 250 to be more conservative
                'format': 'json',
                'startdatetime': start_date.strftime('%Y%m%d%H%M%S'),
                'enddatetime': end_date.strftime('%Y%m%d%H%M%S')
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    print(f"  GDELT rate limited, waiting 120s...")
                    time.sleep(120)
                    response = requests.get(base_url, params=params, timeout=30)
                
                response.raise_for_status()
                
                # Handle empty or invalid JSON
                try:
                    data = response.json()
                except ValueError:
                    print(f"  Invalid JSON response for {term}, skipping...")
                    continue
                
                articles = data.get('articles', [])
                for article in articles:
                    # Try to match article to tickers based on content
                    title = article.get('title', '').lower()
                    matched_tickers = [t for t in self.tickers if t.lower() in title]
                    
                    ticker = matched_tickers[0] if matched_tickers else 'SECTOR'
                    
                    # Parse timestamp
                    timestamp_str = article.get('seendate', '')
                    try:
                        timestamp = pd.to_datetime(timestamp_str, format='%Y%m%dT%H%M%SZ', utc=True).tz_localize(None)
                    except Exception:
                        timestamp = datetime.now().replace(tzinfo=None)
                    
                    all_articles.append({
                        'ticker': ticker,
                        'text': f"{article.get('title', '')}",
                        'timestamp': timestamp,
                        'source': article.get('domain', 'gdelt'),
                        'url': article.get('url', ''),
                        'tone': article.get('tone', 0),  # GDELT tone score
                        'data_source': 'gdelt'
                    })
                
                # GDELT rate limiting: very conservative (can be aggressive)
                # Wait 5 seconds between requests
                time.sleep(5)
                
                print(f"  Processed {i + 1}/{len(search_terms)} search terms...")
                
            except requests.exceptions.RequestException as e:
                if "429" in str(e):
                    print(f"  GDELT rate limit hit, skipping remaining searches...")
                    break
                continue
            except Exception as e:
                continue
        
        df = pd.DataFrame(all_articles)
        # Deduplicate by URL
        if not df.empty:
            df = df.drop_duplicates(subset=['url'])
        
        print(f"Collected {len(df)} articles from GDELT")
        
        return df
    
    def collect_finnhub_news(self, days_back=30):
        """
        Collect company news from Finnhub
        
        Args:
            days_back: Number of days to look back
        
        Returns:
            DataFrame with news articles
        """
        if not config.finnhub_api_key:
            print("Finnhub API key not configured")
            return pd.DataFrame()
        
        print("Collecting news from Finnhub...")
        
        all_articles = []
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        base_url = "https://finnhub.io/api/v1/company-news"
        
        for i, ticker in enumerate(self.tickers):
            params = {
                'symbol': ticker,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': config.finnhub_api_key
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                
                # Handle rate limiting
                if response.status_code == 429:
                    print(f"  Rate limited, waiting 60s...")
                    time.sleep(60)
                    response = requests.get(base_url, params=params, timeout=10)
                
                response.raise_for_status()
                articles = response.json()
                
                for article in articles:
                    timestamp = pd.to_datetime(article.get('datetime'), unit='s')
                    if timestamp.tz is not None:
                        timestamp = timestamp.tz_localize(None)
                    all_articles.append({
                        'ticker': ticker,
                        'text': f"{article.get('headline', '')} {article.get('summary', '')}",
                        'timestamp': timestamp,
                        'source': article.get('source', 'unknown'),
                        'url': article.get('url', ''),
                        'data_source': 'finnhub'
                    })
                
                # Rate limiting: Finnhub free tier = 60 calls/min
                # Sleep 1.5s between calls = ~40 calls/min (safe margin)
                time.sleep(1.5)
                
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(self.tickers)} tickers...")
                
            except requests.exceptions.RequestException as e:
                if "429" in str(e):
                    print(f"  Rate limit hit at {ticker}, skipping remaining...")
                    break
                continue
            except Exception as e:
                continue
        
        df = pd.DataFrame(all_articles)
        print(f"Collected {len(df)} articles from Finnhub")
        
        return df
    
    def collect_all(self, days_back=30, save=True):
        """
        Collect news from all sources
        
        Args:
            days_back: Number of days to look back
            save: Whether to save to disk
        
        Returns:
            Combined DataFrame
        """
        dfs = []
        
        # Finnhub
        finnhub_df = self.collect_finnhub_news(days_back)
        if not finnhub_df.empty:
            dfs.append(finnhub_df)
        
        # GDELT
        gdelt_df = self.collect_gdelt_news(days_back)
        if not gdelt_df.empty:
            dfs.append(gdelt_df)
        
        if not dfs:
            print("No news articles collected")
            return pd.DataFrame()
        
        # Combine all sources
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Clean and deduplicate
        combined_df = combined_df.drop_duplicates(subset=['text', 'timestamp'])
        combined_df = combined_df[combined_df['text'].str.len() > 20]
        combined_df = combined_df.sort_values('timestamp', ascending=False)
        
        if save:
            from ..utils.data_utils import save_dataframe
            save_path = config.data_dir / 'raw' / f'news_{datetime.now().strftime("%Y%m%d")}.parquet'
            save_dataframe(combined_df, save_path)
            print(f"Saved to {save_path}")
        
        return combined_df
