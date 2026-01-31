"""
ETF holdings fetcher for discovering stock universe
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import json
from datetime import datetime, timedelta


def get_etf_top_holdings_yfinance(etf_ticker: str, top_n: int = 50) -> List[str]:
    """
    Try to get top holdings from yfinance
    
    Args:
        etf_ticker: ETF ticker symbol
        top_n: Number of top holdings to fetch
    
    Returns:
        List of stock tickers
    """
    try:
        etf = yf.Ticker(etf_ticker)
        
        # Try to get holdings info - yfinance structure varies
        if hasattr(etf, 'funds_data') and etf.funds_data:
            # Check if funds_data is a dict
            if isinstance(etf.funds_data, dict):
                holdings = etf.funds_data.get('holdings', [])
                if holdings:
                    tickers = [h.get('symbol', '') for h in holdings[:top_n] if isinstance(h, dict) and h.get('symbol')]
                    return [t for t in tickers if t and not t.startswith('%')]
            # If it's an object, try accessing as attribute
            elif hasattr(etf.funds_data, 'holdings'):
                holdings = etf.funds_data.holdings
                if holdings and isinstance(holdings, list):
                    tickers = [h.get('symbol', '') if isinstance(h, dict) else str(h) for h in holdings[:top_n]]
                    return [t for t in tickers if t and not t.startswith('%')]
        
        # yfinance doesn't have reliable ETF holdings data - fallback will be used
        return []
        
    except Exception as e:
        # Silently fail - we have fallback data
        return []


# Fallback: Manually curated holdings from major ETFs (updated periodically)
FALLBACK_HOLDINGS = {
    'XLB': [  # Materials Select Sector SPDR
        'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'CTVA', 'DD', 'NUE', 'VMC',
        'MLM', 'STLD', 'ALB', 'PPG', 'IFF', 'EMN', 'CE', 'BALL', 'AVY', 'CF',
        'MOS', 'FMC', 'PKG', 'IP', 'WRK', 'CLF', 'X', 'AA', 'MT', 'SCCO'
    ],
    'PICK': [  # iShares MSCI Global Metals & Mining
        'BHP', 'RIO', 'VALE', 'GLNCY', 'SCCO', 'FCX', 'NEM', 'GOLD', 'AEM',
        'FNV', 'WPM', 'TECK', 'HBM', 'FM', 'HL', 'RGLD', 'CDE', 'AU', 'AG',
        'EQX', 'WDO', 'PAAS', 'OR', 'SVM', 'MAG', 'IAG', 'KGC', 'NGD'
    ],
    'COPX': [  # Global X Copper Miners
        'FCX', 'SCCO', 'TECK', 'FM', 'HBM', 'CMCL', 'TRQ', 'CPRX', 'CPER',
        'LUNMF', 'GLNCY', 'RIO', 'BHP', 'VALE', 'CPPMF', 'ATUSF'
    ],
    'GDX': [  # VanEck Gold Miners
        'NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'KGC', 'AUY', 'IAG', 'RGLD',
        'AU', 'AGI', 'HMY', 'PAAS', 'EGO', 'OR', 'BTG', 'NGD', 'DRD'
    ],
    'GDXJ': [  # VanEck Junior Gold Miners
        'KGC', 'AUY', 'IAG', 'AU', 'EGO', 'OR', 'BTG', 'NGD', 'SSRM', 'DRD',
        'EDV', 'CDE', 'SAND', 'AGI', 'EQX', 'GATO', 'WDO', 'KNT', 'GPL', 'IAUX'
    ],
    'SLV': [  # iShares Silver Trust (physical silver)
        'HL', 'AG', 'PAAS', 'MAG', 'FSM', 'EXK', 'CDE', 'SILV'
    ],
    'SILJ': [  # Junior Silver Miners
        'HL', 'AG', 'PAAS', 'MAG', 'FSM', 'EXK', 'CDE', 'GPL', 'GATO', 'SILV',
        'USAS', 'MUX', 'ASM', 'SAND', 'IAG', 'AUY'
    ],
    'XME': [  # SPDR S&P Metals & Mining
        'CLF', 'NUE', 'STLD', 'X', 'RS', 'CMC', 'CENX', 'MP', 'ATI', 'HCC',
        'ZEUS', 'WOR', 'AA', 'FCX', 'NEM', 'GOLD', 'SCCO', 'TECK', 'HL', 'MT'
    ],
    'SLX': [  # VanEck Steel
        'NUE', 'STLD', 'CLF', 'X', 'RS', 'CMC', 'TX', 'MT', 'ATI', 'VALE',
        'ROCK', 'ZEUS', 'WOR', 'PKX', 'SXC'
    ],
    'REMX': [  # VanEck Rare Earth/Strategic Metals
        'MP', 'LYSCF', 'AVL', 'UCORE', 'UURAF', 'ALKEF', 'PILBF', 'GWMGF',
        'FSUMF', 'MLLOF', 'ALB', 'SQM', 'LAC', 'LTHM', 'PLL', 'LIT'
    ]
}


def get_cached_holdings(etf_ticker: str, cache_dir: Path) -> List[str]:
    """
    Get cached holdings if available and recent
    
    Args:
        etf_ticker: ETF ticker
        cache_dir: Cache directory
    
    Returns:
        List of tickers or empty list
    """
    cache_file = cache_dir / f"{etf_ticker}_holdings.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Check if cache is recent (< 7 days old)
            cached_date = datetime.fromisoformat(data.get('date', '2000-01-01'))
            if datetime.now() - cached_date < timedelta(days=7):
                return data.get('holdings', [])
        except Exception:
            pass
    
    return []


def save_holdings_cache(etf_ticker: str, holdings: List[str], cache_dir: Path):
    """Save holdings to cache"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{etf_ticker}_holdings.json"
    
    with open(cache_file, 'w') as f:
        json.dump({
            'etf': etf_ticker,
            'date': datetime.now().isoformat(),
            'holdings': holdings
        }, f)


def get_etf_holdings(etf_ticker: str, cache_dir: Path = None) -> List[str]:
    """
    Get holdings from an ETF with fallback strategy
    
    Args:
        etf_ticker: ETF ticker symbol
        cache_dir: Optional cache directory
    
    Returns:
        List of stock tickers
    """
    if cache_dir is None:
        cache_dir = Path('data/cache/etf_holdings')
    
    # Try cache first
    cached = get_cached_holdings(etf_ticker, cache_dir)
    if cached:
        print(f"  Using cached holdings for {etf_ticker}")
        return cached
    
    # Try yfinance
    holdings = get_etf_top_holdings_yfinance(etf_ticker)
    
    # Fallback to manual list
    if not holdings and etf_ticker in FALLBACK_HOLDINGS:
        print(f"  Using fallback holdings for {etf_ticker}")
        holdings = FALLBACK_HOLDINGS[etf_ticker]
    
    # Cache results
    if holdings:
        save_holdings_cache(etf_ticker, holdings, cache_dir)
    
    return holdings


def discover_universe_from_etfs(etf_tickers: List[str], cache_dir: Path = None) -> List[str]:
    """
    Discover stock universe from multiple sector ETFs
    
    Args:
        etf_tickers: List of ETF tickers to analyze
        cache_dir: Optional cache directory
    
    Returns:
        Combined list of unique stock tickers
    """
    print(f"\nDiscovering universe from {len(etf_tickers)} ETFs...")
    
    all_holdings = []
    
    for etf in etf_tickers:
        holdings = get_etf_holdings(etf, cache_dir)
        if holdings:
            print(f"  {etf}: {len(holdings)} holdings")
            all_holdings.extend(holdings)
        else:
            print(f"  {etf}: No holdings found")
    
    # Return unique tickers, sorted
    unique_tickers = sorted(list(set(all_holdings)))
    print(f"\nTotal unique tickers: {len(unique_tickers)}")
    
    return unique_tickers
