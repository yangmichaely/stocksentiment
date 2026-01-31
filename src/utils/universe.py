"""
Universe definition for mining & materials stocks
Dynamically built from sector ETF holdings
"""
from pathlib import Path
from .etf_holdings import discover_universe_from_etfs

# Benchmark ETFs for materials/mining sector
BENCHMARK_ETFS = [
    # Large-cap / Broad materials
    'XLB',    # Materials Select Sector SPDR
    
    # Mid-cap mining / materials
    'GDX',    # VanEck Gold Miners
    'COPX',   # Global X Copper Miners
    'REMX',   # VanEck Rare Earth/Strategic Metals
    'PICK',   # iShares MSCI Global Metals & Mining
    
    # Small-cap mining / materials  
    'XME',    # SPDR S&P Metals & Mining
    'SLX',    # VanEck Steel
    'GDXJ',   # VanEck Junior Gold Miners
    'SILJ',   # ETFMG Prime Junior Silver Miners
    
    # Precious metals
    'SLV',    # iShares Silver Trust
]

# Cache for universe (avoid re-fetching every time)
_UNIVERSE_CACHE = None


def get_all_tickers(refresh=False):
    """
    Get all stock tickers in the universe from ETF holdings
    
    Args:
        refresh: If True, force refresh from ETFs
    
    Returns:
        List of stock tickers
    """
    global _UNIVERSE_CACHE
    
    if _UNIVERSE_CACHE is None or refresh:
        _UNIVERSE_CACHE = discover_universe_from_etfs(BENCHMARK_ETFS)
    
    return _UNIVERSE_CACHE


def get_tradeable_tickers(refresh=False):
    """
    Get tradeable tickers (excludes ETFs)
    
    Args:
        refresh: If True, force refresh
    
    Returns:
        List of tradeable tickers
    """
    all_tickers = get_all_tickers(refresh=refresh)
    
    # Filter out ETFs and benchmarks
    return [t for t in all_tickers if t not in BENCHMARK_ETFS]


def get_benchmark_etfs():
    """Get list of benchmark ETFs"""
    return BENCHMARK_ETFS.copy()


# Sector classification based on known categories
SECTOR_KEYWORDS = {
    'precious_metals': ['NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'RGLD', 'KGC', 'IAG', 'AU', 'PAAS', 'OR', 'BTG', 'NGD', 'HL', 'AG', 'MAG', 'FSM', 'EXK', 'CDE'],
    'industrial_metals': ['FCX', 'SCCO', 'TECK', 'FM', 'HBM', 'CMCL', 'TRQ', 'CPRX'],
    'steel': ['MT', 'CLF', 'X', 'STLD', 'NUE', 'CMC'],
    'fertilizers': ['MOS', 'CF', 'NTR', 'FMC', 'CTVA'],
    'lithium': ['ALB', 'SQM', 'LAC', 'LTHM'],
    'diversified': ['BHP', 'RIO', 'VALE', 'GLNCY', 'AAL'],
    'chemicals': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'PPG', 'EMN'],
}


def get_segment(ticker):
    """
    Get the segment for a given ticker
    
    Args:
        ticker: Stock ticker
    
    Returns:
        Segment name
    """
    for segment, tickers in SECTOR_KEYWORDS.items():
        if ticker in tickers:
            return segment
    return 'other'  # default for unclassified


def refresh_universe():
    """Force refresh universe from ETF holdings"""
    return get_all_tickers(refresh=True)
