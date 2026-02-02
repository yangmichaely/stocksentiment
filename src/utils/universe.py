"""
Stock universe for mining & materials sector
Dynamically built from ETF holdings: SILJ, COPX, GDX, GDXJ, XLB, XME, PICK, REMX, SLX
Maps foreign tickers to US-traded equivalents using OTCMKT screener
"""

import os
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _load_stock_screener():
    """Load the OTCMKT stock screener CSV"""
    holdings_dir = Path(__file__).parent.parent.parent / 'holdings'
    screener_path = holdings_dir / 'Stock_Screener.csv'
    
    if not screener_path.exists():
        logger.warning(f"Stock screener not found at {screener_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(screener_path)
    # Create lookup dict: company name -> US ticker
    # Prioritize ADRs and Foreign Ordinary Shares
    return df

def _load_etf_holdings():
    """Load all ETF holdings from the holdings directory"""
    holdings_dir = Path(__file__).parent.parent.parent / 'holdings'
    etf_files = ['COPX.csv', 'GDX.csv', 'GDXJ.csv', 'PICK.csv', 'REMX.csv', 
                 'SILJ.csv', 'SLX.csv', 'XLB.csv', 'XME.csv']
    
    all_holdings = []
    
    for filename in etf_files:
        filepath = holdings_dir / filename
        if not filepath.exists():
            logger.warning(f"ETF holding file not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        
        # Normalize column names (different CSVs use different formats)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map different column name variations
        if 'ticker' in df.columns:
            df['Ticker'] = df['ticker']
        elif 'symbol' in df.columns:
            df['Ticker'] = df['symbol']
        
        if 'name' in df.columns:
            df['Name'] = df['name']
        elif 'security name' in df.columns:
            df['Name'] = df['security name']
        
        # Add ETF source column
        df['etf'] = filename.replace('.csv', '')
        all_holdings.append(df)
    
    if not all_holdings:
        raise FileNotFoundError("No ETF holding files found")
    
    combined = pd.concat(all_holdings, ignore_index=True)
    return combined

def _is_us_ticker(ticker, screener_df=None):
    """
    Determine if a ticker is already US-listed (NYSE/NASDAQ)
    
    Logic:
    - Has " US" suffix → US exchange ticker (keep directly)
    - Has any other country suffix including " CN" (Canadian) → foreign exchange ticker (look up in OTC)
    - No suffix → assume US exchange listing (NYSE/NASDAQ) like BHP, RIO, VALE, FCX
    """
    if not ticker or pd.isna(ticker):
        return False
    
    ticker = str(ticker).strip()
    
    # If it has a space, it has a country/exchange code
    if ' ' in ticker:
        parts = ticker.split()
        if len(parts) >= 2:
            country_code = parts[-1]
            # Only " US" suffix means US exchange
            if country_code == 'US':
                return True
            # Everything else is foreign exchange listing (including Canadian " CN")
            return False
    
    # No suffix - skip numeric or single-char tickers
    if ticker.isdigit() or len(ticker) <= 1:
        return False
    
    # No suffix = assume US exchange listing (NYSE/NASDAQ/AMEX)
    # Examples: BHP, RIO, VALE (NYSE ADRs), FCX, NUE (US companies)
    return True


def _extract_company_name(name):
    """Clean company name for matching"""
    if pd.isna(name):
        return ""
    
    name = str(name).upper()
    # Remove common suffixes
    for suffix in [' LTD', ' LIMITED', ' CORP', ' CORPORATION', ' INC', ' PLC', 
                   ' SA', ' AB', ' NV', ' AG', ' SE', ' CO', ' GROUP']:
        name = name.replace(suffix, '')
    
    return name.strip()

def _find_us_ticker(company_name, ticker_hint, screener_df):
    """
    Find the US-traded version of a foreign stock using the OTCMKT screener
    
    Args:
        company_name: Name of the company from ETF holdings
        ticker_hint: Original ticker (for logging/debugging)
        screener_df: OTCMKT screener dataframe
    
    Returns:
        US ticker symbol or None
    """
    if screener_df.empty:
        return None
    
    clean_name = _extract_company_name(company_name)
    if not clean_name:
        return None
    
    # Search for matching company in screener
    # Look for ADRs first, then Foreign Ordinary Shares
    screener_df['clean_name'] = screener_df['Security Name'].apply(_extract_company_name)
    
    # Exact match
    matches = screener_df[screener_df['clean_name'] == clean_name]
    
    # If no exact match, try partial match (contains)
    if matches.empty:
        matches = screener_df[screener_df['clean_name'].str.contains(clean_name, na=False, regex=False)]
    
    if matches.empty:
        return None
    
    # Prioritize ADRs over ordinary shares
    adrs = matches[matches['Sec Type'] == 'ADRs']
    if not adrs.empty:
        return adrs.iloc[0]['Symbol']
    
    # Next priority: Foreign Ordinary Shares
    foreign_ord = matches[matches['Sec Type'] == 'Foreign Ordinary Shares']
    if not foreign_ord.empty:
        return foreign_ord.iloc[0]['Symbol']
    
    # Fallback to first match
    if not matches.empty:
        return matches.iloc[0]['Symbol']
    return None

def _parse_tickers():
    """
    Parse ETF holdings and map to US-traded tickers
    
    Process:
    1. Load all ETF holdings
    2. For each ticker:
       - If US-based: include directly
       - If foreign: look up company name in OTCMKT screener for US ticker
       - Skip if no US version found
    """
    logger.info("Loading ETF holdings and stock screener...")
    
    holdings_df = _load_etf_holdings()
    screener_df = _load_stock_screener()
    
    us_tickers = set()
    foreign_mapped = 0
    us_direct = 0
    discarded = 0
    
    # Group by ticker to avoid duplicates across ETFs
    # Use custom aggregation to prefer non-NaN names
    unique_holdings = holdings_df.groupby('Ticker').agg({
        'Name': lambda x: (d := x.dropna()).iloc[0] if not d.empty else None,
        'etf': 'first'  # Just take first ETF for reference
    }).reset_index()
    
    logger.info(f"Processing {len(unique_holdings)} unique tickers from {len(holdings_df)} total holdings")
    
    for _, row in unique_holdings.iterrows():
        ticker = row['Ticker']
        name = row.get('Name', '')
        
        # Check if it's a US ticker (pass screener for better detection)
        if _is_us_ticker(ticker, screener_df):
            # Clean up ticker (remove slashes, etc.)
            clean_ticker = str(ticker).split()[0]  # Take first part before any space
            clean_ticker = clean_ticker.replace('/', '.').replace('*', '')
            
            # Skip purely numeric tickers or single characters
            if clean_ticker and len(clean_ticker) > 1 and not clean_ticker.isdigit():
                us_tickers.add(clean_ticker)
                us_direct += 1
        else:
            # Foreign ticker - check if it trades on US exchange under same symbol
            # Extract base ticker (e.g., "HBM CN" -> "HBM", "TECK/B CN" -> "TECK.B")
            base_ticker = str(ticker).split()[0]
            base_ticker = base_ticker.replace('/', '.').replace('*', '')
            
            # First try to find OTC equivalent
            us_ticker = _find_us_ticker(name, ticker, screener_df)
            
            if us_ticker:
                # Found OTC mapping
                us_tickers.add(us_ticker)
                foreign_mapped += 1
                logger.debug(f"Mapped {ticker} ({name}) -> {us_ticker} (OTC)")
            elif base_ticker and len(base_ticker) > 1 and not base_ticker.isdigit():
                # No OTC mapping, but assume base ticker trades on US exchange (NYSE/NASDAQ)
                # This handles Canadian stocks like "HBM CN" -> "HBM" that are dual-listed
                us_tickers.add(base_ticker)
                foreign_mapped += 1
                logger.debug(f"Mapped {ticker} ({name}) -> {base_ticker} (assumed US exchange)")
            else:
                discarded += 1
                logger.debug(f"No US ticker found for {ticker} ({name})")
    
    logger.info(f"Universe built: {len(us_tickers)} tickers")
    logger.info(f"  - {us_direct} US/Canadian tickers added directly")
    logger.info(f"  - {foreign_mapped} foreign tickers mapped to US equivalents")
    logger.info(f"  - {discarded} tickers discarded (no US equivalent)")
    
    return sorted(us_tickers)

def get_universe():
    """
    Returns the stock universe as a sorted list of tickers
    """
    return _parse_tickers()

# Benchmark ETFs
BENCHMARK_ETFS = ['XLB', 'GDX', 'COPX', 'SLV', 'PICK', 'XME', 'SLX', 'GDXJ', 'SILJ', 'REMX']

def get_all_tickers(refresh=False):
    """
    Get complete stock universe
    
    Args:
        refresh: Ignored (kept for compatibility)
    
    Returns:
        List of ticker symbols
    """
    return get_universe()

def get_benchmark_etfs():
    """
    Get benchmark ETF tickers
    
    Returns:
        List of ETF symbols
    """
    return BENCHMARK_ETFS.copy()

def classify_by_segment(tickers):
    """
    Classify tickers by mining segment (simplified heuristic-based)
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dict mapping segment names to lists of tickers
    """
    # Common precious metals keywords
    precious_keywords = ['GOLD', 'SILVER', 'PLAT', 'PALLA', 'GLD', 'SLV', 'RGLD', 
                         'FNV', 'WPM', 'PAAS', 'HL', 'AG', 'CDE']
    
    # Common base metals keywords
    base_keywords = ['COPPER', 'COPP', 'ZINC', 'NICKEL', 'FCX', 'SCCO', 'TGB',
                     'HBM', 'FM', 'TECK']
    
    # Industrial/steel keywords
    industrial_keywords = ['STEEL', 'NUCOR', 'NUE', 'STLD', 'CLF', 'CMC', 'RS', 
                          'MT', 'AA']
    
    # Rare earth keywords
    rare_earth_keywords = ['RARE', 'EARTH', 'MP', 'LYC', 'REE']
    
    result = {
        'precious_metals': [],
        'base_metals': [],
        'industrial_metals': [],
        'rare_earth': [],
        'materials_broad': [],
        'other': []
    }
    
    for ticker in tickers:
        ticker_upper = ticker.upper()
        
        # Check keywords
        if any(kw in ticker_upper for kw in precious_keywords):
            result['precious_metals'].append(ticker)
        elif any(kw in ticker_upper for kw in base_keywords):
            result['base_metals'].append(ticker)
        elif any(kw in ticker_upper for kw in industrial_keywords):
            result['industrial_metals'].append(ticker)
        elif any(kw in ticker_upper for kw in rare_earth_keywords):
            result['rare_earth'].append(ticker)
        else:
            result['other'].append(ticker)
    
    return result

if __name__ == '__main__':
    # Test the universe
    import logging
    logging.basicConfig(level=logging.INFO)
    
    universe = get_universe()
    print(f"\nTotal universe size: {len(universe)}")
    print(f"\nSample tickers: {universe[:20]}")

