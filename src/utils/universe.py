"""
Stock universe for mining & materials sector
Dynamically built from ETF holdings: SILJ, COPX, GDX, GDXJ, XLB, XME, PICK, REMX, SLX
Maps foreign tickers to US-traded equivalents using Polygon API and OTCMKT screener
"""

import os
import pandas as pd
import logging
import requests
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Polygon API configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
POLYGON_BASE_URL = "https://api.polygon.io"

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
        
        # Map different column name variations for ticker
        if 'ticker' in df.columns:
            df['Ticker'] = df['ticker']
        elif 'symbol' in df.columns:
            df['Ticker'] = df['symbol']
        
        # Map different column name variations for company name
        # Priority order: holding name > name > security name
        if 'holding name' in df.columns:
            df['Name'] = df['holding name']
        elif 'name' in df.columns:
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

def _extract_company_name(name):
    """Clean company name for matching"""
    if pd.isna(name):
        return ""
    
    name = str(name).upper().strip()
    
    # Remove common suffixes only at the end of the string
    # Use word boundaries to avoid removing parts of words
    suffixes = [
        ' LIMITED', ' LTD', ' LTD.', 
        ' CORPORATION', ' CORP', ' CORP.',
        ' INCORPORATED', ' INC', ' INC.',
        ' COMPANY', ' CO', ' CO.',
        ' PUBLIC LIMITED COMPANY', ' PLC', ' PLC.',
        ' SOCIEDAD ANONIMA', ' SA', ' S.A.',
        ' AKTIEBOLAG', ' AB',
        ' NAAMLOZE VENNOOTSCHAP', ' NV', ' N.V.',
        ' AKTIENGESELLSCHAFT', ' AG',
        ' SOCIETAS EUROPAEA', ' SE', ' S.E.',
        ' LIMITED LIABILITY COMPANY', ' LLC', ' L.L.C.',
        ' LIMITED PARTNERSHIP', ' LP', ' L.P.',
        ' LIMITED LIABILITY PARTNERSHIP', ' LLP', ' L.L.P.',
        ' GROUP', ' THE'
    ]
    
    # Sort by length (longest first) to avoid partial matches
    suffixes.sort(key=len, reverse=True)
    
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
            break  # Only remove one suffix
    
    return name

def _lookup_polygon_ticker(company_name, original_ticker):
    """
    Look up US ticker for a company using Polygon API
    
    Args:
        company_name: Company name from ETF holdings
        original_ticker: Original ticker from holdings (for logging)
    
    Returns:
        US ticker symbol or None
    """
    if not POLYGON_API_KEY:
        logger.debug("Polygon API key not configured")
        return None
    
    if not company_name or pd.isna(company_name):
        return None
    
    original_name = str(company_name).upper().strip()[:50]
    clean_name = _extract_company_name(company_name)[:50]
    
    # Try both original and cleaned names - prioritize original
    search_queries = []
    if original_name:
        search_queries.append(('original', original_name))
    if clean_name and clean_name != original_name:
        search_queries.append(('cleaned', clean_name))
    
    for query_type, search_name in search_queries:
        # Polygon search endpoint
        url = f"{POLYGON_BASE_URL}/v3/reference/tickers"
        params = {
            'search': search_name,
            'active': 'true',
            'limit': 10,
            'apiKey': POLYGON_API_KEY
        }
        
        print(f"\n📤 Polygon API Request for {original_ticker} ({query_type} name):")
        print(f"   Company Name: {company_name}")
        print(f"   Search Query: {search_name}")
        print(f"   URL: {url}")
        
        max_retries = 10000
        retry_count = 0
        response = None
        
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                
                print(f"📥 Polygon API Response:")
                print(f"   Status Code: {response.status_code}")
                
                # Handle rate limiting - pause and retry
                if response.status_code == 429:
                    retry_count += 1
                    print(f"   ⚠️  Rate limit hit, pausing 13 seconds and retrying... (attempt {retry_count}/{max_retries})")
                    logger.warning(f"Polygon API rate limit hit for {search_name}, retry {retry_count}")
                    time.sleep(13)  # Wait slightly longer than minimum
                    continue  # Retry the loop
                
                if response.status_code != 200:
                    print(f"   ❌ Error response")
                    logger.debug(f"Polygon API error {response.status_code} for {search_name}")
                    break  # Try next search query
                
                # Success - break out of retry loop
                break
                
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Request Exception: {e}")
                logger.debug(f"Polygon API request failed for {search_name}: {e}")
                break  # Try next search query
            except Exception as e:
                print(f"   ❌ Error: {e}")
                logger.debug(f"Polygon API error for {search_name}: {e}")
                break  # Try next search query
        
        if retry_count >= max_retries or response is None or response.status_code != 200:
            if retry_count >= max_retries:
                print(f"   ❌ Max retries exceeded")
            continue  # Try next search query
        
        try:
            data = response.json()
            results = data.get('results', [])
            
            print(f"   Results Count: {len(results)}")
            
            if results:
                print(f"   Top Results:")
                for i, result in enumerate(results[:5], 1):
                    ticker = result.get('ticker', 'N/A')
                    name = result.get('name', 'N/A')
                    locale = result.get('locale', 'N/A')
                    market = result.get('market', 'N/A')
                    exchange = result.get('primary_exchange', 'N/A')
                    print(f"      {i}. {ticker:8s} | {name[:40]:40s} | {locale:4s} | {market:8s} | {exchange}")
            
            if not results:
                print(f"   ℹ️  No results found")
                continue  # Try next search query
            
            # Find best match: prioritize exact name matches on US exchanges
            for result in results:
                result_name = result.get('name', '').upper()
                ticker = result.get('ticker', '')
                market = result.get('market', '')
                locale = result.get('locale', '')
                
                # Only accept US locale with stocks or otc market
                if locale != 'us' or market not in ['stocks', 'otc']:
                    continue
                
                # Check for exact or very close match
                if search_name.upper() in result_name or result_name in search_name.upper():
                    print(f"   ✅ Match Found: {ticker} ({result_name}) [{market.upper()}] [using {query_type} name]")
                    logger.debug(f"Polygon: {original_ticker} ({company_name}) -> {ticker} ({result_name})")
                    return ticker
            
            # If no exact match, return first US stocks/otc result as fallback
            for result in results:
                locale = result.get('locale', '')
                market = result.get('market', '')
                if locale == 'us' and market in ['stocks', 'otc']:
                    ticker = result.get('ticker', '')
                    result_name = result.get('name', '')
                    print(f"   ⚠️  Fallback Match: {ticker} ({result_name}) [{market.upper()}] [using {query_type} name]")
                    logger.debug(f"Polygon (fallback): {original_ticker} ({company_name}) -> {ticker}")
                    return ticker
            
            print(f"   ❌ No US stocks/OTC found in results")
            # Continue to next search query if this one didn't match
            
        except Exception as e:
            print(f"   ❌ Error parsing response: {e}")
            logger.debug(f"Polygon API error for {search_name}: {e}")
            continue
    
    # No matches found with either query
    return None

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
    
    if not company_name or pd.isna(company_name):
        return None
    
    original_name = str(company_name).upper().strip()
    clean_name = _extract_company_name(company_name)
    
    # Search for matching company in screener
    # Try ORIGINAL name first, then cleaned name
    
    # Attempt 1: Exact match with original name
    matches = screener_df[screener_df['Security Name'].str.upper().str.strip() == original_name]
    
    # Attempt 2: Partial match with original name
    if matches.empty:
        matches = screener_df[screener_df['Security Name'].str.upper().str.contains(original_name, na=False, regex=False)]
    
    # Attempt 3: Exact match with cleaned name (if different from original)
    if matches.empty and clean_name and clean_name != original_name:
        screener_df['clean_name'] = screener_df['Security Name'].apply(_extract_company_name)
        matches = screener_df[screener_df['clean_name'] == clean_name]
    
    # Attempt 4: Partial match with cleaned name
    if matches.empty and clean_name:
        if 'clean_name' not in screener_df.columns:
            screener_df['clean_name'] = screener_df['Security Name'].apply(_extract_company_name)
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
       - Look up company name via Polygon API to find US ticker
       - If not found, look up in Stock_Screener.csv
       - If still not found, discard
    """
    logger.info("Loading ETF holdings and stock screener...")
    
    holdings_df = _load_etf_holdings()
    screener_df = _load_stock_screener()
    
    us_tickers = set()
    polygon_mapped = 0
    screener_mapped = 0
    discarded = 0
    api_calls = 0
    processed = 0
    
    # Group by ticker to avoid duplicates across ETFs
    # Use custom aggregation to prefer non-NaN names
    def get_first_non_nan(x):
        d = x.dropna()
        return d.iloc[0] if not d.empty else None
    
    unique_holdings = holdings_df.groupby('Ticker').agg({
        'Name': get_first_non_nan,
        'etf': 'first'  # Just take first ETF for reference
    }).reset_index()
    
    logger.info(f"Processing {len(unique_holdings)} unique tickers from {len(holdings_df)} total holdings")
    logger.info("Using Polygon API for company name lookups (may take a few minutes)...")
    
    for idx, row in unique_holdings.iterrows():
        processed += 1
        ticker = row['Ticker']
        name = row.get('Name', '')
        
        # Convert ticker to string for display purposes
        ticker = str(ticker).strip() if not pd.isna(ticker) else 'N/A'
        
        # Step 1: Look up via Polygon API by company name
        us_ticker = None
        if name and not pd.isna(name):
            us_ticker = _lookup_polygon_ticker(name, ticker)
            api_calls += 1
            
            # Rate limiting for free tier (5 calls/minute)
            if api_calls % 5 == 0:
                logger.debug(f"Processed {api_calls} Polygon API calls, pausing...")
                time.sleep(12)  # 12 seconds between batches of 5
        
        if us_ticker:
            us_tickers.add(us_ticker)
            polygon_mapped += 1
            print(f"✓ [{processed}/{len(unique_holdings)}] {ticker} → {us_ticker} (Polygon)")
            continue
        
        # Step 2: Look up in Stock_Screener.csv by company name
        screener_ticker = _find_us_ticker(name, ticker, screener_df)
        if screener_ticker:
            us_tickers.add(screener_ticker)
            screener_mapped += 1
            print(f"✓ [{processed}/{len(unique_holdings)}] {ticker} → {screener_ticker} (Stock_Screener)")
            logger.debug(f"Stock_Screener: {ticker} ({name}) -> {screener_ticker}")
            continue
        
        # Step 3: Discard if not found
        discarded += 1
        print(f"✗ [{processed}/{len(unique_holdings)}] {ticker} ({name}) - No US ticker found")
        logger.debug(f"No US ticker found for {ticker} ({name})")
    
    logger.info(f"Universe built: {len(us_tickers)} tickers")
    logger.info(f"  - {polygon_mapped} tickers found via Polygon API")
    logger.info(f"  - {screener_mapped} tickers found via Stock_Screener.csv")
    logger.info(f"  - {discarded} tickers discarded (no US equivalent)")
    logger.info(f"  - {api_calls} Polygon API calls made")
    
    return sorted(us_tickers)

def get_universe():
    """
    Returns the stock universe as a sorted list of tickers
    Caches results to avoid rebuilding on every call
    Delete results/universe_cache.txt to force rebuild
    """
    # Check for cached universe
    cache_file = Path(__file__).parent.parent.parent / 'results' / 'universe_cache.txt'
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use cache if it exists
    if cache_file.exists():
        logger.info(f"Loading universe from cache")
        with open(cache_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from cache")
        return tickers
    
    # Build universe from scratch
    logger.info("No cache found, building universe from ETF holdings...")
    tickers = _parse_tickers()
    
    # Save to cache
    with open(cache_file, 'w') as f:
        f.write('\n'.join(tickers))
    logger.info(f"Universe cached to {cache_file}")
    
    return tickers

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

