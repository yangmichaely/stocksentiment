"""
Stock universe for mining & materials sector
Hardcoded from ETF holdings: SILJ, COPX, GDX, GDXJ, XLB, XME, PICK, REMX, SLX
Includes ADRs for major foreign miners
"""

# ADR mappings: Foreign ticker → US-traded equivalent
ADR_MAPPINGS = {
    # Major miners (dual-listed or ADRs)
    'BHP AU': 'BHP',           # BHP Group (NYSE/ASX dual-listed)
    'RIO AU': 'RIO',           # Rio Tinto (NYSE/LSE/ASX)
    'VALE3': 'VALE',           # Vale (NYSE)
    'ANTO LN': 'ANFGY',        # Antofagasta (OTC ADR)
    'GLEN LN': None,           # Glencore - no US listing
    'AAL': 'AAL',              # Anglo American (OTC: AAUKF, keep as AAL)
    
    # Australian miners
    'NST AU': 'NESRF',         # Northern Star (OTC)
    'EVN AU': 'EVNGY',         # Evolution Mining (OTC ADR)
    'FMG AU': 'FSUGY',         # Fortescue Metals (OTC ADR)
    'BSL AU': None,            # Bluescope - limited US trading
    'MIN AU': None,            # Mineral Resources
    'IGO AU': None,            # IGO Limited
    'LYC AU': 'LYSCF',         # Lynas Rare Earths (OTC)
    'PLS AU': 'PILBF',         # Pilbara Minerals (OTC)
    'ILU AU': None,            # Iluka Resources
    
    # South African miners
    'GFI': 'GFI',              # Gold Fields (NYSE)
    'HMY': 'HMY',              # Harmony Gold (NYSE)
    'SIBANYE': 'SBSW',         # Sibanye-Stillwater
    
    # Canadian miners (TSX) - many dual-listed on NYSE/NASDAQ
    'ARTG CN': 'ARTG',         # Artemis Gold (NASDAQ)
    'AYA CN': 'AYA',           # Aya Gold & Silver
    'LUCA CN': 'LUCA',         # Lucara Diamond
    'LUN CN': 'LUNMF',         # Lundin Mining (OTC)
    'HBM CN': 'HBM',           # Hudbay Minerals (NYSE)
    'FM CN': 'FM',             # First Quantum (NASDAQ)
    'TECK.B CN': 'TECK',       # Teck Resources (NYSE)
    'IVN CN': 'IVPAF',         # Ivanhoe Mines (OTC)
    'CS CN': 'CSHHF',          # Capstone Copper (OTC)
    'ERO CN': 'ERRPF',         # Ero Copper (OTC)
    'NGEX CN': 'NGEXF',        # NGEx Minerals (OTC)
    'FOM CN': None,            # Foran Mining
    'ALS CN': None,            # Altius Minerals
    'NDM CN': 'NAK',           # Northern Dynasty (NYSE American)
    'CNL CN': None,            # Contact Gold
    'SLS CN': None,            # Solaris Resources
    'OGC CN': 'OGC',           # OceanaGold (NASDAQ)
    'DPM CN': 'DPM',           # Dundee Precious Metals
    'LUG CN': 'LUGDF',         # Lundin Gold (OTC)
    'DSV CN': None,            # Discovery Silver
    'GMIN CN': None,           # G Mining Ventures
    'TXG CN': 'TXG',           # Torex Gold (OTC)
    'KNT CN': 'KNT',           # K92 Mining
    'SKE CN': 'SKE',           # Skeena Resources
    'ABRA CN': None,           # Abra Mining
    'SCZ CN': None,            # Santacruz Silver
    'WRN CN': 'WRN',           # Western Copper & Gold
    'USA CN': 'USAMF',         # Americas Gold & Silver (OTC)
    
    # Other mappings
    'MT': 'MT',                # ArcelorMittal (NYSE)
    'SCCO': 'SCCO',            # Southern Copper (NYSE)
    'SSAB B': None,            # SSAB - Swedish, limited US access
}

# Raw ticker list from ETF holdings (includes country codes)
RAW_TICKERS = """
HL AG CDE WPM EXK KGH PW PPTA BOL SS PAAS BVN SSRM SA OR ARTG CN SVM HYMC AYA CN TMQ TFPM FNV FSM SKE CN RGLD VZLA FRES LN ASM HOC LN USA CN PE&OLES* MM MUX ABRA CN SCZ CN WRN IAUX KCN AU GGD CN AGPXX SVL AU FF CN HSTR CN GSVR CN APM CN DV CN SVRS CN AAG CN OM CN PZG TUD CN MKR AU VOLCABC1 PE AGMR CN CUU CN IPT CN CKG CN FPC CN WAM CN MFRISCOA MF CAD PEN WVM CN AMM CN PML CN TV CN
LUCA CN KGH PW 5713 JP LUN CN BOL SS GLEN LN SCCO FCX HBM CN ANTO LN FM CN 2899 HK BHP AU TECK/B CN IVN CN 2099 HK SFR AU 358 HK CS CN NDA GR 1208 HK 5711 JP ERO CN NGEX CN 1258 HK TGB 3939 HK IE AMAK AB ATYM LN 1515 JP FOM CN ALS CN NDM CN CNL CN SLS CN DVP AU SOLG LN CAML LN LUNR CN WA1 AU
AEM US NEM US B US GFI US WPM US AU US FNV US KGC US PAAS US NST AU RGLD US AGI US PE&OLES* MF EQX US HL US EVN AU CDE US AG US EDV LN IAG US FRES LN HMY US NGD US EGO US OGC CN 1818 HK DPM CN LUG CN BVN US BTG US AMMN IJ OR US RMS AU GMIN CN DSV CN BRMS IJ ARTG CN SSRM US PRU AU TXG CN GMD AU KNT CN CMM AU ORLA US WGX AU FSM US SA US WDO CN
PAAS US CDE US AGI US RGLD US EQX US PE&OLES* MF HL US EVN AU AG US EDV LN IAG US HMY US NGD US EGO US OGC CN 1818 HK DPM CN LUG CN BVN US BTG US OR US RMS AU GMIN CN DSV CN BRMS IJ GGP AU ARTG CN SSRM US PRU AU TXG CN GMD AU KNT CN VAU AU CMM AU EXK US ORLA US WGX AU ARMN US RRL AU SKE US CGAU US HOC LN AURA33 BZ TFPM US PPTA US SA US AAUC CN WDO CN 3939 HK EMR AU NG US SVM US AYA CN WAF AU PAF LN VZLA US ASM US RSG AU CYL AU KOZAL TI MUX US IAUX US OBM AU GGD CN BGL AU KCN AU DRD US 6693 HK PNR AU ALK AU GROY US BC8 AU KOZAA TI DC US IDR US MTA US NFG CN SBM AU MEK AU CMCL US GLDG US CTGO US LUCA CN AMI AU FFX AU BGL US FMR CN NIX CN ASM AU VGCX CN
LIN NEM FCX CRH SHW ECL CTVA NUE APD MLM VMC PPG STLD SW ALB IP AMCR PKG DOW DD IFF BALL CF AVY LYB MOS
CDE HL UEC FCX RGLD AA NEM BTU RS HCC CNR LEU CMC CLF NUE MP STLD CENX AMR IE USAR MTRN MUX UAMY KALU METC CMP URG SXC IDR WS LTBR ABAT RYI MTUS
BHP RIO FCX GLEN VALE3 AAL NUE RIO GMEXICOB VAL 1211 TECK.B MT FMG STLD FM BOL IMP ANTO 5401 LUN SCCO 5490 RS TATASTEEL 5713 AA HINDALCO KGH S32 SSW 1378 NHY 3993 VEDL IVN HBM NPH PLS CS LYC 5016 BSL MP CMC JSWSTEEL CLF 5706 5411 2002 MIN PMETAL 2600 SFR 358 TKA XTSLA BVN AMMN 1208 NDA USD GGBR4 5406 SSAB B VOE HCC JINDALSTEL IGO APLAPOLLO 603993 NATIONALUM ERO 5711 BRMS 5714 1258 TKO NGEX MTRN CSTM JSL NMDC KTY CENX ACX LTR 5444 AMR 2027 HILS HINDCOPPER SGM 5741 EREGL.E OUT1V KALU VZLA 4020 APAM USA 3939 IE MDKA GRNG AII SSAB A ARI 5463 PRN 601600 600111 ILU ALLEI BEKB 5471 CIA IPX GOAU4 TMC 103140 FOM QAMC NIC LIF ATYM IMD 600019 ALS DRR PTRO AMG WS SZG 600362 CSNA3 LAC BRAP4 1322 VUL 807 600010 CNL DVP PTCIL 5703 USAR 1515 630 2006 UAMY 426 SLS 975 MDI CMP 5451 3858 639 2015 SGML 600219 601168 600549 1430 RYI 2532 INCO SMR MTUS METC ERA VIO WELCORP NB 9958 600673 933 SXC USHAMART GPIL WA1 TMQ TRMET.E USIM5 CAP RATNAMANI 826 1321 5482 601958 GRAVITA 2211 KRDMD.E 5423 323 ABAT 831 2014 688122 5009 SARDAEN 532527 708 JINDALSAW BRSAN.E JSW 2023 932 VSL NSLNISP AUD 2240 500245 SANDUMA 5440 3833 MOIL 600985 BRE FESA4 5449 1320 MAHSEAMLES MIDHANI SURYAROSNI JAIBALAJI 2362 TTST SALFT EUR PLN ZAR NOK HKD MXN CNH SEK GBP CAD SGD CHF TRY RUB JPY ALRS RUAL 3030 CNY IDR PHP GMKN 773 NLMK LLL MTLRP CHMF
ALB US 600111 C1 600549 C1 LYC AU MP US 601958 C1 SQM US PLS AU 600392 C1 LTR AU 1772 HK AII CN AMG NA IPX US LAC US ILU AU LAR US 603067 C1 SLI US ERA FP 600456 C1 SGML US VUL AU TROX US AVZ AU
RIO US BHP US VALE US RIO AU NUE US FMG AU MT US RS US PKX US 5401 JP STLD US 2002 TT 5411 JP TS US CLF US BSL AU CMC US MIN AU GGB US SSABB SS 2027 TT ACX SM EREGL TI 5444 JP OUT1V FH TX US 5463 JP APAM NA SGM AU 8078 JP RUS CN 004020 KS LIF CN SID US KIO SJ 2015 TT WS US VSVS LN
"""

def _parse_tickers():
    """Parse raw ticker list, map foreign stocks to ADRs, and include US/Canadian stocks"""
    all_tickers = set()
    
    for ticker in RAW_TICKERS.split():
        # Skip currency codes
        if ticker in ['EUR', 'PLN', 'ZAR', 'NOK', 'HKD', 'MXN', 'CNH', 'SEK', 'GBP', 
                      'CAD', 'SGD', 'CHF', 'TRY', 'RUB', 'JPY', 'CNY', 'IDR', 'PHP', 'AUD', 'USD']:
            continue
        
        if ticker.isdigit():
            continue
        
        # Handle tickers with country codes
        parts = ticker.split()
        
        if len(parts) == 2:
            symbol, country = parts
            foreign_ticker = f"{symbol} {country}"
            
            # Check if there's an ADR mapping
            if foreign_ticker in ADR_MAPPINGS:
                adr = ADR_MAPPINGS[foreign_ticker]
                if adr:  # Only add if mapping exists (skip None values)
                    all_tickers.add(adr)
                continue
            
            # Keep US and Canadian stocks (traded on North American exchanges)
            if country in ['US', 'CN']:
                # Clean up special characters
                symbol = symbol.replace('/', '.').replace('*', '')
                if symbol and not symbol.isdigit():
                    all_tickers.add(symbol)
        
        elif len(parts) == 1:
            # No country code - check if it's a known ADR or US stock
            symbol = parts[0]
            
            # Check if it's in ADR mappings (some have no country code)
            if symbol in ADR_MAPPINGS:
                adr = ADR_MAPPINGS[symbol]
                if adr:
                    all_tickers.add(adr)
                continue
            
            # Clean up special characters
            symbol = symbol.replace('/', '.').replace('*', '')
            
            # Skip if it's a foreign exchange code pattern (2 letters + 2 letters)
            if len(symbol) == 4 and symbol[:2].isalpha() and symbol[2:].isalpha():
                continue
            
            # Skip pure numeric or very short codes likely to be foreign
            if symbol.isdigit() or len(symbol) <= 1:
                continue
            
            # Skip obvious foreign patterns (ends with numbers like 5713, 2899, etc.)
            if symbol and symbol[-1].isdigit() and len(symbol) > 2:
                # Check if it's mostly numbers (foreign ticker pattern)
                if sum(c.isdigit() for c in symbol) >= len(symbol) / 2:
                    continue
            
            if symbol:
                all_tickers.add(symbol)
    
    return sorted(list(all_tickers))


# Pre-parsed universe
UNIVERSE = _parse_tickers()

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
    return UNIVERSE.copy()


def get_benchmark_etfs():
    """
    Get benchmark ETF tickers
    
    Returns:
        List of ETF symbols
    """
    return BENCHMARK_ETFS.copy()


def classify_by_segment(tickers):
    """
    Classify tickers by mining segment
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dict mapping segment names to lists of tickers
    """
    # Simplified classification based on known ETF composition
    segments = {
        'precious_metals': ['AEM', 'NEM', 'AU', 'FNV', 'WPM', 'PAAS', 'RGLD', 'HL', 
                           'AG', 'CDE', 'KGC', 'EQX', 'GFI', 'BTG', 'OR', 'AGI',
                           'HMY', 'SBSW', 'EVNGY', 'NESRF'],
        'base_metals': ['FCX', 'SCCO', 'HBM', 'FM', 'LUNMF', 'TGB', 'TECK', 'IVPAF', 
                       'CSHHF', 'ERRPF', 'ANFGY', 'VALE'],
        'industrial_metals': ['NUE', 'STLD', 'CLF', 'CMC', 'RS', 'AA', 'CENX', 'HCC',
                             'MT', 'BHP', 'RIO', 'FSUGY'],
        'rare_earth': ['MP', 'LYSCF', 'PILBF', 'LTR', 'ILU', 'AII'],
        'materials_broad': ['LIN', 'APD', 'ECL', 'SHW', 'PPG', 'CTVA', 'ALB', 'DOW', 
                           'DD', 'IP', 'PKG', 'IFF', 'BALL', 'CF', 'MOS']
    }
    
    result = {k: [] for k in segments.keys()}
    result['other'] = []
    
    for ticker in tickers:
        classified = False
        for segment, segment_tickers in segments.items():
            if ticker in segment_tickers:
                result[segment].append(ticker)
                classified = True
                break
        
        if not classified:
            result['other'].append(ticker)
    
    return result


# For backwards compatibility
SECTOR_KEYWORDS = classify_by_segment(UNIVERSE)
