# Project Architecture & Implementation Summary

## Overview

A production-ready sentiment analysis system for predicting mining & materials stock performance using multi-source text data.

## Project Structure

```
stocksentiment/
├── src/
│   ├── data/                    # Data collection modules
│   │   ├── reddit_collector.py  # Reddit API integration
│   │   ├── news_collector.py    # NewsAPI + Finnhub
│   │   ├── earnings_collector.py # Earnings data via Alpha Vantage
│   │   └── sec_collector.py     # SEC filings (8-K, 10-Q, 10-K)
│   │
│   ├── sentiment/               # Sentiment analysis
│   │   ├── finbert_analyzer.py  # FinBERT model wrapper
│   │   ├── domain_lexicon.py    # Mining-specific lexicons
│   │   └── aggregator.py        # Sentiment aggregation
│   │
│   ├── features/                # Feature engineering
│   │   ├── sentiment_features.py # Rolling, momentum, spike features
│   │   └── technical_features.py # Price, volatility, RSI
│   │
│   ├── models/                  # ML models
│   │   ├── baseline.py          # Logistic/Linear regression
│   │   ├── xgboost_ranker.py    # XGBoost ranking model
│   │   └── portfolio.py         # Portfolio construction & backtest
│   │
│   ├── evaluation/              # Evaluation metrics
│   │   └── metrics.py           # IC, Sharpe, hit rate, etc.
│   │
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration manager
│       ├── universe.py          # Stock universe definition
│       └── data_utils.py        # Data I/O utilities
│
├── examples/
│   └── sentiment_demo.py        # Quick demo script
│
├── data/                        # Data storage (gitignored)
├── models/                      # Saved models (gitignored)
├── results/                     # Predictions & backtests (gitignored)
├── main.py                      # Main orchestration pipeline
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Project documentation
└── QUICKSTART.md               # Quick start guide
```

## Key Design Decisions

### 1. Data Collection Strategy

**Multi-source approach** to capture diverse sentiment signals:

- **Reddit**: Retail investor sentiment (r/investing, r/stocks, r/commodities)
- **News**: Professional media coverage (NewsAPI, Finnhub)
- **Earnings**: Fundamental surprises (Alpha Vantage)
- **SEC Filings**: Material events (SEC EDGAR)

**Time alignment**:
- All timestamps aligned to market close (4 PM ET)
- Prevents look-ahead bias from intraday sentiment
- Forward returns calculated post-sentiment

### 2. Sentiment Analysis Architecture

**Two-layer sentiment scoring**:

1. **FinBERT**: General financial sentiment (positive/negative/neutral)
2. **Domain Lexicon**: Mining-specific features
   - Supply risk (strikes, shutdowns, outages)
   - Demand signals (China PMI, EV demand)
   - Cost pressures (energy, labor)
   - Regulatory risk (permits, ESG)
   - Production sentiment (guidance, output)

**Combined score**: Weighted average emphasizing general sentiment (50%) + domain factors (50%)

### 3. Feature Engineering

**Sentiment features**:
- Rolling windows (1d, 5d, 21d)
- Momentum (rate of change)
- Volatility (std dev)
- Spikes (z-score > 2)
- Volume-weighted sentiment

**Technical features**:
- Forward returns (1d, 5d, 21d) - **targets**
- Historical returns - features
- Volatility, RSI, SMA
- Relative to benchmark (XLB)

**Critical**: No forward-looking features in training data (prevents leakage)

### 4. Modeling Approach

**Baseline**: Logistic regression
- Target: Binary outperform/underperform benchmark
- Purpose: Establish minimum viable signal

**Advanced**: XGBoost Regressor
- Target: Continuous 5-day forward return
- Ranking-based (not absolute prediction)
- Hyperparameters from config.yaml

**Evaluation metrics**:
- **Information Coefficient (IC)**: Primary metric (Spearman correlation)
- Hit rate, Sharpe ratio, max drawdown
- **IC > 0.05** = usable signal, **IC > 0.1** = strong signal

### 5. Portfolio Construction

**Long-short strategy**:
- Long: Top 20% by predicted return
- Short: Bottom 20% by predicted return
- Weekly rebalancing

**Backtest includes**:
- Transaction costs (10 bps)
- Slippage considerations
- Sector-neutral returns vs benchmark

## Data Flow

```
[APIs] → [Collectors] → [Raw Data]
           ↓
    [Sentiment Analysis]
           ↓
    [Aggregation by ticker/date]
           ↓
    [Feature Engineering] ← [Price Data]
           ↓
    [Train/Test Split]
           ↓
    [Model Training]
           ↓
    [Predictions] → [Portfolio] → [Backtest]
```

## Key Algorithms

### Sentiment Aggregation

```python
combined_sentiment = 0.5 * finbert_score +
                     0.2 * demand_score +
                     0.1 * supply_risk_score +
                     0.1 * cost_pressure_score +
                     0.1 * production_sentiment
```

### Feature Scaling

- StandardScaler for all features
- Fit on training set only
- Transform both train/test

### Time-Series Split

- Temporal train/test split (70/30)
- No shuffling (preserves time ordering)
- Prevents future information leakage

## Configuration System

All key parameters in `config.yaml`:

- **Universe**: Tickers, segments, benchmarks
- **Data collection**: Sources, keywords, lookback
- **Sentiment**: Model names, lexicon keywords
- **Features**: Time windows, aggregations
- **Modeling**: Hyperparameters, targets
- **Evaluation**: Metrics, backtest settings

## Error Handling

- Try-catch blocks around API calls
- Rate limiting (sleep between requests)
- Graceful degradation (continue if one source fails)
- Logging of errors and warnings

## Performance Considerations

1. **Caching**: Data collection results cached for 1 day
2. **Batch processing**: FinBERT processes in batches of 32
3. **Parallel data collection**: Could be parallelized (not implemented)
4. **GPU support**: FinBERT uses CUDA if available

## Extensibility Points

1. **New data sources**: Add collector in `src/data/`
2. **Custom features**: Extend feature engineers
3. **New models**: Implement in `src/models/`
4. **Different targets**: Configure in `config.yaml`
5. **Segment-specific models**: Train per segment (gold vs copper)

## Mining Industry Specifics

**Key factors addressed**:

1. **Supply shocks matter**: Strike detection, production disruptions
2. **Commodity ≠ company**: Segment-based analysis (gold ≠ lithium)
3. **Geopolitical sensitivity**: China demand, trade tensions
4. **Small-cap volatility**: Mid-caps included for higher signal
5. **ESG importance**: Regulatory and environmental keywords

## Production Considerations

**Not included** (would be needed for production):

- Real-time data streaming
- Automated daily execution (cron/scheduler)
- Database integration (currently uses Parquet files)
- API rate limit management (advanced)
- Model monitoring and retraining
- Alerts and notifications
- Dashboard/visualization

**Security**:

- API keys in `.env` (not committed)
- `.gitignore` for sensitive data
- No hardcoded credentials

## Known Limitations

1. **Free API tiers**: Limited data volume
2. **Historical backtest**: Not walk-forward
3. **Transaction costs**: Simplified model
4. **Market impact**: Not modeled
5. **Survivorship bias**: Not addressed
6. **Sector rotation**: Not explicitly modeled

## Performance Expectations

**Realistic targets**:

- IC: 0.05 - 0.15 (decent to strong)
- Hit rate: 55% - 60%
- Sharpe: 1.0 - 2.0
- Max drawdown: -10% to -20%

**Commodities are noisy**: Don't expect equity-like Sharpe ratios

## Testing

**Recommended validation**:

1. Run `sentiment_demo.py` first
2. Start with `--days 7` for quick iteration
3. Check data quality before training
4. Compare to buy-and-hold benchmark
5. Test on different time periods

## Next Steps for Users

1. **Get API keys** (Reddit, NewsAPI, Alpha Vantage)
2. **Run demo** to verify setup
3. **Collect 30 days** of data
4. **Train initial model**
5. **Analyze feature importance**
6. **Iterate on features** based on results
7. **Consider segment-specific models**
8. **Add custom lexicon keywords**

## Conclusion

This is a **research-grade** sentiment analysis system with:

✅ Production-quality code structure
✅ Comprehensive feature engineering
✅ Proper evaluation methodology
✅ Mining industry specifics
✅ Extensible architecture

Not included (by design):
❌ Production deployment infrastructure
❌ Real-time execution
❌ Advanced portfolio optimization

Perfect for: Academic research, strategy development, proof-of-concept
Requires work for: Live trading, production deployment
