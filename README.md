# Mining & Materials Sentiment Stock Picker

A sentiment analysis-based stock picker for the mining and materials sector using multiple data sources (Reddit, news, earnings reports, SEC filings).

## 🎯 Project Goal

Use text sentiment to predict short- to medium-term stock performance of mining & materials companies (metals, aggregates, chemicals, miners).

### Outputs
- Weekly stock rankings by expected alpha
- Long-short portfolio (top vs bottom sentiment decile)
- Binary classifier: outperform vs underperform sector ETF (XLB, PICK, COPX)

## 🏭 Universe

### Sectors Covered
- **Metals & Mining** (NAICS 212)
- **Materials Sector** (GICS)
  - Precious metals (gold, silver)
  - Industrial metals (copper, lithium)
  - Steel / aluminum
  - Fertilizers / chemicals

### Key Tickers
- **Majors**: BHP, RIO, VALE, FCX, NEM, AA
- **ETF Benchmarks**: XLB, PICK, COPX

## 📊 Data Sources

1. **Earnings Call Transcripts** - Forward guidance & capex sentiment
2. **Press Releases** (8-K, PR Newswire) - Production issues, strikes, M&A
3. **News Headlines** (Reuters, Bloomberg-style feeds) - Macro + geopolitics
4. **Reddit** (r/investing, r/stocks, r/commodities) - Retail flow
5. **SEC Filings** - Material events

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repo>
cd stocksentiment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Get API Keys

1. **Reddit**: https://www.reddit.com/prefs/apps
2. **NewsAPI**: https://newsapi.org/register
3. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key

### Basic Usage

```bash
# Collect data
python main.py --mode collect --days 30

# Run sentiment analysis
python main.py --mode analyze

# Generate predictions
python main.py --mode predict

# Full pipeline
python main.py --mode full
```

## 📁 Project Structure

```
stocksentiment/
├── src/
│   ├── data/                   # Data collection modules
│   │   ├── reddit_collector.py
│   │   ├── news_collector.py
│   │   ├── earnings_collector.py
│   │   └── sec_collector.py
│   ├── sentiment/              # Sentiment analysis
│   │   ├── finbert_analyzer.py
│   │   ├── domain_lexicon.py
│   │   └── aggregator.py
│   ├── features/               # Feature engineering
│   │   ├── sentiment_features.py
│   │   └── technical_features.py
│   ├── models/                 # ML models
│   │   ├── baseline.py
│   │   ├── xgboost_ranker.py
│   │   └── portfolio.py
│   ├── evaluation/             # Backtesting & metrics
│   │   ├── metrics.py
│   │   └── backtest.py
│   └── utils/
│       ├── config.py
│       ├── universe.py
│       └── data_utils.py
├── data/                       # Data storage
├── models/                     # Saved models
├── results/                    # Output results
├── notebooks/                  # Analysis notebooks
├── config.yaml
├── requirements.txt
└── main.py
```

## 🔬 Methodology

### Sentiment Features

- **General Sentiment**: FinBERT scores (positive, neutral, negative)
- **Domain-Specific**:
  - Supply risk (strikes, outages, shutdowns)
  - Demand signals (China PMI, EV demand)
  - Cost pressure (energy, labor, inflation)
  - Regulatory risk (ESG, environmental)

### Models

1. **Baseline**: Logistic regression (outperform vs ETF)
2. **Advanced**: XGBoost/LightGBM ranking
3. **Portfolio**: Long top 20%, short bottom 20%

### Evaluation Metrics

- Information Coefficient (IC)
- Sharpe Ratio
- Hit Rate
- Sector-Neutral Returns
- Turnover-Adjusted Performance

## 📈 Example Output

```
Weekly Rankings (2025-01-24):
1. FCX  - Sentiment: 0.72 | Supply Risk: Low  | Prediction: +3.2%
2. BHP  - Sentiment: 0.65 | Supply Risk: Low  | Prediction: +2.1%
3. NEM  - Sentiment: 0.58 | Supply Risk: Med  | Prediction: +1.5%
...

Portfolio Performance (30 days):
Alpha: 2.3%
Sharpe: 1.8
Max Drawdown: -4.2%
```

## ⚠️ Important Notes

- Commodities move on macro factors, not just company news
- Gold ≠ copper ≠ lithium (segment separately)
- Sentiment flips fast during geopolitical shocks
- Small caps overreact (noisy but profitable)

## 📝 License

MIT
