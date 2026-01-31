# Quick Start Guide

This guide will help you get started with the Mining & Materials Sentiment Stock Picker.

## Prerequisites

1. Python 3.8 or higher
2. Virtual environment activated
3. API keys configured

## Setup Steps

### 1. Install Dependencies

```powershell
# Make sure you're in the project directory
cd C:\Users\yangm\StudioProjects\stocksentiment

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```powershell
cp .env.example .env
```

Edit `.env` with your favorite text editor and add:

- **Reddit API**: Get from https://www.reddit.com/prefs/apps
- **NewsAPI**: Get from https://newsapi.org/register
- **Alpha Vantage**: Get from https://www.alphavantage.co/support/#api-key

### 3. Test the Setup

Run the sentiment demo to verify everything works:

```powershell
python examples/sentiment_demo.py
```

You should see sentiment analysis results for sample texts.

## Running the Full Pipeline

### Option 1: Full Pipeline (Recommended for First Run)

```powershell
python main.py --mode full --days 30
```

This will:
1. Collect data from Reddit, news, earnings, SEC filings (last 30 days)
2. Perform sentiment analysis using FinBERT + domain lexicons
3. Engineer features (sentiment + technical)
4. Train models (Baseline + XGBoost)
5. Generate predictions and backtest portfolio

**Note**: Full pipeline can take 30-60 minutes depending on API rate limits.

### Option 2: Step-by-Step Execution

```powershell
# Step 1: Collect data
python main.py --mode collect --days 30

# Step 2: Analyze sentiment
python main.py --mode analyze

# Step 3: Create features
python main.py --mode features

# Step 4: Train models and predict
python main.py --mode predict
```

## Understanding the Output

### Data Files

All data is saved in the `data/` directory:

- `data/raw/` - Raw collected data
- `data/processed/` - Sentiment analysis results
- `data/processed/features_*.parquet` - Feature datasets

### Model Files

Trained models are saved in `models/`:

- `baseline_model.joblib` - Logistic regression baseline
- `xgboost_model.joblib` - XGBoost ranking model

### Results

Predictions and backtests are in `results/`:

- `predictions_*.parquet` - Stock rankings and predictions
- `backtest_*.parquet` - Portfolio performance over time

## Interpreting Results

### Top Predictions

The pipeline will output a ranked list of stocks:

```
TOP 10 STOCK PREDICTIONS
==================================================
 1. FCX    | Sentiment:  0.652 | Predicted Return: +3.2%
 2. BHP    | Sentiment:  0.581 | Predicted Return: +2.1%
 3. NEM    | Sentiment:  0.423 | Predicted Return: +1.5%
...
```

### Portfolio Performance

```
PORTFOLIO PERFORMANCE SUMMARY
==================================================
Total Return:    5.2%
Avg Return:      0.8%
Volatility:      2.1%
Sharpe Ratio:    1.8
Max Drawdown:    -4.2%
Win Rate:        62%
Periods:         30
```

**Key Metrics**:
- **Sharpe Ratio > 1.5**: Good risk-adjusted returns
- **Win Rate > 55%**: Better than random
- **IC > 0.1**: Predictive signal present

## Common Issues

### Issue 1: API Rate Limits

If you hit rate limits:

```powershell
# Collect less data
python main.py --mode collect --days 7
```

### Issue 2: Not Enough Data

If you see "Not enough data for training":

- Increase `--days` parameter
- Check that API keys are valid
- Ensure data was successfully collected

### Issue 3: FinBERT Model Download

First run will download ~400MB FinBERT model. This is normal and only happens once.

## Next Steps

1. **Experiment with parameters**: Edit `config.yaml` to adjust:
   - Time windows
   - Portfolio percentiles
   - Model hyperparameters

2. **Add more data sources**: Extend collectors in `src/data/`

3. **Custom features**: Add domain-specific features in `src/features/`

4. **Backtesting**: Analyze results in `results/` directory

## Tips for Success

1. **Start small**: Use `--days 7` for quick iteration
2. **Monitor API quotas**: Free tiers have limits
3. **Check data quality**: Review raw data files before training
4. **Compare to benchmark**: Always compare returns to XLB/PICK
5. **Focus on IC**: Information Coefficient is key metric

## Getting Help

- Check logs in `logs/` directory
- Review code in `src/` for implementation details
- Refer to `README.md` for project overview

Happy trading! 📈
