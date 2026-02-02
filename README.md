# Mining & Materials Sentiment Stock Picker

A sentiment analysis-based stock picker for the mining and materials sector using multiple data sources (Reddit, news, etc). For FIN367 with Kocher.

### Installation

```bash
# Clone and setup
git clone https://github.com/yangmichaely/stocksentiment
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

### Basic Usage

```bash
# Full pipeline (initial portfolio creation)
python main.py --mode full --days 180 --portfolio 10000000   #10M portfolio

# Individual steps
python main.py --mode collect --days 180     # Collect data only
python main.py --mode analyze                # Run sentiment analysis
python main.py --mode features               # Create features
python main.py --mode predict                # Train models and generate predictions

# Weekly rebalancing (for live trading after portfolio creation)
python main.py --mode live --days 180 --portfolio 10000000

# Custom portfolio size examples
python main.py --mode full --days 90 --portfolio 50000      # $50K portfolio with 90 day lookback history
python main.py --mode full --days 180 --portfolio 1000000   # $1M portfolio with 180 day lookback history
```

**Parameters:**
- `--mode`: Execution mode (collect, analyze, features, predict, full, live)
- `--days`: Historical data window in days (default: 180)
- `--portfolio`: Portfolio size in dollars (default: 10000000)