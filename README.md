# Mining & Materials Sentiment Stock Picker

A sentiment analysis-based stock picker for the mining and materials sector using multiple data sources (Reddit, news, etc). For FIN367 with Kocher.

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