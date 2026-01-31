"""
Example: Quick sentiment analysis demo
Run this after collecting some data to test the pipeline
"""
import pandas as pd
from datetime import datetime

from src.sentiment.aggregator import SentimentAggregator
from src.utils.config import config

# Sample texts about mining stocks
sample_texts = [
    {
        'ticker': 'FCX',
        'text': 'Freeport-McMoRan reports strong copper production growth, beating estimates. Expansion in Chile mine proceeding ahead of schedule.',
        'timestamp': datetime.now(),
        'source': 'demo'
    },
    {
        'ticker': 'FCX',
        'text': 'Strike at FCX Chilean operations disrupts copper output. Workers demanding higher wages amid inflation concerns.',
        'timestamp': datetime.now(),
        'source': 'demo'
    },
    {
        'ticker': 'NEM',
        'text': 'Newmont Mining raises full-year gold production guidance. Cost reduction initiatives showing results.',
        'timestamp': datetime.now(),
        'source': 'demo'
    },
    {
        'ticker': 'BHP',
        'text': 'BHP Group announces $10B investment in renewable energy transition for mining operations. ESG rating upgraded.',
        'timestamp': datetime.now(),
        'source': 'demo'
    },
    {
        'ticker': 'VALE',
        'text': 'Vale faces regulatory investigation over tailings dam safety. Environmental concerns mounting.',
        'timestamp': datetime.now(),
        'source': 'demo'
    },
    {
        'ticker': 'ALB',
        'text': 'Albemarle expects lithium demand to surge 40% on EV adoption. Expanding production capacity in Nevada.',
        'timestamp': datetime.now(),
        'source': 'demo'
    }
]

def run_demo():
    """Run sentiment analysis demo"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS DEMO")
    print("="*60)
    
    # Create dataframe
    df = pd.DataFrame(sample_texts)
    
    print(f"\nAnalyzing {len(df)} sample texts...\n")
    
    # Initialize aggregator
    aggregator = SentimentAggregator()
    
    # Analyze sentiment
    analyzed_df = aggregator.analyze_dataframe(df)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for idx, row in analyzed_df.iterrows():
        print(f"\nTicker: {row['ticker']}")
        print(f"Text: {row['text'][:100]}...")
        print(f"Sentiment Score: {row['sentiment_score']:+.3f}")
        print(f"  Positive: {row['positive']:.3f}")
        print(f"  Negative: {row['negative']:.3f}")
        print(f"  Neutral: {row['neutral']:.3f}")
        print(f"Supply Risk: {row['supply_risk_score']:+.3f}")
        print(f"Demand Score: {row['demand_score']:+.3f}")
        print(f"Combined: {row['combined_sentiment']:+.3f}")
    
    # Aggregate by ticker
    print("\n" + "="*60)
    print("TICKER SUMMARY")
    print("="*60)
    
    summary = analyzed_df.groupby('ticker').agg({
        'sentiment_score': 'mean',
        'combined_sentiment': 'mean',
        'supply_risk_score': 'mean',
        'demand_score': 'mean'
    }).round(3)
    
    print(summary)
    
    print("\n✓ Demo complete!\n")


if __name__ == '__main__':
    run_demo()
