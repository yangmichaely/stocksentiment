"""
Main pipeline for mining stock sentiment analysis
"""
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Data collection
from src.data.reddit_collector import RedditCollector
from src.data.news_collector import NewsCollector
from src.data.earnings_collector import EarningsCollector
from src.data.sec_collector import SECCollector

# Sentiment analysis
from src.sentiment.aggregator import SentimentAggregator, analyze_all_data

# Features
from src.features.sentiment_features import SentimentFeatureEngineer
from src.features.technical_features import TechnicalFeatureEngineer

# Models
from src.models.baseline import BaselineModel
from src.models.xgboost_ranker import XGBoostRanker
from src.models.portfolio import Portfolio

# Evaluation
from src.evaluation.metrics import evaluate_predictions, print_evaluation_report

# Utils
from src.utils.config import config
from src.utils.data_utils import save_dataframe, load_dataframe


def collect_data(days_back=30):
    """
    Collect data from all sources
    
    Args:
        days_back: Number of days to look back
    """
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)
    
    all_data = []
    """
    # Reddit
    print("\n[1/4] Collecting Reddit data...")
    try:
        reddit_collector = RedditCollector()
        reddit_df = reddit_collector.collect_all(days_back=days_back, save=True)
        if not reddit_df.empty:
            all_data.append(reddit_df)
            print(f"✓ Collected {len(reddit_df)} Reddit posts/comments")
    except Exception as e:
        print(f"✗ Reddit collection failed: {e}")
    """
    # News
    print("\n[2/4] Collecting news data...")
    try:
        news_collector = NewsCollector()
        news_df = news_collector.collect_all(days_back=days_back, save=True)
        if not news_df.empty:
            all_data.append(news_df)
            print(f"✓ Collected {len(news_df)} news articles")
    except Exception as e:
        print(f"✗ News collection failed: {e}")
    
    # Earnings
    print("\n[3/4] Collecting earnings data...")
    try:
        earnings_collector = EarningsCollector()
        earnings_df = earnings_collector.collect_all(save=True)
        if not earnings_df.empty:
            all_data.append(earnings_df)
            print(f"✓ Collected {len(earnings_df)} earnings reports")
    except Exception as e:
        print(f"✗ Earnings collection failed: {e}")
    
    # SEC filings
    print("\n[4/4] Collecting SEC filings...")
    try:
        sec_collector = SECCollector()
        sec_df = sec_collector.collect_all(days_back=days_back, save=True)
        if not sec_df.empty:
            all_data.append(sec_df)
            print(f"✓ Collected {len(sec_df)} SEC filings")
    except Exception as e:
        print(f"✗ SEC collection failed: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        save_path = config.data_dir / 'raw' / f'combined_{datetime.now().strftime("%Y%m%d")}.parquet'
        save_dataframe(combined_df, save_path)
        print(f"\n✓ Total data collected: {len(combined_df)} items")
        print(f"✓ Saved to: {save_path}")
        return combined_df
    else:
        print("\n✗ No data collected")
        return pd.DataFrame()


def analyze_sentiment(df=None):
    """
    Perform sentiment analysis
    
    Args:
        df: Input dataframe (if None, loads from disk)
    """
    print("\n" + "="*60)
    print("STEP 2: SENTIMENT ANALYSIS")
    print("="*60)
    
    if df is None:
        # Load most recent combined data
        data_files = list((config.data_dir / 'raw').glob('combined_*.parquet'))
        if not data_files:
            print("✗ No data files found. Run data collection first.")
            return None
        
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading data from: {latest_file}")
        df = load_dataframe(latest_file)
    
    # Analyze sentiment
    analyzed_df = analyze_all_data(df, save=True)
    
    print(f"\n✓ Sentiment analysis complete: {len(analyzed_df)} items analyzed")
    
    return analyzed_df


def create_features(sentiment_df=None):
    """
    Create features for modeling
    
    Args:
        sentiment_df: Sentiment dataframe (if None, loads from disk)
    """
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    # Load sentiment data if not provided
    if sentiment_df is None:
        sentiment_files = list((config.data_dir / 'processed').glob('sentiment_aggregated_*.parquet'))
        if not sentiment_files:
            print("✗ No sentiment data found. Run sentiment analysis first.")
            return None
        
        latest_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading sentiment data from: {latest_file}")
        sentiment_df = load_dataframe(latest_file)
    
    # Create sentiment features
    sentiment_engineer = SentimentFeatureEngineer()
    features_df = sentiment_engineer.create_all_features(sentiment_df)
    
    # Get technical features (price data)
    print("\nCollecting price data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    technical_engineer = TechnicalFeatureEngineer()
    technical_df = technical_engineer.create_all_features(start_date, end_date)
    
    # Merge sentiment and technical features
    if not technical_df.empty:
        print("\nMerging sentiment and technical features...")
        # Convert date columns to datetime and normalize timezones
        features_df['date'] = pd.to_datetime(features_df['date']).dt.tz_localize(None)
        technical_df['date'] = pd.to_datetime(technical_df['date']).dt.tz_localize(None)
        
        # Merge on ticker and date
        combined_df = pd.merge(
            features_df,
            technical_df,
            on=['ticker', 'date'],
            how='inner'
        )
        
        print(f"✓ Feature engineering complete: {len(combined_df)} samples with {len(combined_df.columns)} features")
    else:
        combined_df = features_df
        print(f"✓ Feature engineering complete: {len(combined_df)} samples (sentiment only)")
    
    # Save features
    save_path = config.data_dir / 'processed' / f'features_{datetime.now().strftime("%Y%m%d")}.parquet'
    save_dataframe(combined_df, save_path)
    print(f"✓ Saved to: {save_path}")
    
    return combined_df


def train_and_predict(features_df=None):
    """
    Train models and make predictions
    
    Args:
        features_df: Features dataframe (if None, loads from disk)
    """
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING & PREDICTION")
    print("="*60)
    
    # Load features if not provided
    if features_df is None:
        feature_files = list((config.data_dir / 'processed').glob('features_*.parquet'))
        if not feature_files:
            print("✗ No feature data found. Run feature engineering first.")
            return None
        
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading features from: {latest_file}")
        features_df = load_dataframe(latest_file)
    
    # Train baseline model
    print("\n[1/2] Training Baseline Model...")
    baseline = BaselineModel(model_type='linear')  # Use linear regression instead of logistic
    baseline_metrics = baseline.train(features_df, target_col='forward_return_5d')  # Use continuous target
    
    if baseline_metrics:
        baseline_path = config.models_dir / 'baseline_model.joblib'
        baseline.save(baseline_path)
    
    # Train XGBoost model
    print("\n[2/2] Training XGBoost Model...")
    xgb_model = XGBoostRanker()
    xgb_metrics = xgb_model.train(features_df, target_col='forward_return_5d')
    
    if xgb_metrics:
        xgb_path = config.models_dir / 'xgboost_model.joblib'
        xgb_model.save(xgb_path)
        
        # Make predictions and rank stocks
        print("\nMaking predictions...")
        predictions_df = xgb_model.rank_stocks(features_df)
        
        # Add forward_return column for backtesting if not present
        if 'forward_return' not in predictions_df.columns and 'forward_return_5d' in predictions_df.columns:
            predictions_df['forward_return'] = predictions_df['forward_return_5d']
        
        # Save predictions
        pred_path = config.results_dir / f'predictions_{datetime.now().strftime("%Y%m%d")}.parquet'
        save_dataframe(predictions_df, pred_path)
        print(f"✓ Predictions saved to: {pred_path}")
        
        # Evaluate predictions
        print("\nEvaluating predictions...")
        eval_metrics = evaluate_predictions(predictions_df)
        print_evaluation_report(eval_metrics)
        
        # Backtest portfolio
        print("\nBacktesting portfolio...")
        portfolio = Portfolio(long_percentile=80, short_percentile=20)
        backtest_df = portfolio.backtest(predictions_df)
        
        if not backtest_df.empty:
            portfolio.print_performance_summary(backtest_df)
            
            # Save backtest results
            backtest_path = config.results_dir / f'backtest_{datetime.now().strftime("%Y%m%d")}.parquet'
            save_dataframe(backtest_df, backtest_path)
            print(f"✓ Backtest results saved to: {backtest_path}")
        
        # Print top predictions
        print("\n" + "="*60)
        print("TOP 10 STOCK PREDICTIONS")
        print("="*60)
        
        latest_date = predictions_df['date'].max()
        latest_predictions = predictions_df[predictions_df['date'] == latest_date].head(10)
        
        for i, row in enumerate(latest_predictions.itertuples(), 1):
            sentiment = getattr(row, 'combined_sentiment', getattr(row, 'sentiment_score_mean', 0))
            pred_return = getattr(row, 'predicted_return', 0)
            
            print(f"{i:2d}. {row.ticker:6s} | Sentiment: {sentiment:6.3f} | Predicted Return: {pred_return:+.2%}")
        
        # Print full portfolio positions
        print("\n" + "="*60)
        print("CURRENT PORTFOLIO POSITIONS")
        print("="*60)
        
        current_positions = portfolio.get_current_positions()
        
        # Get portfolio details (may be empty if backtest failed)
        last_portfolio = portfolio.positions[-1] if portfolio.positions else {}
        long_positions_detail = last_portfolio.get('long_positions', {})
        short_positions_detail = last_portfolio.get('short_positions', {})
        
        print(f"\nLONG POSITIONS ({current_positions['num_long']} stocks):")
        print("-" * 60)
        if current_positions['long'] and long_positions_detail:
            # Get position details from last portfolio construction
            
            # Create list of (ticker, weight) sorted by weight descending
            long_sorted = sorted(long_positions_detail.items(), 
                               key=lambda x: x[1]['weight'], reverse=True)
            
            for ticker, details in long_sorted:
                # Get prediction details for this ticker
                ticker_pred = predictions_df[(predictions_df['ticker'] == ticker) & 
                                            (predictions_df['date'] == latest_date)]
                if not ticker_pred.empty:
                    sentiment = ticker_pred.iloc[0].get('combined_sentiment', 
                                                       ticker_pred.iloc[0].get('sentiment_score_mean', 0))
                    pred_return = ticker_pred.iloc[0].get('predicted_return', 0)
                    print(f"  {ticker:6s} | {details['percent']:5.2f}% | ${details['dollars']:>12,.0f} | "
                          f"Sentiment: {sentiment:6.3f} | Predicted: {pred_return:+.2%}")
                else:
                    print(f"  {ticker:6s} | {details['percent']:5.2f}% | ${details['dollars']:>12,.0f}")
        else:
            print("  (No long positions)")
        
        print(f"\nSHORT POSITIONS ({current_positions['num_short']} stocks):")
        print("-" * 60)
        if current_positions['short'] and short_positions_detail:
            
            # Create list of (ticker, weight) sorted by weight descending
            short_sorted = sorted(short_positions_detail.items(), 
                                key=lambda x: x[1]['weight'], reverse=True)
            
            for ticker, details in short_sorted:
                # Get prediction details for this ticker
                ticker_pred = predictions_df[(predictions_df['ticker'] == ticker) & 
                                            (predictions_df['date'] == latest_date)]
                if not ticker_pred.empty:
                    sentiment = ticker_pred.iloc[0].get('combined_sentiment', 
                                                       ticker_pred.iloc[0].get('sentiment_score_mean', 0))
                    pred_return = ticker_pred.iloc[0].get('predicted_return', 0)
                    print(f"  {ticker:6s} | {details['percent']:5.2f}% | ${details['dollars']:>12,.0f} | "
                          f"Sentiment: {sentiment:6.3f} | Predicted: {pred_return:+.2%}")
                else:
                    print(f"  {ticker:6s} | {details['percent']:5.2f}% | ${details['dollars']:>12,.0f}")
        else:
            print("  (No short positions)")
        
        # Save portfolio positions to file (only if portfolio was created)
        if last_portfolio:
            portfolio_positions = {
                'date': str(latest_date),
                'total_value': last_portfolio.get('total_value', 10_000_000),
            'long_positions': [
                {
                    'ticker': ticker,
                    'percent': details['percent'],
                    'dollars': details['dollars'],
                    'weight': details['weight']
                }
                for ticker, details in sorted(long_positions_detail.items(), 
                                             key=lambda x: x[1]['weight'], reverse=True)
            ] if long_positions_detail else [],
            'short_positions': [
                {
                    'ticker': ticker,
                    'percent': details['percent'],
                    'dollars': details['dollars'],
                    'weight': details['weight']
                }
                for ticker, details in sorted(short_positions_detail.items(), 
                                             key=lambda x: x[1]['weight'], reverse=True)
            ] if short_positions_detail else [],
            'num_long': current_positions['num_long'],
            'num_short': current_positions['num_short']
        }
        
        import json
        portfolio_path = config.results_dir / f'portfolio_positions_{datetime.now().strftime("%Y%m%d")}.json'
        with open(portfolio_path, 'w') as f:
            json.dump(portfolio_positions, f, indent=2)
        print(f"\n✓ Portfolio positions saved to: {portfolio_path}")
        
        # Also save as CSV for easy viewing
        import pandas as pd
        long_df = pd.DataFrame([
            {
                'ticker': ticker,
                'side': 'LONG',
                'percent': details['percent'],
                'dollars': details['dollars']
            }
            for ticker, details in sorted(long_positions_detail.items(), 
                                         key=lambda x: x[1]['weight'], reverse=True)
        ]) if long_positions_detail else pd.DataFrame()
        
        short_df = pd.DataFrame([
            {
                'ticker': ticker,
                'side': 'SHORT',
                'percent': details['percent'],
                'dollars': details['dollars']
            }
            for ticker, details in sorted(short_positions_detail.items(), 
                                         key=lambda x: x[1]['weight'], reverse=True)
        ]) if short_positions_detail else pd.DataFrame()
        
        if not long_df.empty or not short_df.empty:
            portfolio_csv_df = pd.concat([long_df, short_df], ignore_index=True)
            csv_path = config.results_dir / f'portfolio_positions_{datetime.now().strftime("%Y%m%d")}.csv'
            portfolio_csv_df.to_csv(csv_path, index=False)
            print(f"✓ Portfolio CSV saved to: {csv_path}")
        
        return predictions_df
    
    return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Mining Stock Sentiment Analysis')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['collect', 'analyze', 'features', 'predict', 'full'],
                       help='Execution mode')
    parser.add_argument('--days', type=int, default=30,
                       help='Days of historical data to collect')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MINING & MATERIALS SENTIMENT STOCK PICKER")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        if args.mode == 'collect':
            collect_data(days_back=args.days)
        
        elif args.mode == 'analyze':
            analyze_sentiment()
        
        elif args.mode == 'features':
            create_features()
        
        elif args.mode == 'predict':
            train_and_predict()
        
        elif args.mode == 'full':
            # Run full pipeline
            df = collect_data(days_back=args.days)
            if not df.empty:
                sentiment_df = analyze_sentiment(df)
                if sentiment_df is not None:
                    features_df = create_features(sentiment_df)
                    if features_df is not None:
                        train_and_predict(features_df)
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
