"""
Portfolio construction and backtesting
"""
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.config import config


class Portfolio:
    """Long-only portfolio based on sentiment rankings with absolute thresholds"""
    
    def __init__(self, sentiment_threshold=0.25, max_positions=15, long_only=True, portfolio_value=10_000_000):
        """
        Initialize portfolio
        
        Args:
            sentiment_threshold: Minimum absolute sentiment score to consider (default 0.25)
            max_positions: Maximum number of positions to hold (default 15)
            long_only: If True, only long positions; if False, market-neutral long-short (default True)
            portfolio_value: Total portfolio size in dollars (default $10M)
        """
        self.sentiment_threshold = sentiment_threshold
        self.max_positions = max_positions
        self.long_only = long_only
        self.portfolio_value = portfolio_value
        
        # Legacy parameters for backwards compatibility
        self.long_percentile = 80
        self.short_percentile = 20
        
        self.positions = []
        self.current_holdings = {'long': [], 'short': []}  # Track current positions
        self.state_file = config.results_dir / 'portfolio_state.json'
    
    def save_state(self):
        """Save current portfolio state for weekly rebalancing"""
        if not self.positions:
            return
        
        state = {
            'date': str(self.positions[-1].get('date', datetime.now().date())),
            'current_holdings': self.current_holdings,
            'last_portfolio': self.positions[-1],
            'sentiment_threshold': self.sentiment_threshold,
            'max_positions': self.max_positions,
            'long_only': self.long_only
        }
        
        import json
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✓ Portfolio state saved: {self.state_file}")
    
    def load_state(self) -> bool:
        """Load previous portfolio state. Returns True if state loaded."""
        if not self.state_file.exists():
            print("No previous portfolio state found (first run)")
            return False
        
        import json
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.current_holdings = state['current_holdings']
            self.positions = [state['last_portfolio']]
            
            print(f"✓ Loaded portfolio state from {state['date']}")
            print(f"  Current: {len(self.current_holdings['long'])} long, {len(self.current_holdings['short'])} short")
            return True
        except Exception as e:
            print(f"Warning: Could not load state: {e}")
            return False
    
    def calculate_position_weights(self, ranked_df: pd.DataFrame, tickers: list, 
                                   side: str, max_weight=None, min_weight=0.01) -> dict:
        """
        Calculate position weights based on signal strength with exponential decay
        
        Args:
            ranked_df: DataFrame with predictions
            tickers: List of tickers to weight
            side: 'long' or 'short'
            max_weight: Maximum position weight (auto-calculated if None)
            min_weight: Minimum position weight (default 1%)
        
        Returns:
            Dict mapping ticker to weight
        """
        if not tickers:
            return {}
        
        # Get predicted returns for these tickers
        ticker_df = ranked_df[ranked_df['ticker'].isin(tickers)].copy()
        
        # Use absolute value of predicted return as signal strength
        ticker_df['signal_strength'] = ticker_df['predicted_return'].abs()
        
        # Rank-based weighting: higher rank = higher weight
        # Use exponential decay to concentrate in top positions
        ticker_df = ticker_df.sort_values('signal_strength', ascending=False)
        n = len(ticker_df)
        
        # Auto-calculate max_weight based on number of positions if not specified
        # With exponential decay, top position typically gets ~20% in small portfolios
        # For larger portfolios (20+ stocks), cap at 10%
        if max_weight is None:
            if n <= 5:
                max_weight = 0.30  # Up to 30% for concentrated portfolios
            elif n <= 10:
                max_weight = 0.20  # Up to 20% for medium portfolios
            elif n <= 20:
                max_weight = 0.15  # Up to 15% for larger portfolios
            else:
                max_weight = 0.10  # Cap at 10% for very diversified portfolios
        
        # Exponential weights: top position gets most, decays exponentially
        # decay_factor controls concentration (0.5 = moderate, 0.3 = aggressive)
        decay_factor = 0.4
        ranks = np.arange(1, n + 1)
        raw_weights = np.exp(-decay_factor * (ranks - 1) / n)
        
        # Normalize to sum to 1
        raw_weights = raw_weights / raw_weights.sum()
        
        # Apply min/max constraints
        weights = np.clip(raw_weights, min_weight, max_weight)
        
        # Renormalize after clipping
        weights = weights / weights.sum()
        
        # Create weight dictionary
        weight_dict = dict(zip(ticker_df['ticker'].tolist(), weights))
        
        return weight_dict
    
    def construct_portfolio(self, ranked_df: pd.DataFrame, date=None, incremental=True) -> dict:
        """
        Construct long-only portfolio with absolute sentiment threshold and top-K selection
        
        Strategy:
        1. Filter for stocks with sentiment_score > threshold (high conviction only)
        2. Rank by predicted return and take top K positions
        3. Weight by signal strength with exponential decay
        4. Remaining capital held as cash (earning risk-free rate)
        
        Args:
            ranked_df: DataFrame with predictions, sentiment_score, and ranks
            date: Optional date for this portfolio snapshot
            incremental: If True, only adjust positions that change significantly
        
        Returns:
            Dict with long positions, weights, dollar amounts, and cash position
        """
        portfolio_value = self.portfolio_value
        # Use predicted_return as the signal (XGBoost model output)
        # Convert threshold from sentiment scale to predicted_return scale
        # Since predicted_return is typically in range -0.1 to 0.1, use a lower threshold
        signal_col = 'predicted_return'
        
        # Auto-detect appropriate threshold based on data distribution
        if signal_col in ranked_df.columns:
            # Use top 30% as candidates (more relaxed than fixed 0.25 sentiment threshold)
            threshold_percentile = 70  # Top 30%
            threshold = ranked_df[signal_col].quantile(threshold_percentile / 100)
            
            print(f"\nUsing {signal_col} as signal (threshold: {threshold:.4f} = top {100-threshold_percentile}%)")
        else:
            threshold = 0
            print(f"\nWarning: {signal_col} not found")
        
        # Step 1: Filter for high-conviction signals (above threshold)
        if self.long_only:
            # Long-only: only positive signals above threshold
            candidates = ranked_df[
                ranked_df[signal_col] > threshold
            ].copy()
            
            print(f"Filtering: {len(candidates)} stocks with {signal_col} > {threshold:.4f}")
        else:
            # Market-neutral: both long and short
            long_candidates = ranked_df[
                ranked_df[signal_col] > threshold
            ].copy()
            short_candidates = ranked_df[
                ranked_df[signal_col] < -threshold
            ].copy()
            
            print(f"Filtering: {len(long_candidates)} long candidates, {len(short_candidates)} short candidates")
        
        # Step 2: Rank by signal strength and take top K
        if self.long_only:
            # Sort by predicted return (descending) and take top K
            candidates = candidates.sort_values('predicted_return', ascending=False)
            long_stocks = candidates.head(self.max_positions)
            short_stocks = pd.DataFrame()  # Empty for long-only
            
            print(f"Selected top {len(long_stocks)} positions (max: {self.max_positions})")
        else:
            # Market-neutral: take top K longs and top K shorts
            long_candidates = long_candidates.sort_values('predicted_return', ascending=False)
            short_candidates = short_candidates.sort_values('predicted_return', ascending=True)
            
            long_stocks = long_candidates.head(self.max_positions)
            short_stocks = short_candidates.head(self.max_positions)
            
            print(f"Selected {len(long_stocks)} long, {len(short_stocks)} short (max each: {self.max_positions})")
        
        new_long = long_stocks['ticker'].tolist()
        new_short = short_stocks['ticker'].tolist() if not self.long_only else []
        
        # Step 3: Calculate position weights with exponential decay
        long_weights = self.calculate_position_weights(ranked_df, new_long, 'long')
        short_weights = self.calculate_position_weights(ranked_df, new_short, 'short') if not self.long_only else {}
        
        # Step 4: Calculate dollar amounts
        if self.long_only:
            # Long-only: allocate up to 100% to long positions, rest is cash
            # If we have fewer positions than max, we'll hold more cash
            long_capital = portfolio_value  # Full portfolio for long positions
            short_capital = 0
            
            long_positions = {ticker: {'weight': weight, 
                                       'dollars': weight * long_capital,
                                       'percent': weight * 100}
                             for ticker, weight in long_weights.items()}
            short_positions = {}
            
            # Calculate cash position
            invested_capital = sum(pos['dollars'] for pos in long_positions.values())
            cash_position = portfolio_value - invested_capital
            cash_percent = (cash_position / portfolio_value) * 100
            
            print(f"Capital allocation: {cash_percent:.1f}% cash, {100-cash_percent:.1f}% invested")
        else:
            # Market-neutral: 50% long, 50% short
            long_capital = portfolio_value * 0.5
            short_capital = portfolio_value * 0.5
            
            long_positions = {ticker: {'weight': weight, 
                                       'dollars': weight * long_capital,
                                       'percent': weight * 100}
                             for ticker, weight in long_weights.items()}
            
            short_positions = {ticker: {'weight': weight,
                                        'dollars': weight * short_capital,
                                        'percent': weight * 100}
                              for ticker, weight in short_weights.items()}
            
            cash_position = 0  # Fully invested in market-neutral
            cash_percent = 0
        
        # Calculate incremental changes
        if incremental and self.current_holdings['long']:
            # Determine what to buy/sell
            long_to_buy = set(new_long) - set(self.current_holdings['long'])
            long_to_sell = set(self.current_holdings['long']) - set(new_long)
            short_to_buy = set(new_short) - set(self.current_holdings['short'])
            short_to_sell = set(self.current_holdings['short']) - set(new_short)
        else:
            # First portfolio - buy everything
            long_to_buy = set(new_long)
            long_to_sell = set()
            short_to_buy = set(new_short)
            short_to_sell = set()
        
        # Update current holdings
        self.current_holdings['long'] = new_long
        self.current_holdings['short'] = new_short
        
        # Calculate turnover (number of positions changed)
        turnover = len(long_to_buy) + len(long_to_sell) + len(short_to_buy) + len(short_to_sell)
        
        portfolio = {
            'date': date,
            'long': new_long,
            'short': new_short,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'num_long': len(new_long),
            'num_short': len(new_short),
            'long_to_buy': list(long_to_buy),
            'long_to_sell': list(long_to_sell),
            'short_to_buy': list(short_to_buy),
            'short_to_sell': list(short_to_sell),
            'turnover': turnover,
            'total_value': portfolio_value,
            'cash_position': cash_position,
            'cash_percent': cash_percent,
            'long_only': self.long_only
        }
        
        self.positions.append(portfolio)
        
        return portfolio
    
    def backtest_walk_forward(self, features_df: pd.DataFrame, model, returns_col='forward_return_5d', 
                             target_col='forward_return_5d', incremental=True, min_train_periods=30) -> pd.DataFrame:
        """
        Walk-forward backtest: train model on expanding window, predict out-of-sample only
        
        This eliminates look-ahead bias by ensuring predictions are always out-of-sample:
        - For each period, train on ALL data BEFORE that period
        - Predict ONLY on that period (never seen by model)
        - Construct portfolio and measure performance
        
        Portfolio size: ${self.portfolio_value:,.0f}
        
        Args:
            features_df: DataFrame with features and forward returns (NOT predictions)
            model: XGBoostRanker instance (will be retrained each period)
            returns_col: Column with actual returns to evaluate
            target_col: Column to train model on
            incremental: If True, track turnover from incremental changes
            min_train_periods: Minimum training periods before starting backtest
        
        Returns:
            DataFrame with backtest results
        """
        print("\n" + "="*80)
        print("WALK-FORWARD BACKTEST (Out-of-Sample Only)")
        print("="*80)
        print("Each prediction is made on data the model has NEVER seen during training.")
        print("This prevents look-ahead bias and overfitting.\n")
        
        if 'date' not in features_df.columns:
            print("Warning: No date column found")
            return pd.DataFrame()
        
        # Get unique dates sorted
        dates = sorted(features_df['date'].unique())
        
        if len(dates) < min_train_periods + 5:
            print(f"Insufficient data: need at least {min_train_periods + 5} periods, have {len(dates)}")
            return pd.DataFrame()
        
        # Reset portfolio state
        self.current_holdings = {'long': [], 'short': []}
        self.positions = []
        
        results = []
        all_predictions = []
        
        print(f"Total periods: {len(dates)}")
        print(f"Training window: expanding (starts with {min_train_periods} periods)")
        print(f"Out-of-sample periods: {len(dates) - min_train_periods}\n")
        
        # Walk forward through time
        for i, current_date in enumerate(dates[min_train_periods:], start=min_train_periods):
            # Train on all data BEFORE current_date
            train_metrics = model.train_on_period(features_df, current_date, target_col, min_train_periods)
            
            if train_metrics is None:
                print(f"  Period {i+1}/{len(dates)}: Skipping {current_date} (insufficient training data)")
                continue
            
            # Get current period data (out-of-sample)
            current_period = features_df[features_df['date'] == current_date].copy()
            
            if len(current_period) == 0:
                continue
            
            # Make predictions (out-of-sample)
            predictions = model.predict(current_period)
            current_period['predicted_return'] = predictions
            
            # Construct portfolio based on predictions
            portfolio = self.construct_portfolio(current_period, current_date, incremental=incremental)
            
            # CRITICAL FIX: Evaluate on REALIZED returns (forward_return_5d is the actual outcome)
            # We use the current period's forward_return_5d which represents what actually happened
            # in the next 5 days AFTER this prediction was made.
            # This is correct because forward_return_5d was calculated as shift(-5), meaning
            # it contains the future return that we want to predict.
            
            # Calculate actual returns from the forward return column
            long_stocks = current_period[current_period['ticker'].isin(portfolio['long'])]
            short_stocks = current_period[current_period['ticker'].isin(portfolio['short'])]
            
            # Verify returns_col exists and has valid data
            if returns_col not in current_period.columns:
                print(f"  Warning: {returns_col} not found in period {current_date}")
                continue
            
            long_return = long_stocks[returns_col].mean() if len(long_stocks) > 0 else 0
            short_return = short_stocks[returns_col].mean() if len(short_stocks) > 0 else 0
            portfolio_return = long_return - short_return
            
            # Store predictions for later IC calculation
            all_predictions.append(current_period[['ticker', 'date', 'predicted_return', returns_col]])
            
            results.append({
                'date': current_date,
                'long_return': long_return,
                'short_return': short_return,
                'portfolio_return': portfolio_return,
                'num_long': portfolio['num_long'],
                'num_short': portfolio['num_short'],
                'turnover': portfolio.get('turnover', 0),
                'trades_executed': portfolio.get('turnover', 0) > 0,
                'train_samples': train_metrics['train_samples'],
                'train_ic': train_metrics['train_ic']
            })
            
            # Progress update every 10 periods
            if (i - min_train_periods + 1) % 10 == 0:
                print(f"  Completed {i - min_train_periods + 1}/{len(dates) - min_train_periods} periods...")
        
        if not results:
            print("No valid backtest periods")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        print(f"\n" + "="*80)
        print("BACKTEST RESULTS (Out-of-Sample Only)")
        print("="*80)
        print(f"Periods: {len(results_df)}")
        print(f"Avg Long Return: {results_df['long_return'].mean():.4%}")
        print(f"Avg Short Return: {results_df['short_return'].mean():.4%}")
        print(f"Avg Portfolio Return: {results_df['portfolio_return'].mean():.4%}")
        print(f"Avg Turnover: {results_df['turnover'].mean():.1f} positions/period")
        print(f"Total Trades: {int(results_df['turnover'].sum())}")
        print(f"Avg Train IC: {results_df['train_ic'].mean():.4f}")
        print(f"Sharpe Ratio: {self.calculate_sharpe(results_df['portfolio_return']):.2f}")
        
        # Print full performance summary with IC and hit rate
        self.print_performance_summary(results_df, predictions_df)
        
        return results_df
    
    def backtest(self, predictions_df: pd.DataFrame, returns_col='forward_return', incremental=True) -> pd.DataFrame:
        """
        DEPRECATED: In-sample backtest (may contain look-ahead bias)
        
        Use backtest_walk_forward() instead for proper out-of-sample testing.
        
        Args:
            predictions_df: DataFrame with predictions and actual returns
            returns_col: Column with actual returns
            incremental: If True, track turnover from incremental changes
        
        Returns:
            DataFrame with backtest results
        """
        print("\n⚠️  WARNING: Using in-sample backtest. Results may be overly optimistic.")
        print("   For accurate performance, use backtest_walk_forward() instead.\n")
        
        if 'date' not in predictions_df.columns:
            print("Warning: No date column found")
            return pd.DataFrame()
        
        # Ensure we have required columns
        required_cols = ['ticker', 'date', 'predicted_return', returns_col]
        if not all(col in predictions_df.columns for col in required_cols):
            print(f"Missing required columns: {required_cols}")
            return pd.DataFrame()
        
        # Remove rows with NaN returns
        predictions_df = predictions_df[predictions_df[returns_col].notna()].copy()
        
        # Reset portfolio state
        self.current_holdings = {'long': [], 'short': []}
        self.positions = []
        
        results = []
        
        # Group by date
        for date, group in predictions_df.groupby('date'):
            # Construct portfolio for this date (incremental)
            portfolio = self.construct_portfolio(group, date, incremental=incremental)
            
            # Calculate returns
            long_stocks = group[group['ticker'].isin(portfolio['long'])]
            short_stocks = group[group['ticker'].isin(portfolio['short'])]
            
            long_return = long_stocks[returns_col].mean() if len(long_stocks) > 0 else 0
            short_return = short_stocks[returns_col].mean() if len(short_stocks) > 0 else 0
            
            # Long-short return
            portfolio_return = long_return - short_return
            
            results.append({
                'date': date,
                'long_return': long_return,
                'short_return': short_return,
                'portfolio_return': portfolio_return,
                'num_long': portfolio['num_long'],
                'num_short': portfolio['num_short'],
                'turnover': portfolio.get('turnover', 0),
                'trades_executed': portfolio.get('turnover', 0) > 0
            })
        
        results_df = pd.DataFrame(results)
        
        print(f"\nBacktest Results:")
        print(f"Periods: {len(results_df)}")
        print(f"Avg Long Return: {results_df['long_return'].mean():.4%}")
        print(f"Avg Short Return: {results_df['short_return'].mean():.4%}")
        print(f"Avg L/S Return: {results_df['portfolio_return'].mean():.4%}")
        print(f"Avg Turnover: {results_df['turnover'].mean():.1f} positions/period")
        print(f"Total Trades: {int(results_df['turnover'].sum())}")
        print(f"Sharpe Ratio: {self.calculate_sharpe(results_df['portfolio_return']):.2f}")
        
        # Print full performance summary with IC and hit rate
        self.print_performance_summary(results_df, predictions_df)
        
        return results_df
    
    def calculate_sharpe(self, returns: pd.Series, risk_free_rate=0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # Annualized Sharpe (assuming weekly returns)
        excess_return = returns.mean() - (risk_free_rate / 52)
        return (excess_return / returns.std()) * np.sqrt(52)
    
    def get_performance_metrics(self, backtest_df: pd.DataFrame, predictions_df: pd.DataFrame = None) -> dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            backtest_df: Backtest results dataframe
            predictions_df: Original predictions dataframe with predicted_return and forward_return
        
        Returns:
            Dict with performance metrics
        """
        if backtest_df.empty or 'portfolio_return' not in backtest_df.columns:
            return {}
        
        returns = backtest_df['portfolio_return']
        
        # Cumulative return
        cumulative_return = (1 + returns).prod() - 1
        
        # Sharpe ratio
        sharpe = self.calculate_sharpe(returns)
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Information Coefficient (correlation between predicted and actual returns)
        ic = np.nan
        if predictions_df is not None and 'predicted_return' in predictions_df.columns and 'forward_return' in predictions_df.columns:
            valid_preds = predictions_df[['predicted_return', 'forward_return']].dropna()
            if len(valid_preds) > 1:
                ic = np.corrcoef(valid_preds['predicted_return'], valid_preds['forward_return'])[0, 1]
        
        # Hit Rate (% of stocks where predicted direction matched actual direction)
        hit_rate = np.nan
        if predictions_df is not None and 'predicted_return' in predictions_df.columns and 'forward_return' in predictions_df.columns:
            valid_preds = predictions_df[['predicted_return', 'forward_return']].dropna()
            if len(valid_preds) > 0:
                predicted_direction = valid_preds['predicted_return'] > 0
                actual_direction = valid_preds['forward_return'] > 0
                hit_rate = (predicted_direction == actual_direction).sum() / len(valid_preds)
        
        metrics = {
            'total_return': cumulative_return,
            'avg_return': returns.mean(),
            'volatility': returns.std(),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'information_coefficient': ic,
            'hit_rate': hit_rate,
            'num_periods': len(returns)
        }
        
        return metrics
    
    def print_performance_summary(self, backtest_df: pd.DataFrame, predictions_df: pd.DataFrame = None):
        """Print performance summary"""
        metrics = self.get_performance_metrics(backtest_df, predictions_df)
        
        strategy_type = "LONG-ONLY" if self.long_only else "MARKET-NEUTRAL LONG/SHORT"
        
        print("\n" + "="*60)
        print(f"PORTFOLIO PERFORMANCE SUMMARY ({strategy_type})")
        print("="*60)
        print(f"Total Return:    {metrics.get('total_return', 0):.2%}")
        print(f"Avg Return:      {metrics.get('avg_return', 0):.4%}")
        print(f"Volatility:      {metrics.get('volatility', 0):.4%}")
        print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}")
        print(f"Win Rate:        {metrics.get('win_rate', 0):.2%}")
        ic = metrics.get('information_coefficient', np.nan)
        if not np.isnan(ic):
            print(f"Info Coeff (IC): {ic:.4f}")
        hit_rate = metrics.get('hit_rate', np.nan)
        if not np.isnan(hit_rate):
            print(f"Hit Rate:        {hit_rate:.2%}")
        print(f"Periods:         {metrics.get('num_periods', 0)}")
        
        # Add turnover stats if available
        if 'turnover' in backtest_df.columns:
            avg_turnover = backtest_df['turnover'].mean()
            total_trades = backtest_df['turnover'].sum()
            print(f"Avg Turnover:    {avg_turnover:.1f} positions/period")
            print(f"Total Trades:    {int(total_trades)}")
        
        print("="*60 + "\n")
    
    def get_current_positions(self) -> dict:
        """
        Get current portfolio positions
        
        Returns:
            Dict with current long and short positions
        """
        return {
            'long': self.current_holdings['long'].copy(),
            'short': self.current_holdings['short'].copy(),
            'num_long': len(self.current_holdings['long']),
            'num_short': len(self.current_holdings['short'])
        }
    
    def print_trade_list(self, new_portfolio: dict, predictions_df: pd.DataFrame = None):
        """Print actionable trade list for weekly execution"""
        strategy_type = "LONG-ONLY" if new_portfolio.get('long_only', True) else "MARKET-NEUTRAL"
        
        print("\n" + "="*80)
        print(f"WEEKLY REBALANCING TRADE LIST ({strategy_type})")
        print("="*80)
        
        # Trades to execute
        long_to_buy = new_portfolio.get('long_to_buy', [])
        long_to_sell = new_portfolio.get('long_to_sell', [])
        short_to_buy = new_portfolio.get('short_to_buy', [])
        short_to_sell = new_portfolio.get('short_to_sell', [])
        
        long_positions = new_portfolio.get('long_positions', {})
        short_positions = new_portfolio.get('short_positions', {})
        
        cash_percent = new_portfolio.get('cash_percent', 0)
        
        total_trades = len(long_to_buy) + len(long_to_sell) + len(short_to_buy) + len(short_to_sell)
        
        if total_trades == 0:
            print("\n✓ No trades needed - portfolio unchanged")
            if new_portfolio.get('long_only', True):
                print(f"  Holding: {len(long_positions)} positions, {cash_percent:.1f}% cash")
            return
        
        print(f"\nTotal Trades: {total_trades}")
        if new_portfolio.get('long_only', True):
            print(f"Cash Position: {cash_percent:.1f}% (risk-free rate)")
        
        # SELL orders (execute first)
        if long_to_sell or short_to_sell:
            print("\n[1] CLOSE POSITIONS (Execute First):")
            print("-" * 80)
            
            for ticker in long_to_sell:
                print(f"  CLOSE LONG  | {ticker:6s} | Close entire position")
            
            if not new_portfolio.get('long_only', True):
                for ticker in short_to_sell:
                    print(f"  COVER SHORT | {ticker:6s} | Cover entire position")
        
        # BUY orders (execute second)
        if long_to_buy or short_to_buy:
            print("\n[2] OPEN NEW POSITIONS (Execute Second):")
            print("-" * 80)
            
            for ticker in long_to_buy:
                if ticker in long_positions:
                    pos = long_positions[ticker]
                    pred_str = ""
                    if predictions_df is not None:
                        pred = predictions_df[predictions_df['ticker'] == ticker]
                        if not pred.empty:
                            pred_return = pred.iloc[0]['predicted_return']
                            pred_str = f" | Predicted: {pred_return:+.2%}"
                    print(f"  OPEN LONG   | {ticker:6s} | ${pos['dollars']:>12,.0f} ({pos['percent']:5.2f}%){pred_str}")
            
            for ticker in short_to_buy:
                if ticker in short_positions:
                    pos = short_positions[ticker]
                    pred_str = ""
                    if predictions_df is not None:
                        pred = predictions_df[predictions_df['ticker'] == ticker]
                        if not pred.empty:
                            pred_return = pred.iloc[0]['predicted_return']
                            pred_str = f" | Predicted: {pred_return:+.2%}"
                    print(f"  OPEN SHORT  | {ticker:6s} | ${pos['dollars']:>12,.0f} ({pos['percent']:5.2f}%){pred_str}")
        
        print("\n" + "="*80)
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Get history of all portfolio positions
        
        Returns:
            DataFrame with position history
        """
        return pd.DataFrame(self.positions)
