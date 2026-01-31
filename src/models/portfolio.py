"""
Portfolio construction and backtesting
"""
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.config import config


class Portfolio:
    """Long-short portfolio based on sentiment rankings"""
    
    def __init__(self, long_percentile=80, short_percentile=20):
        """
        Initialize portfolio
        
        Args:
            long_percentile: Top percentile for long positions
            short_percentile: Bottom percentile for short positions
        """
        self.long_percentile = long_percentile
        self.short_percentile = short_percentile
        self.positions = []
        self.current_holdings = {'long': [], 'short': []}  # Track current positions
    
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
    
    def construct_portfolio(self, ranked_df: pd.DataFrame, date=None, incremental=True, 
                           portfolio_value=10_000_000) -> dict:
        """
        Construct long-short portfolio with incremental rebalancing and position sizing
        
        Args:
            ranked_df: DataFrame with predictions and ranks
            date: Optional date for this portfolio snapshot
            incremental: If True, only adjust positions that change percentiles
            portfolio_value: Total portfolio value in dollars (default $10M)
        
        Returns:
            Dict with long and short positions, weights, and dollar amounts
        """
        if 'percentile' not in ranked_df.columns:
            print("Warning: percentile column not found, using top/bottom 20%")
            ranked_df = ranked_df.sort_values('predicted_return', ascending=False)
            n = len(ranked_df)
            top_n = max(1, int(n * 0.2))
            
            long_stocks = ranked_df.head(top_n)
            short_stocks = ranked_df.tail(top_n)
        else:
            # Long top percentile
            long_stocks = ranked_df[
                ranked_df['percentile'] >= self.long_percentile / 100
            ]
            
            # Short bottom percentile
            short_stocks = ranked_df[
                ranked_df['percentile'] <= self.short_percentile / 100
            ]
        
        new_long = long_stocks['ticker'].tolist()
        new_short = short_stocks['ticker'].tolist()
        
        # Calculate position weights
        long_weights = self.calculate_position_weights(ranked_df, new_long, 'long')
        short_weights = self.calculate_position_weights(ranked_df, new_short, 'short')
        
        # Calculate dollar amounts (50% long, 50% short for market-neutral)
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
            'total_value': portfolio_value
        }
        
        self.positions.append(portfolio)
        
        return portfolio
    
    def backtest(self, predictions_df: pd.DataFrame, returns_col='forward_return', incremental=True) -> pd.DataFrame:
        """
        Backtest portfolio strategy with incremental rebalancing
        
        Args:
            predictions_df: DataFrame with predictions and actual returns
            returns_col: Column with actual returns
            incremental: If True, track turnover from incremental changes
        
        Returns:
            DataFrame with backtest results
        """
        print("\nBacktesting portfolio strategy...")
        
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
    
    def get_performance_metrics(self, backtest_df: pd.DataFrame) -> dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            backtest_df: Backtest results dataframe
        
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
        
        metrics = {
            'total_return': cumulative_return,
            'avg_return': returns.mean(),
            'volatility': returns.std(),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_periods': len(returns)
        }
        
        return metrics
    
    def print_performance_summary(self, backtest_df: pd.DataFrame):
        """Print performance summary"""
        metrics = self.get_performance_metrics(backtest_df)
        
        print("\n" + "="*50)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total Return:    {metrics.get('total_return', 0):.2%}")
        print(f"Avg Return:      {metrics.get('avg_return', 0):.4%}")
        print(f"Volatility:      {metrics.get('volatility', 0):.4%}")
        print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}")
        print(f"Win Rate:        {metrics.get('win_rate', 0):.2%}")
        print(f"Periods:         {metrics.get('num_periods', 0)}")
        
        # Add turnover stats if available
        if 'turnover' in backtest_df.columns:
            avg_turnover = backtest_df['turnover'].mean()
            total_trades = backtest_df['turnover'].sum()
            print(f"Avg Turnover:    {avg_turnover:.1f} positions/period")
            print(f"Total Trades:    {int(total_trades)}")
        
        print("="*50 + "\n")
    
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
    
    def get_position_history(self) -> pd.DataFrame:
        """
        Get history of all portfolio positions
        
        Returns:
            DataFrame with position history
        """
        return pd.DataFrame(self.positions)
