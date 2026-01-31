"""
Evaluation metrics for stock prediction
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def information_coefficient(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Information Coefficient (Spearman correlation)
    
    Args:
        predictions: Predicted values
        actuals: Actual values
    
    Returns:
        IC value
    """
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[mask]
    actuals = actuals[mask]
    
    if len(predictions) < 2:
        return 0.0
    
    ic, _ = spearmanr(predictions, actuals)
    return ic if not np.isnan(ic) else 0.0


def hit_rate(predictions: np.ndarray, actuals: np.ndarray, threshold=0) -> float:
    """
    Calculate hit rate (% of correct direction predictions)
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        threshold: Threshold for positive prediction
    
    Returns:
        Hit rate (0-1)
    """
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions = predictions[mask]
    actuals = actuals[mask]
    
    if len(predictions) == 0:
        return 0.0
    
    # Check if signs match
    correct = np.sign(predictions - threshold) == np.sign(actuals - threshold)
    return correct.sum() / len(correct)


def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate alpha (excess return over benchmark)
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
    
    Returns:
        Alpha
    """
    if len(portfolio_returns) != len(benchmark_returns):
        return 0.0
    
    return portfolio_returns.mean() - benchmark_returns.mean()


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.02, periods_per_year=52) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (52 for weekly)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_return = returns.mean() - (risk_free_rate / periods_per_year)
    return (excess_return / returns.std()) * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns: Cumulative return series
    
    Returns:
        Maximum drawdown (negative value)
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    cumulative = (1 + cumulative_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def turnover_rate(positions_df: pd.DataFrame) -> float:
    """
    Calculate portfolio turnover rate
    
    Args:
        positions_df: DataFrame with columns 'date' and 'ticker'
    
    Returns:
        Average turnover rate
    """
    if len(positions_df) < 2:
        return 0.0
    
    turnovers = []
    
    for i in range(1, len(positions_df)):
        prev_positions = set(positions_df.iloc[i-1]['tickers'])
        curr_positions = set(positions_df.iloc[i]['tickers'])
        
        # Calculate turnover as % of positions changed
        total_positions = len(prev_positions.union(curr_positions))
        if total_positions > 0:
            changed = len(prev_positions.symmetric_difference(curr_positions))
            turnover = changed / total_positions
            turnovers.append(turnover)
    
    return np.mean(turnovers) if turnovers else 0.0


def rank_correlation(ranks1: np.ndarray, ranks2: np.ndarray) -> float:
    """
    Calculate rank correlation (Spearman)
    
    Args:
        ranks1: First ranking
        ranks2: Second ranking
    
    Returns:
        Rank correlation
    """
    mask = ~(np.isnan(ranks1) | np.isnan(ranks2))
    ranks1 = ranks1[mask]
    ranks2 = ranks2[mask]
    
    if len(ranks1) < 2:
        return 0.0
    
    corr, _ = spearmanr(ranks1, ranks2)
    return corr if not np.isnan(corr) else 0.0


def evaluate_predictions(predictions_df: pd.DataFrame, 
                         prediction_col='predicted_return',
                         actual_col='forward_return_5d') -> dict:
    """
    Comprehensive evaluation of predictions
    
    Args:
        predictions_df: DataFrame with predictions and actuals
        prediction_col: Column with predictions
        actual_col: Column with actual returns
    
    Returns:
        Dict with evaluation metrics
    """
    # Remove NaN values
    valid_df = predictions_df[[prediction_col, actual_col]].dropna()
    
    if len(valid_df) == 0:
        return {
            'ic': 0.0,
            'hit_rate': 0.0,
            'mse': 0.0,
            'mae': 0.0,
            'r2': 0.0,
            'n_samples': 0
        }
    
    predictions = valid_df[prediction_col].values
    actuals = valid_df[actual_col].values
    
    # Information Coefficient
    ic = information_coefficient(predictions, actuals)
    
    # Hit Rate
    hr = hit_rate(predictions, actuals)
    
    # MSE
    mse = np.mean((predictions - actuals) ** 2)
    
    # MAE
    mae = np.mean(np.abs(predictions - actuals))
    
    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'ic': ic,
        'hit_rate': hr,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'n_samples': len(valid_df)
    }


def print_evaluation_report(metrics: dict):
    """Print formatted evaluation report"""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Information Coefficient: {metrics.get('ic', 0):.4f}")
    print(f"Hit Rate:               {metrics.get('hit_rate', 0):.2%}")
    print(f"R²:                     {metrics.get('r2', 0):.4f}")
    print(f"MSE:                    {metrics.get('mse', 0):.6f}")
    print(f"MAE:                    {metrics.get('mae', 0):.6f}")
    print(f"Samples:                {metrics.get('n_samples', 0)}")
    print("="*50 + "\n")
