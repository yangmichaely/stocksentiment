"""
Evaluation modules
"""
from .metrics import (
    information_coefficient,
    hit_rate,
    calculate_alpha,
    sharpe_ratio,
    max_drawdown,
    evaluate_predictions,
    print_evaluation_report
)

__all__ = [
    'information_coefficient',
    'hit_rate',
    'calculate_alpha',
    'sharpe_ratio',
    'max_drawdown',
    'evaluate_predictions',
    'print_evaluation_report'
]
