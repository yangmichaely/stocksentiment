"""
Data collection modules
"""
from .reddit_collector import RedditCollector
from .news_collector import NewsCollector

__all__ = [
    'RedditCollector',
    'NewsCollector'
]
