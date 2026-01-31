"""
Reddit data collector
Collects posts and comments from relevant subreddits
"""
import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import time

from ..utils.config import config
from ..utils.universe import get_all_tickers


class RedditCollector:
    """Collect Reddit posts and comments related to mining stocks"""
    
    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent
        )
        
        self.subreddits = config.get('data_collection', 'reddit', 'subreddits', default=[
            'investing', 'stocks', 'commodities', 'mining', 'wallstreetbets'
        ])
        
        self.tickers = get_all_tickers()
    
    def collect_posts(self, days_back=30, limit=100):
        """
        Collect posts mentioning mining stocks
        
        Args:
            days_back: Number of days to look back
            limit: Maximum posts per subreddit
        
        Returns:
            DataFrame with columns: ticker, text, timestamp, score, subreddit, url
        """
        print(f"Collecting Reddit posts from {len(self.subreddits)} subreddits...")
        
        all_data = []
        start_time = datetime.now() - timedelta(days=days_back)
        
        for subreddit_name in self.subreddits:
            print(f"  Collecting from r/{subreddit_name}...")
            
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for each ticker
                for ticker in self.tickers:
                    try:
                        # Search recent posts
                        for post in subreddit.search(ticker, limit=limit, time_filter='month'):
                            post_time = datetime.fromtimestamp(post.created_utc)
                            
                            if post_time < start_time:
                                continue
                            
                            all_data.append({
                                'ticker': ticker,
                                'text': f"{post.title} {post.selftext}",
                                'timestamp': post_time,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'subreddit': subreddit_name,
                                'url': post.url,
                                'source': 'reddit_post'
                            })
                        
                        # Rate limiting
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"    Error collecting {ticker}: {e}")
                        continue
                
            except Exception as e:
                print(f"  Error accessing r/{subreddit_name}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"Collected {len(df)} Reddit posts")
        
        return df
    
    def collect_comments(self, posts_df, max_comments_per_post=50):
        """
        Collect comments from posts
        
        Args:
            posts_df: DataFrame of posts from collect_posts
            max_comments_per_post: Maximum comments to collect per post
        
        Returns:
            DataFrame with columns: ticker, text, timestamp, score, parent_post
        """
        print(f"Collecting comments from {len(posts_df)} posts...")
        
        all_comments = []
        
        for idx, post_row in posts_df.iterrows():
            try:
                # Extract post ID from URL
                post_id = post_row['url'].split('/')[-3] if '/' in post_row['url'] else None
                
                if not post_id:
                    continue
                
                submission = self.reddit.submission(id=post_id)
                submission.comments.replace_more(limit=0)  # Remove "load more comments"
                
                for comment in submission.comments.list()[:max_comments_per_post]:
                    all_comments.append({
                        'ticker': post_row['ticker'],
                        'text': comment.body,
                        'timestamp': datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'parent_post': post_row['url'],
                        'source': 'reddit_comment'
                    })
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error collecting comments: {e}")
                continue
        
        df = pd.DataFrame(all_comments)
        print(f"Collected {len(df)} comments")
        
        return df
    
    def collect_all(self, days_back=30, save=True):
        """
        Collect both posts and comments
        
        Args:
            days_back: Number of days to look back
            save: Whether to save to disk
        
        Returns:
            Combined DataFrame
        """
        # Collect posts
        posts_df = self.collect_posts(days_back=days_back)
        
        if posts_df.empty:
            print("No posts collected")
            return pd.DataFrame()
        
        # Collect comments
        comments_df = self.collect_comments(posts_df)
        
        # Combine
        combined_df = pd.concat([posts_df, comments_df], ignore_index=True)
        
        # Clean and deduplicate
        combined_df = combined_df.drop_duplicates(subset=['text', 'timestamp'])
        combined_df = combined_df[combined_df['text'].str.len() > 20]  # Remove very short texts
        
        if save:
            from ..utils.data_utils import save_dataframe
            save_path = config.data_dir / 'raw' / f'reddit_{datetime.now().strftime("%Y%m%d")}.parquet'
            save_dataframe(combined_df, save_path)
            print(f"Saved to {save_path}")
        
        return combined_df
