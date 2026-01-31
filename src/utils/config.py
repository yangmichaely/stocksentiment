"""
Configuration utilities
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration manager"""
    
    def __init__(self, config_path='config.yaml'):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = self.project_root / config_path
        
        # Load YAML config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # API Keys from environment
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        # Paths
        self.data_dir = self.project_root / self.config['output']['paths']['data']
        self.models_dir = self.project_root / self.config['output']['paths']['models']
        self.results_dir = self.project_root / self.config['output']['paths']['results']
        self.logs_dir = self.project_root / self.config['output']['paths']['logs']
        
        # Create directories
        self._create_dirs()
    
    def _create_dirs(self):
        """Create necessary directories"""
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, *keys, default=None):
        """Get nested config value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

# Global config instance
config = Config()
