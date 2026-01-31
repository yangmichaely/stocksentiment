"""
Domain-specific sentiment lexicon for mining & materials industry
"""
import re
from typing import Dict, List
import pandas as pd

from ..utils.config import config


class DomainLexicon:
    """Mining industry-specific sentiment lexicon"""
    
    def __init__(self):
        """Initialize domain lexicons"""
        
        # Supply risk keywords (negative sentiment)
        self.supply_risk_keywords = {
            'high': ['strike', 'shutdown', 'halt', 'suspension', 'closure', 'accident', 
                     'fatality', 'explosion', 'collapse', 'disaster'],
            'medium': ['outage', 'disruption', 'delay', 'slowdown', 'maintenance', 
                       'flood', 'fire', 'protest'],
            'low': ['weather', 'power', 'issue', 'problem']
        }
        
        # Demand signal keywords (positive/negative)
        self.demand_keywords = {
            'positive': ['china pmi increase', 'ev demand surge', 'infrastructure spending',
                        'construction boom', 'manufacturing growth', 'demand increase',
                        'shortage', 'tight supply', 'deficit'],
            'negative': ['china slowdown', 'demand decline', 'oversupply', 'surplus',
                        'inventory buildup', 'demand destruction', 'recession']
        }
        
        # Cost pressure keywords (negative sentiment)
        self.cost_keywords = {
            'high': ['energy crisis', 'diesel shortage', 'labor shortage', 'wage inflation'],
            'medium': ['cost increase', 'price pressure', 'rising costs', 'input costs'],
            'low': ['electricity', 'fuel']
        }
        
        # Regulatory risk keywords (negative sentiment)
        self.regulatory_keywords = {
            'high': ['permit denied', 'environmental violation', 'lawsuit', 'fine',
                    'suspension', 'investigation'],
            'medium': ['regulatory review', 'permit delay', 'esg concern', 'compliance'],
            'low': ['regulation', 'environmental', 'tailings']
        }
        
        # Production/capacity keywords (positive sentiment)
        self.production_keywords = {
            'positive': ['production increase', 'capacity expansion', 'new mine',
                        'higher output', 'ramp up', 'record production', 'guidance raise'],
            'negative': ['production cut', 'lower guidance', 'miss', 'below expectations']
        }
    
    def calculate_supply_risk_score(self, text: str) -> float:
        """
        Calculate supply risk score (-1 to 0, where -1 is highest risk)
        
        Args:
            text: Input text
        
        Returns:
            Supply risk score
        """
        text_lower = text.lower()
        score = 0
        
        # Check for supply risk keywords
        for keyword in self.supply_risk_keywords['high']:
            if keyword in text_lower:
                score -= 0.4
        
        for keyword in self.supply_risk_keywords['medium']:
            if keyword in text_lower:
                score -= 0.2
        
        for keyword in self.supply_risk_keywords['low']:
            if keyword in text_lower:
                score -= 0.1
        
        return max(score, -1.0)
    
    def calculate_demand_score(self, text: str) -> float:
        """
        Calculate demand signal score (-1 to 1)
        
        Args:
            text: Input text
        
        Returns:
            Demand score
        """
        text_lower = text.lower()
        score = 0
        
        for keyword in self.demand_keywords['positive']:
            if keyword in text_lower:
                score += 0.3
        
        for keyword in self.demand_keywords['negative']:
            if keyword in text_lower:
                score -= 0.3
        
        return max(min(score, 1.0), -1.0)
    
    def calculate_cost_pressure_score(self, text: str) -> float:
        """
        Calculate cost pressure score (-1 to 0, where -1 is highest pressure)
        
        Args:
            text: Input text
        
        Returns:
            Cost pressure score
        """
        text_lower = text.lower()
        score = 0
        
        for keyword in self.cost_keywords['high']:
            if keyword in text_lower:
                score -= 0.4
        
        for keyword in self.cost_keywords['medium']:
            if keyword in text_lower:
                score -= 0.2
        
        for keyword in self.cost_keywords['low']:
            if keyword in text_lower:
                score -= 0.1
        
        return max(score, -1.0)
    
    def calculate_regulatory_risk_score(self, text: str) -> float:
        """
        Calculate regulatory risk score (-1 to 0, where -1 is highest risk)
        
        Args:
            text: Input text
        
        Returns:
            Regulatory risk score
        """
        text_lower = text.lower()
        score = 0
        
        for keyword in self.regulatory_keywords['high']:
            if keyword in text_lower:
                score -= 0.4
        
        for keyword in self.regulatory_keywords['medium']:
            if keyword in text_lower:
                score -= 0.2
        
        for keyword in self.regulatory_keywords['low']:
            if keyword in text_lower:
                score -= 0.1
        
        return max(score, -1.0)
    
    def calculate_production_sentiment(self, text: str) -> float:
        """
        Calculate production-specific sentiment (-1 to 1)
        
        Args:
            text: Input text
        
        Returns:
            Production sentiment score
        """
        text_lower = text.lower()
        score = 0
        
        for keyword in self.production_keywords['positive']:
            if keyword in text_lower:
                score += 0.3
        
        for keyword in self.production_keywords['negative']:
            if keyword in text_lower:
                score -= 0.3
        
        return max(min(score, 1.0), -1.0)
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text using domain lexicon
        
        Args:
            text: Input text
        
        Returns:
            Dict with all domain-specific scores
        """
        return {
            'supply_risk_score': self.calculate_supply_risk_score(text),
            'demand_score': self.calculate_demand_score(text),
            'cost_pressure_score': self.calculate_cost_pressure_score(text),
            'regulatory_risk_score': self.calculate_regulatory_risk_score(text),
            'production_sentiment': self.calculate_production_sentiment(text)
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column='text') -> pd.DataFrame:
        """
        Add domain-specific scores to dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of column containing text
        
        Returns:
            DataFrame with added domain-specific columns
        """
        if df.empty or text_column not in df.columns:
            return df
        
        print(f"Calculating domain-specific sentiment scores...")
        
        domain_scores = df[text_column].apply(self.analyze_text)
        domain_df = pd.DataFrame(domain_scores.tolist())
        
        result_df = pd.concat([df.reset_index(drop=True), domain_df], axis=1)
        
        return result_df


def get_domain_lexicon():
    """Get domain lexicon instance (singleton pattern)"""
    if not hasattr(get_domain_lexicon, 'instance'):
        get_domain_lexicon.instance = DomainLexicon()
    return get_domain_lexicon.instance
