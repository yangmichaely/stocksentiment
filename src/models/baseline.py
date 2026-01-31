"""
Baseline models for stock ranking
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from ..utils.config import config


class BaselineModel:
    """Baseline regression/classification models"""
    
    def __init__(self, model_type='logistic'):
        """
        Initialize baseline model
        
        Args:
            model_type: 'logistic' or 'linear'
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame, target_col='outperform_benchmark') -> tuple:
        """
        Prepare feature matrix and target
        
        Args:
            df: Input dataframe
            target_col: Target column name
        
        Returns:
            X, y, feature_columns
        """
        # Sentiment features
        sentiment_features = [col for col in df.columns if 'sentiment' in col.lower()]
        
        # Domain-specific features
        domain_features = [col for col in df.columns if any(
            x in col for x in ['supply_risk', 'demand', 'cost_pressure', 'regulatory', 'production']
        )]
        
        # Technical features
        technical_features = [col for col in df.columns if any(
            x in col for x in ['return', 'volatility', 'sma', 'rsi', 'volume']
        )]
        
        # Combine all features
        feature_columns = list(set(sentiment_features + domain_features + technical_features))
        
        # Only keep features that exist in dataframe
        feature_columns = [col for col in feature_columns if col in df.columns and col != target_col]
        
        # Remove any columns with all NaN
        feature_columns = [col for col in feature_columns if df[col].notna().any()]
        
        if not feature_columns:
            raise ValueError("No valid features found in dataframe")
        
        X = df[feature_columns].fillna(0)
        y = df[target_col]
        
        self.feature_columns = feature_columns
        
        return X, y, feature_columns
    
    def train(self, df: pd.DataFrame, target_col='outperform_benchmark', test_size=0.3):
        """
        Train baseline model
        
        Args:
            df: Training dataframe
            target_col: Target column
            test_size: Test set size
        
        Returns:
            Training and test scores
        """
        print(f"\nTraining {self.model_type} regression model...")
        
        # Remove rows with NaN target
        df = df[df[target_col].notna()].copy()
        
        if len(df) < 20:
            print(f"Not enough data for training (need at least 20 samples, have {len(df)})")
            return None
        
        # Prepare features
        X, y, feature_columns = self.prepare_features(df, target_col)
        
        print(f"Features: {len(feature_columns)}")
        print(f"Samples: {len(X)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # Time-series split
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Train score: {train_score:.4f}")
        print(f"Test score: {test_score:.4f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'n_features': len(feature_columns),
            'n_samples': len(X)
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            df: Input dataframe
        
        Returns:
            Predictions
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained yet")
        
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'logistic':
            # Return probabilities for positive class
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance/coefficients
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_columns is None:
            return pd.DataFrame()
        
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0] if self.model_type == 'logistic' else self.model.coef_)
            
            df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            })
            
            return df.sort_values('importance', ascending=False)
        
        return pd.DataFrame()
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        print(f"Model loaded from {path}")
