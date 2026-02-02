"""
XGBoost ranking model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from ..utils.config import config


class XGBoostRanker:
    """XGBoost model for stock ranking"""
    
    def __init__(self):
        """Initialize XGBoost ranker"""
        self.scaler = StandardScaler()
        
        # Get hyperparameters from config
        params = config.get('modeling', 'hyperparameters', 'xgboost', default={})
        
        self.model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1
        )
        
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame, target_col='forward_return_5d') -> tuple:
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
        
        # Technical features (exclude forward returns to avoid leakage)
        technical_features = [col for col in df.columns if any(
            x in col for x in ['historical_return', 'volatility', 'sma', 'rsi', 'volume']
        )]
        
        # Volume features
        volume_features = [col for col in df.columns if 'num_texts' in col or 'log_num' in col]
        
        # Combine all features
        feature_columns = list(set(sentiment_features + domain_features + technical_features + volume_features))
        
        # Only keep features that exist in dataframe
        feature_columns = [col for col in feature_columns if col in df.columns and col != target_col]
        
        # Remove forward-looking features (prevent leakage)
        feature_columns = [col for col in feature_columns if 'forward' not in col]
        
        # Remove any columns with all NaN
        feature_columns = [col for col in feature_columns if df[col].notna().any()]
        
        if not feature_columns:
            raise ValueError("No valid features found in dataframe")
        
        X = df[feature_columns].fillna(0)
        y = df[target_col]
        
        self.feature_columns = feature_columns
        
        return X, y, feature_columns
    
    def train(self, df: pd.DataFrame, target_col='forward_return_5d', test_size=0.3):
        """
        Train XGBoost model
        
        Args:
            df: Training dataframe
            target_col: Target column
            test_size: Test set size
        
        Returns:
            Training metrics
        """
        print(f"\nTraining XGBoost ranking model...")
        
        # Remove rows with NaN target
        df = df[df[target_col].notna()].copy()
        
        if len(df) < 20:
            print(f"Not enough data for training (need at least 20 samples, have {len(df)})")
            return None
        
        # Prepare features
        X, y, feature_columns = self.prepare_features(df, target_col)
        
        print(f"Features: {len(feature_columns)}")
        print(f"Samples: {len(X)}")
        
        # Train-test split (time-series aware)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Make predictions for additional metrics
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate Information Coefficient (IC)
        ic_train = np.corrcoef(y_train, y_pred_train)[0, 1]
        ic_test = np.corrcoef(y_test, y_pred_test)[0, 1]
        
        print(f"Train R²: {train_score:.4f} | IC: {ic_train:.4f}")
        print(f"Test R²: {test_score:.4f} | IC: {ic_test:.4f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'ic_train': ic_train,
            'ic_test': ic_test,
            'n_features': len(feature_columns),
            'n_samples': len(X)
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            df: Input dataframe
        
        Returns:
            Predicted returns
        """
        if self.feature_columns is None:
            raise ValueError("Model not trained yet")
        
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def train_on_period(self, df: pd.DataFrame, train_end_date, target_col='forward_return_5d', min_train_periods=20, window_days=None):
        """
        Train model on data up to (but not including) train_end_date
        Used for walk-forward analysis
        
        Args:
            df: Full dataframe with features
            train_end_date: Train on all data before this date
            target_col: Target column
            min_train_periods: Minimum number of training samples required
            window_days: If specified, use rolling window of last N days (e.g., 180)
                        If None, use expanding window (all historical data)
        
        Returns:
            Training metrics or None if insufficient data
        """
        # Filter training data (only past)
        if window_days is not None:
            # Rolling window: only use last window_days before train_end_date
            train_start_date = train_end_date - pd.Timedelta(days=window_days)
            train_df = df[(df['date'] >= train_start_date) & (df['date'] < train_end_date)].copy()
        else:
            # Expanding window: use all data before train_end_date
            train_df = df[df['date'] < train_end_date].copy()
        
        train_df = train_df[train_df[target_col].notna()]
        
        if len(train_df) < min_train_periods:
            print(f"  Insufficient training data: {len(train_df)} < {min_train_periods}")
            return None
        
        # Prepare features
        X, y, feature_columns = self.prepare_features(train_df, target_col)
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Scale and train (no train/test split - use all available past data)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, verbose=False)
        
        # Quick validation
        train_score = self.model.score(X_scaled, y)
        y_pred = self.model.predict(X_scaled)
        ic = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else 0
        
        return {
            'train_samples': len(X),
            'train_r2': train_score,
            'train_ic': ic
        }
    
    def rank_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks by predicted return
        
        Args:
            df: Input dataframe with features
        
        Returns:
            DataFrame with predictions and ranks
        """
        df = df.copy()
        df['predicted_return'] = self.predict(df)
        
        # Rank within each date
        if 'date' in df.columns:
            df['rank'] = df.groupby('date')['predicted_return'].rank(ascending=False)
            df['percentile'] = df.groupby('date')['predicted_return'].rank(pct=True)
        else:
            df['rank'] = df['predicted_return'].rank(ascending=False)
            df['percentile'] = df['predicted_return'].rank(pct=True)
        
        return df.sort_values('predicted_return', ascending=False)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_columns is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False)
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {path}")
