"""
UrbanCanopy-Opt: AI Allometry & Uncertainty Quantification Module
Trains predictive models to bridge the gap between LiDAR geometry (height, crown)
and field inventory attributes (trunk circumference, age stage).
Implements Monte Carlo error propagation for monetary valuation.
Author: Candidate / Heudiasyc Internship Preparation
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "paris_trees_clean.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

class TreeAllometryAI:
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.residual_std = None
        self.feature_names = None
        os.makedirs(MODELS_DIR, exist_ok=True)

    def train_circumference_model(self, df):
        """
        Trains an ensemble regression pipeline to predict trunk circumference (cm)
        from LiDAR observable features: height (m), crown diameter (m), crown area (m2), and genre group.
        """
        print("[*] Training Circumference Allometry Model (LiDAR -> DBH/Circumference)...")
        
        X = df[["hauteurenm", "crown_diameter_m", "crown_area_m2", "genre_group"]]
        y = df["circonferenceencm"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", ["hauteurenm", "crown_diameter_m", "crown_area_m2"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["genre_group"])
            ]
        )
        
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("reg", GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        residuals = y_test - y_pred
        self.residual_std = float(np.std(residuals))
        self.regressor = pipeline
        
        print(f"    [+] Model Performance: R2 = {r2:.3f} | MAE = {mae:.2f} cm | RMSE = {rmse:.2f} cm")
        print(f"    [+] Residual standard deviation (uncertainty sigma): {self.residual_std:.2f} cm")
        
        # Save model
        joblib.dump(self.regressor, os.path.join(MODELS_DIR, "allometry_regressor.joblib"))
        return {"r2": r2, "mae": mae, "rmse": rmse, "residual_std": self.residual_std}

    def train_stage_classifier(self, df):
        """
        Trains a classifier to predict developmental stage (Jeune, Jeune_Adulte, Adulte, Mature)
        from height, predicted circumference, crown size, and genre group.
        """
        print("[*] Training Developmental Stage Classifier...")
        
        X = df[["hauteurenm", "crown_diameter_m", "circonferenceencm", "genre_group"]]
        y = df["stage"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", ["hauteurenm", "crown_diameter_m", "circonferenceencm"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["genre_group"])
            ]
        )
        
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = pipeline.score(X_test, y_test)
        print(f"    [+] Stage Classification Accuracy: {acc * 100:.2f}%")
        
        self.classifier = pipeline
        joblib.dump(self.classifier, os.path.join(MODELS_DIR, "stage_classifier.joblib"))
        return {"accuracy": acc}

    def predict_with_uncertainty(self, df_lidar, n_simulations=500):
        """
        Predicts trunk circumference and generates Monte Carlo realizations 
        to propagate LiDAR prediction errors to valuation models.
        
        Returns:
            mean_pred: np.array of expected circumferences (cm)
            mc_samples: np.ndarray of shape (len(df_lidar), n_simulations)
        """
        if self.regressor is None:
            model_path = os.path.join(MODELS_DIR, "allometry_regressor.joblib")
            if os.path.exists(model_path):
                self.regressor = joblib.load(model_path)
            else:
                raise ValueError("Model must be trained or loaded before prediction.")
                
        features = df_lidar[["hauteurenm", "crown_diameter_m", "crown_area_m2", "genre_group"]]
        mean_pred = self.regressor.predict(features)
        
        # Monte Carlo error sampling: circumference ~ Normal(mean_pred, residual_std)
        # Clamped to positive realistic bounds (> 10 cm)
        sigma = self.residual_std if self.residual_std else 12.0
        np.random.seed(42)
        noise = np.random.normal(0, sigma, size=(n_simulations, len(mean_pred)))
        mc_samples = np.clip(mean_pred + noise, 10.0, 600.0).T  # Shape: (N_trees, n_simulations)
        
        return mean_pred, mc_samples

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DATA_PATH):
        from data_loader import fetch_paris_trees, clean_and_enrich_trees
        df = clean_and_enrich_trees(fetch_paris_trees(limit=3000))
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
    ai = TreeAllometryAI()
    reg_metrics = ai.train_circumference_model(df)
    clf_metrics = ai.train_stage_classifier(df)
    
    # Test Monte Carlo sampling on 5 sample trees
    test_sample = df.head(5)
    means, mc_draws = ai.predict_with_uncertainty(test_sample, n_simulations=10)
    print("\n[+] Verification of Monte Carlo draws on first 3 trees:")
    for idx in range(3):
        print(f"Tree #{idx} ({test_sample.iloc[idx]['genre']}): Predicted Circumference = {means[idx]:.1f} cm | 95% CI: [{np.percentile(mc_draws[idx], 2.5):.1f}, {np.percentile(mc_draws[idx], 97.5):.1f}] cm (True = {test_sample.iloc[idx]['circonferenceencm']:.1f} cm)")
