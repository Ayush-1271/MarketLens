import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import torch

class BaselineModels:
    def __init__(self):
        self.models = {}
        self.results = {}

    def _extract_arrays(self, dataloader):
        """Helper to extract numpy arrays from dataloader."""
        X_list, y_list = [], []
        for batch in dataloader:
            features = batch['features'].numpy()
            target = batch['target_return'].numpy()
            X_list.append(features)
            y_list.append(target)
        
        if not X_list:
            raise ValueError("DataLoader is empty. Not enough data for evaluation.")
            
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0).flatten()
        return X, y

    def _evaluate(self, name, y_true, y_pred):
        """Calculates metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate Directional Accuracy
        # Assuming y_true and y_pred are returns
        # Direction is correct if signs match
        true_sign = np.sign(y_true)
        pred_sign = np.sign(y_pred)
        dir_acc = np.mean(true_sign == pred_sign)
        
        return {
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mae,
            "R2": r2,
            "Dir_Acc": dir_acc
        }

    def run_all(self, train_loader, test_loader, scaler=None):
        """Runs all baselines and returns a report.
           If scaler is provided, evaluates on Unscaled Log Returns.
        """
        print("Extracting data for baselines...")
        X_train, y_train = self._extract_arrays(train_loader)
        X_test, y_test = self._extract_arrays(test_loader)
        
        # Inverse Transform Targets if Scaler provided
        # y is 'log_returns' (next step). 
        # config.FEATURE_COLS = [..., 'log_returns', ...]
        # We need the index of 'log_returns' in the scaler.
        
        if scaler:
            # Find index of 'log_returns'
            # Assuming config is available or we pass the index
            # Hardcoding lookup or import config
            from src import config
            idx = config.FEATURE_COLS.index('log_returns')
            
            # StandardHelper (sklearn StandardScaler) stores mean_ and scale_
            mu = scaler.mean_[idx]
            sigma = scaler.scale_[idx]
            
            # Unscale Targets
            y_train = (y_train * sigma) + mu
            y_test = (y_test * sigma) + mu
            
            # Note: X_train/X_test are still SCALED variables (features).
            # Predictions made using X will be in SCALED units (because X is scaled?).
            # Wait. 
            # Naive Persistence: returns X[:, -1, idx] -> This is a scaled feature.
            # So its prediction is scaled. We Unscale it -> Correct.
            # Linear Regression: Fits on Scaled X -> Scaled y. Prediction is Scaled. Unscale -> Correct.
            
        # Flatten X for non-sequential models (LinearReg, XGB)
        # X is (N, T, F). We can flatten to (N, T*F) or just use the last step?
        # Usually for simple models we might use just the last step features or flatten.
        # Let's flatten to capture the window context.
        N_train, T, F = X_train.shape
        X_train_flat = X_train.reshape(N_train, -1)
        
        N_test, _, _ = X_test.shape
        X_test_flat = X_test.reshape(N_test, -1)

        # -----------------------------------------------
        # 1. Naive Persistence (Last Step Return)
        # -----------------------------------------------
        y_pred_persist = X_test[:, -1, 0] 
        if scaler: y_pred_persist = (y_pred_persist * sigma) + mu
        self.results["Naive_Persistence"] = self._evaluate("Naive_Persistence", y_test, y_pred_persist)

        # -----------------------------------------------
        # 2. Rolling Mean (Critical Baseline)
        # -----------------------------------------------
        y_pred_rolling = np.mean(X_test[:, :, 0], axis=1)
        if scaler: y_pred_rolling = (y_pred_rolling * sigma) + mu
        self.results["Rolling_Mean"] = self._evaluate("Rolling_Mean", y_test, y_pred_rolling)

        # -----------------------------------------------
        # 3. Linear Regression
        # -----------------------------------------------
        lr = LinearRegression()
        lr.fit(X_train_flat, y_train) # Note: Fits on UNSCALED target if y_train was unscaled above?
                                      # Wait! I unscaled y_train above at line ~70.
                                      # So LR fits Scaled X -> Unscaled y.
                                      # Prediction will be Unscaled automatically.
                                      # BUT X_train_flat is Scaled.
                                      # If I unscaled y_train, then LR learns mapping ScaledX->UnscaledY.
                                      # So y_pred_lr is Unscaled. Correct.
                                      
        # Let's verify where I unscaled y_train.
        # "y_train = (y_train * sigma) + mu"
        # Yes, I overwrote y_train.
        # So models trained AFTER this line use Unscaled Targets.
        
        y_pred_lr = lr.predict(X_test_flat)
        # Result is already Unscaled.
        self.results["Linear_Regression"] = self._evaluate("Linear_Regression", y_test, y_pred_lr)

        # -----------------------------------------------
        # 4. XGBoost (Shallow)
        # -----------------------------------------------
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=3, # Shallow as per plan
            learning_rate=0.1, 
            objective='reg:squarederror',
            n_jobs=1
        )
        xgb_model.fit(X_train_flat, y_train) # Fits Scaled X -> Unscaled y
        y_pred_xgb = xgb_model.predict(X_test_flat)
        self.results["XGBoost"] = self._evaluate("XGBoost", y_test, y_pred_xgb)

        return pd.DataFrame(self.results).T

