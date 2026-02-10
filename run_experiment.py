import argparse
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from src import config
from src.data_loader import UnifiedDataLoader
from src.model import HybridModel
from src.train import Trainer
from src.baselines import BaselineModels
from src.metrics import calculate_metrics

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_serializable(obj):
    """Recursively converts numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return make_serializable(obj.tolist())
    elif torch.is_tensor(obj):
        return make_serializable(obj.detach().cpu().numpy())
    elif obj is None:
        return None
    return str(obj) # Fallback for unknown types (e.g. timestamps)

def is_trainable_ticker(ticker):
    """Checks if a ticker meets data sufficiency criteria for deep training."""
    report_dir = os.path.join(config.PROJECT_ROOT, "reports", ticker)
    metrics_path = os.path.join(report_dir, "metrics_all.json")
    
    if not os.path.exists(metrics_path):
        return False, "No metrics_all.json found (Baselines not run?)"
        
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        stats = metrics.get("data_stats", {})
        if not stats:
            return False, "No data_stats in metrics file"
            
        # 1. Total Rows
        total_rows = stats.get("total_rows", 0)
        # Fallback for legacy runs where total_rows might be 0
        if total_rows == 0:
            total_rows = stats.get("train_samples", 0) + stats.get("val_samples", 0)
            
        if total_rows < config.MIN_ROWS:
            return False, f"Insufficient total rows ({total_rows} < {config.MIN_ROWS})"
            
        # 2. Split Sizes
        if stats.get("train_samples", 0) < config.MIN_TRAIN_SAMPLES:
            return False, f"Insufficient train samples ({stats.get('train_samples')} < {config.MIN_TRAIN_SAMPLES})"
            
        if stats.get("val_samples", 0) < config.MIN_VAL_SAMPLES:
            return False, f"Insufficient val samples ({stats.get('val_samples')} < {config.MIN_VAL_SAMPLES})"
            
        # 3. Directional Balance (Strict Gate)
        n_valid = stats.get("dir_n_valid", 0)
        n_up = stats.get("dir_n_up", 0)
        n_down = stats.get("dir_n_down", 0)
        
        limit_valid = getattr(config, 'MIN_DIR_VALID_STRICT', 200)
        limit_class = getattr(config, 'MIN_DIR_PER_CLASS_STRICT', 30)
        
        if n_valid < limit_valid:
             return False, f"Insufficient valid direction samples ({n_valid} < {limit_valid})"
             
        if n_up < limit_class or n_down < limit_class:
             return False, f"Imbalanced direction classes (Up={n_up}, Down={n_down} < {limit_class})"
             
        # 4. Minority Class Ratio Check (Prevent extreme imbalance)
        minority_ratio = min(n_up, n_down) / max(1, (n_up + n_down))
        if minority_ratio < 0.05:
            return False, f"Direction too imbalanced (minority_ratio={minority_ratio:.3f} < 0.05)"
            
        return True, "OK"
        
    except Exception as e:
        return False, f"Error checking trainability: {e}"

def run_baselines_only(ticker):
    """Runs only baselines for a ticker and saves results."""
    logger.info(f"Running Baselines for: {ticker}")
    
    report_dir = os.path.join(config.PROJECT_ROOT, "reports", ticker)
    
    try:
        os.makedirs(report_dir, exist_ok=True)
        
        # Data Loading
        loader = UnifiedDataLoader(ticker, market_type="auto") 
        try:
            train_loader, val_loader, test_loader, stats = loader.get_data_loaders()
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {ticker} (Data Error): {e}")
            return

        # Calculate Data Stats for Gate
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        n_test = len(test_loader.dataset)
        
        # Robust total_rows calculation
        if hasattr(loader, 'df') and loader.df is not None:
             total_rows = len(loader.df)
        else:
             total_rows = n_train + n_val + n_test
        
        # Direction Stats from Validation Set
        y_dir_all = []
        for batch in val_loader:
             y_dir_all.append(batch['target_direction'].numpy())
        
        if y_dir_all:
            y_dir = np.concatenate(y_dir_all).flatten()
            valid_mask = (y_dir != -1)
            dir_n_valid = int(valid_mask.sum())
            if dir_n_valid > 0:
                y_valid = y_dir[valid_mask]
                dir_n_up = int((y_valid == 1).sum())
                dir_n_down = int((y_valid == 0).sum())
            else:
                dir_n_up = 0
                dir_n_down = 0
        else:
            dir_n_valid = 0
            dir_n_up = 0
            dir_n_down = 0
            
        data_stats = {
            "total_rows": total_rows,
            "train_samples": n_train,
            "val_samples": n_val,
            "dir_n_valid": dir_n_valid,
            "dir_n_up": dir_n_up,
            "dir_n_down": dir_n_down
        }
    
        # Run Baselines
        bl = BaselineModels()
        # Pass scaler to evaluate on Unscaled Log Returns
        scaler = stats.get('scaler', None)
        bl_results = bl.run_all(train_loader, test_loader, scaler=scaler)
        
        # Save as CSV (Legacy support)
        save_path = os.path.join(report_dir, "baselines.csv")
        bl_results.to_csv(save_path)
        logger.info(f"Baselines saved to {save_path}")
    
        # Unified Metrics JSON
        metrics_all = {
            "ticker": ticker,
            "data_stats": data_stats,
            "baselines": bl_results.to_dict(orient="index"),
            "deep_model": None # Placeholder
        }
        
        # Save Unified JSON
        metrics_all_path = os.path.join(report_dir, "metrics_all.json")
        # Convert numpy types in baselines dict
        metrics_all_safe = make_serializable(metrics_all)
        
        with open(metrics_all_path, "w") as f:
            json.dump(metrics_all_safe, f, indent=4)
        logger.info(f"Unified metrics saved to {metrics_all_path}")
        
    except Exception as e:
        logger.error(f"Failed to run baselines for {ticker}: {e}")
        # Continue execution implies returning gracefully
        return

def select_top_tickers(n):
    """Scans baseline reports and selects top N tickers based on Rolling Mean RMSE."""
    logger.info(f"Selecting Top {n} Tickers based on Rolling Mean RMSE...")
    
    report_root = os.path.join(config.PROJECT_ROOT, "reports")
    if not os.path.exists(report_root):
        logger.error("No reports directory found. Run baselines first.")
        return []

    ticker_stats = []
    
    for ticker in os.listdir(report_root):
        # Use metrics_all.json if available, else fallback to CSV
        metrics_all_path = os.path.join(report_root, ticker, "metrics_all.json")
        baseline_path = os.path.join(report_root, ticker, "baselines.csv")
        
        try:
            rmse = float("inf")
            if os.path.exists(metrics_all_path):
                with open(metrics_all_path, "r") as f:
                    data = json.load(f)
                    if "baselines" in data and "Rolling_Mean" in data["baselines"]:
                        rmse = data["baselines"]["Rolling_Mean"]["RMSE"]
                        ticker_stats.append((ticker, rmse))
            elif os.path.exists(baseline_path):
                df = pd.read_csv(baseline_path, index_col=0)
                if "Rolling_Mean" in df.index and "RMSE" in df.columns:
                    rmse = float(df.loc["Rolling_Mean", "RMSE"])
                    ticker_stats.append((ticker, rmse))
        except Exception as e:
            logger.warning(f"Error reading baselines for {ticker}: {e}")
    
    # Sort by RMSE (Lower is better)
    ticker_stats.sort(key=lambda x: x[1])
    
    selected = [t[0] for t in ticker_stats[:n]]
    
    # Save selection
    out_path = os.path.join(report_root, "selected_tickers.txt")
    with open(out_path, "w") as f:
        for t in selected:
            f.write(f"{t}\n")
            
    logger.info(f"Selected {len(selected)} tickers. Saved to {out_path}")
    return selected

def run_deep_training(ticker):
    """Runs deep model training for a selected ticker."""
    logger.info(f"Starting Deep Training for: {ticker}")
    
    report_dir = os.path.join(config.PROJECT_ROOT, "reports", ticker)
    os.makedirs(report_dir, exist_ok=True)
    
    # Reproducibility
    config.set_seed(config.RANDOM_SEED)

    # Data Loading
    loader = UnifiedDataLoader(ticker, market_type="auto") 
    try:
        train_loader, val_loader, test_loader, stats = loader.get_data_loaders()
    except Exception as e:
        logger.error(f"Failed to load data for {ticker}: {e}")
        return

    # VALIDITY CHECK: Check against Rolling Mean
    # Load Unified Metrics if available
    metrics_all_path = os.path.join(report_dir, "metrics_all.json")
    metrics_all = {}
    rolling_rmse = float("inf")
    
    if os.path.exists(metrics_all_path):
        with open(metrics_all_path, "r") as f:
            metrics_all = json.load(f)
            if "baselines" in metrics_all and "Rolling_Mean" in metrics_all["baselines"]:
                 rolling_rmse = metrics_all["baselines"]["Rolling_Mean"]["RMSE"]
    # Fallback to CSV
    elif os.path.exists(os.path.join(report_dir, "baselines.csv")):
        df = pd.read_csv(os.path.join(report_dir, "baselines.csv"), index_col=0)
        if "Rolling_Mean" in df.index:
            rolling_rmse = float(df.loc["Rolling_Mean", "RMSE"])
    
    # Model Setup
    model = HybridModel(
        num_features=len(config.FEATURE_COLS), 
        seq_len=config.MAX_SEQ_LEN
    )
    
    # Training
    trainer = Trainer(model, train_loader, val_loader, scaler=stats.get('scaler', None))
    best_val_loss, best_auc = trainer.fit(epochs=config.EPOCHS)
    
    # Save Best AUC Model (Specialized Checkpoint)
    if trainer.best_model_auc_state is not None:
         model_auc_path = os.path.join(config.ARTIFACTS_DIR, f"{ticker}_model_best_auc.pth")
         torch.save(trainer.best_model_auc_state, model_auc_path)
         logger.info(f"Saved Best AUC Model (AUC: {best_auc:.4f}) to {model_auc_path}")
    
    # Final Evaluation
    logger.info("Running Final Evaluation...")
    trainer.model.eval()
    all_ret_pred, all_ret_true = [], []
    all_dir_pred, all_dir_true = [], []
    all_reg_pred, all_reg_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(trainer.device)
            y_ret = batch['target_return'].to(trainer.device)
            y_dir = batch['target_direction'].to(trainer.device)
            y_reg = batch['target_regime'].to(trainer.device)
            
            pred_ret, pred_dir, pred_reg = trainer.model(x)
            
            all_ret_pred.append(pred_ret.cpu())
            all_ret_true.append(y_ret.cpu())
            all_dir_pred.append(pred_dir.cpu())
            all_dir_true.append(y_dir.cpu())
            all_reg_pred.append(pred_reg.cpu())
            all_reg_true.append(y_reg.cpu())

    ret_pred = torch.cat(all_ret_pred)
    ret_true = torch.cat(all_ret_true)
    dir_pred = torch.cat(all_dir_pred)
    dir_true = torch.cat(all_dir_true)
    reg_pred = torch.cat(all_reg_pred)
    reg_true = torch.cat(all_reg_true)
    
    reg_true = torch.cat(all_reg_true)
    
    # Standardize RMSE: Unscale Targets/Preds
    scaler = stats.get('scaler', None)
    if scaler:
         idx = config.FEATURE_COLS.index('log_returns')
         mu = scaler.mean_[idx]
         sigma = scaler.scale_[idx]
         
         # Unscale P10, P50, P90
         ret_pred_unscaled = (ret_pred.cpu() * sigma) + mu
         ret_true_unscaled = (ret_true.cpu() * sigma) + mu
         
         # Use Unscaled for Metrics (RMSE, MAE, etc.)
         metrics = calculate_metrics(ret_pred_unscaled, ret_true_unscaled, dir_pred, dir_true, reg_pred, reg_true)
         
         # For plotting, use unscaled
         ret_pred_plot = ret_pred_unscaled
         ret_true_plot = ret_true_unscaled
    else:
         metrics = calculate_metrics(ret_pred.cpu(), ret_true.cpu(), dir_pred, dir_true, reg_pred, reg_true)
         ret_pred_plot = ret_pred.cpu()
         ret_true_plot = ret_true.cpu()
    
    # VALIDITY CHECK DECISION
    # ---------------------------------------------------------
    # Criteria 1: RMSE Check (Standard: Beating the naive baseline)
    model_rmse = metrics['RMSE']
    
    # Criteria 2: AUC Check (Strategic: Beating random chance)
    # We prioritize Validation AUC because Test Set might be too small/masked
    val_auc = best_auc 
    test_auc = metrics.get('Dir_AUC', 0.5)
    
    # Handle NaNs
    if np.isnan(test_auc): test_auc = 0.5
    
    rolling_rmse = float("inf") if rolling_rmse is None else rolling_rmse
    
    rmse_pass = (model_rmse <= rolling_rmse)
    
    # Check Valid AUC first, then Test AUC as fallback confirmation
    auc_pass = (val_auc > 0.55) or (test_auc > 0.55)
    
    if auc_pass:
        source = "Validation" if val_auc > 0.55 else "Test"
        val = val_auc if val_auc > 0.55 else test_auc
        logger.info(f"Model Accepted: Strong Directional Signal ({source} AUC {val:.4f} > 0.55)")
        valid_model = True
    elif rmse_pass:
        logger.info(f"Model Accepted: Low RMSE ({model_rmse:.4f} <= Baseline {rolling_rmse:.4f})")
        valid_model = True
    else:
        logger.warning(f"Model Discarded: RMSE ({model_rmse:.4f} > {rolling_rmse:.4f}) and AUC (Val={val_auc:.4f}, Test={test_auc:.4f}) insufficient.")
        valid_model = False
        
    if valid_model:
        model_path = os.path.join(config.ARTIFACTS_DIR, f"{ticker}_model.pth")
        torch.save(model.state_dict(), model_path)
        
    # Update Unified Metrics
    metrics['valid_model'] = valid_model
    metrics['best_val_auc'] = val_auc # Log Validation AUC
    metrics['best_auc'] = val_auc # Legacy field compatibility
    metrics['checkpoint'] = f"{ticker}_model.pth" if valid_model else None
    metrics['checkpoint_auc'] = f"{ticker}_model_best_auc.pth" if trainer.best_model_auc_state else None
    
    metrics_all['deep_model'] = metrics
    metrics_all['ticker'] = ticker # Ensure ticker is present
    
    # Save Unified JSON (Safe Serialization)
    metrics_all_safe = make_serializable(metrics_all)
    with open(metrics_all_path, "w") as f:
        json.dump(metrics_all_safe, f, indent=4)
        
    logger.info(f"Unified Metrics saved to {metrics_all_path}")

    # Generate Plots (Only if valid?) - Plot generation is harmless/diagnostic, keep it.
    y_true_np = ret_true.numpy().flatten()
    y_pred_np = ret_pred[:, 1].numpy().flatten()
    # ... (Plots code remains same)

    # Generate Plots (Only if valid?) - Plot generation is harmless/diagnostic, keep it.
    y_true_np = ret_true_plot.numpy().flatten()
    y_pred_np = ret_pred_plot[:, 1].numpy().flatten()
    
    # Plot 1: Prediction vs Actual
    plt.figure(figsize=(12, 6))
    subset = 200
    if len(y_true_np) > subset:
        plt.plot(y_true_np[-subset:], label='Actual', alpha=0.7)
        plt.plot(y_pred_np[-subset:], label='Predicted (P50)', alpha=0.7)
        plt.title(f"{ticker}: Prediction vs Actual (Last {subset} steps)")
    else:
        plt.plot(y_true_np, label='Actual', alpha=0.7)
        plt.plot(y_pred_np, label='Predicted (P50)', alpha=0.7)
        plt.title(f"{ticker}: Prediction vs Actual")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, "prediction_vs_actual.png"))
    plt.close()
    
    # Plot 2: Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_np, y_pred_np, alpha=0.3)
    min_val = min(np.min(y_true_np), np.min(y_pred_np))
    max_val = max(np.max(y_true_np), np.max(y_pred_np))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Log Returns')
    plt.ylabel('Predicted Log Returns')
    plt.title(f'{ticker}: Accuracy Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, "accuracy_scatter.png"))
    plt.close()

def main(args):
    # Determine Ticker List
    tickers = []
    
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers_file:
        if os.path.exists(args.tickers_file):
            with open(args.tickers_file, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.all:
        tickers = UnifiedDataLoader.get_available_tickers()
    
    # Filter by Market if needed
    
    # Mapped Flags for compatibility
    run_baselines = args.baselines_only or args.baseline
    
    # PHASE 1: Baselines
    if run_baselines or args.train_top:
        logger.info(f"Phase 1: Running Baselines for {len(tickers)} tickers...")
        for t in tickers:
            run_baselines_only(t)
            
    # PHASE 2 & 3: Selection & Training
    if args.train_top:
        n = int(args.train_top)
        selected_tickers = select_top_tickers(n)
        
        logger.info(f"Phase 3: Deep Training for Top {len(selected_tickers)} Tickers...")
        for t in selected_tickers:
            # GATE: Check trainability
            is_trainable, reason = is_trainable_ticker(t)
            if not is_trainable:
                logger.info(f"Skipping Deep Training for {t}: {reason}")
                continue
                
            run_deep_training(t)
            
    # Fallback: Single Ticker Full Run (or manual ALL run without top-N)
    elif args.ticker and not run_baselines:
        run_baselines_only(tickers[0])
        run_deep_training(tickers[0])
        
    elif args.all and not run_baselines:
         logger.warning("Running Full Training on ALL tickers is not recommended without --train_top. Proceeding...")
         for t in tickers:
             run_baselines_only(t)
             run_deep_training(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("--all", action="store_true", help="Run for ALL available tickers")
    parser.add_argument("--baselines_only", action="store_true", help="Run ONLY baselines phase")
    parser.add_argument("--train_top", type=int, help="Select top N tickers by RMSE and train deep model")
    parser.add_argument("--tickers_file", type=str, help="Path to text file with list of tickers")
    parser.add_argument("--baseline", action="store_true", help="Legacy flag (mapped to baselines_only)") 
    
    args = parser.parse_args()
    
    main(args)
