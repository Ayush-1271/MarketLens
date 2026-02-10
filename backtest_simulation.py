import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.data_loader import UnifiedDataLoader
from src.model import HybridModel

def run_backtest(ticker, threshold=0.55):
    """
    Simulates a simple trading strategy using the trained model.
    Strategy:
        - Long if P(Up) > threshold
        - Short if P(Up) < (1 - threshold)
        - Cash otherwise
    """
    print(f"--- Backtesting {ticker} ---")
    
    # Paths
    model_path = os.path.join(config.ARTIFACTS_DIR, f"{ticker}_model_best_auc.pth")
    # Fallback to standard model if AUC one doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(config.ARTIFACTS_DIR, f"{ticker}_model.pth")
        
    if not os.path.exists(model_path):
        print(f"No model found for {ticker}")
        return None

    # Load Data (Test Set)
    print(f"DEBUG: Initializing UnifiedDataLoader for {ticker} with market_type='auto'")
    loader = UnifiedDataLoader(ticker, market_type="auto")
    print(f"DEBUG: loader.market_type = {loader.market_type}")
    print(f"DEBUG: config.NASDAQ_DIR = {config.NASDAQ_DIR}")
    train_loader, val_loader, test_loader, stats = loader.get_data_loaders()
    
    # Load Model
    model = HybridModel(num_features=len(config.FEATURE_COLS))
    state = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    
    # Collect Predictions & Returns
    all_returns = []
    all_probs = []
    all_dates = [] 
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(config.DEVICE)
            y_ret = batch['target_return'].cpu().numpy().flatten()
            
            # Forward
            _, pred_dir, _ = model(x)
            prob_up = torch.sigmoid(pred_dir).cpu().numpy().flatten()
            
            all_returns.extend(y_ret)
            all_probs.extend(prob_up)
            
    # Simulation
    returns = np.array(all_returns)
    probs = np.array(all_probs)
    
    signals = np.zeros_like(probs)
    signals[probs > threshold] = 1.0   # Long
    signals[probs < (1 - threshold)] = -1.0 # Short
    
    strat_returns = signals * returns
    
    # Cumulative
    cum_market = np.cumsum(returns)
    cum_strat = np.cumsum(strat_returns)
    
    # Metrics
    win_rate = np.mean(np.sign(strat_returns[signals != 0]) > 0)
    total_return = cum_strat[-1]
    sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252/5) # Annualized approx
    
    print(f"  Total Return: {total_return:.4f} (Market: {cum_market[-1]:.4f})")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Sharpe: {sharpe:.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cum_market, label='Market (Hold)', alpha=0.6)
    plt.plot(cum_strat, label=f'Strategy (Thresh {threshold})', alpha=0.8)
    plt.title(f"Backtest: {ticker} (Sharpe: {sharpe:.2f})")
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    report_dir = os.path.join(config.PROJECT_ROOT, "reports", ticker)
    os.makedirs(report_dir, exist_ok=True)
    plot_path = os.path.join(report_dir, "backtest_equity.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Plot saved to {plot_path}")
    
    return {
        "ticker": ticker,
        "total_return": total_return,
        "market_return": cum_market[-1],
        "win_rate": win_rate,
        "sharpe": sharpe
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default="reports/selected_tickers.txt")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    
    tickers = []
    if os.path.exists(args.tickers):
        with open(args.tickers, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
            
    results = []
    for t in tickers[:10]: # Test top 10 for speed
        res = run_backtest(t, args.threshold)
        if res:
            results.append(res)
            
    # Summary
    if results:
        df = pd.DataFrame(results)
        print("\n--- Summary ---")
        print(df.describe())
        df.to_csv("backtest_results.csv")
