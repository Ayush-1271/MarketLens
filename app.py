import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from src import config
from src.data_loader import UnifiedDataLoader
from src.model import HybridModel

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="MarketLens", layout="wide")
st.title("🛡️ MarketLens")
st.markdown("**Regime-Aware Market Analysis** | Uncertainty Diagnostics | Decision Support")

# ---------------------------------------------------------
# CACHED FUNCTIONS
# ---------------------------------------------------------
@st.cache_resource
def load_system(ticker, use_best_auc=False):
    """
    Loads data, stats, and model for a specific ticker.
    Cached to ensure low latency interaction.
    """
    # 1. Load Data & Stats
    loader = UnifiedDataLoader(ticker, market_type="auto") # Auto-detect market
    try:
        # We re-run the split logic to get the exact same objects as training
        train_loader, val_loader, test_loader, stats = loader.get_data_loaders()
    except Exception as e:
        return None, None, None, f"Error loading data: {str(e)}", None

    # 3. Load Metrics (Unified) - Load FIRST to decide which checkpoint to use
    import json
    report_dir = os.path.join(config.PROJECT_ROOT, "reports", ticker)
    metrics_path = os.path.join(report_dir, "metrics_all.json")
    
    metrics_all = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_all = json.load(f)

    # 2. Load Model
    model = HybridModel(num_features=len(config.FEATURE_COLS), seq_len=config.MAX_SEQ_LEN)
    model.eval()
    
    # Determine Checkpoint Path
    weights_path = None
    status = "⚠️ Untrained Model"
    
    if metrics_all and "deep_model" in metrics_all:
        dm = metrics_all["deep_model"]
        if dm: # Ensure deep_model is not None
             if use_best_auc and dm.get("checkpoint_auc"):
                 weights_path = os.path.join(config.ARTIFACTS_DIR, dm["checkpoint_auc"])
                 status = f"Loaded BEST AUC Model (AUC: {dm.get('best_auc', 'N/A'):.3f})"
             elif dm.get("checkpoint"):
                 weights_path = os.path.join(config.ARTIFACTS_DIR, dm["checkpoint"])
                 status = "Loaded Best Loss Model"
    
    # Fallback to standard name if metrics missing but file exists
    if not weights_path:
         standard_path = os.path.join(config.ARTIFACTS_DIR, f"{ticker}_model.pth")
         if os.path.exists(standard_path):
             weights_path = standard_path
             status = "Loaded Standard Model (No Metrics Found)"

    if weights_path and os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        except RuntimeError as e:
            # Likely feature-count mismatch from old checkpoints
            status = f"⚠️ Checkpoint incompatible (likely old features). Please retrain. Error: {str(e).splitlines()[0]}"
    else:
        status = "⚠️ Untrained Model (Random Weights)"
        
    return loader, model, stats, status, metrics_all

def perform_inference(model, feature_window):
    """
    Runs single-instance inference on CPU.
    Returns prediction and latency.
    """
    start_time = time.time()
    
    # feature_window: (T, F) numpy array
    x_tensor = torch.tensor(feature_window, dtype=torch.float32).unsqueeze(0) # (1, T, F)
    
    with torch.no_grad():
        ret_pred, dir_pred, reg_pred = model(x_tensor)
        
    latency_ms = (time.time() - start_time) * 1000
    return ret_pred, dir_pred, reg_pred, latency_ms

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
# helper to find tickers with TRAINED models
def get_trained_tickers():
    """
    Identifies tickers with valid trained models.
    Checks for both metrics report AND physical model file.
    """
    trained_tickers = []
    
    # 1. Scan Artifacts for Model Files
    # This is more robust than checking reports, as reports might exist for baselines only
    if not os.path.exists(config.ARTIFACTS_DIR):
        return []
        
    # Find all files ending in _model.pth or _model_best_auc.pth
    # extract ticker from filename
    files = os.listdir(config.ARTIFACTS_DIR)
    
    model_files = [f for f in files if f.endswith("_model.pth") or f.endswith("_model_best_auc.pth")]
    
    for f in model_files:
        # Ticker is the part before _model...
        if "_model_best_auc.pth" in f:
            ticker = f.replace("_model_best_auc.pth", "")
        else:
            ticker = f.replace("_model.pth", "")
            
        if ticker not in trained_tickers:
            trained_tickers.append(ticker)
            
    return sorted(trained_tickers)

sidebar = st.sidebar

# Sidebar Selection
if st.sidebar.button("🔄 Refresh Tickers"):
    st.rerun()

available_tickers = get_trained_tickers()

if not available_tickers:
    st.error("⚠️ No trained models found! Please run 'python run_experiment.py --train_top 50'.")
    st.stop()
    
# Default to RELIANCE if available
default_idx = available_tickers.index("RELIANCE") if "RELIANCE" in available_tickers else 0
selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers, index=default_idx)

# Model Selection Controls
st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
use_best_auc = st.sidebar.checkbox("Use Best AUC Model", value=True, help="Load the checkpoint optimized for Directional Accuracy instead of RMSE.")
neutral_tolerance = st.sidebar.slider("Neutral Tolerance", 0.0, 0.4, 0.05, 0.01, help="Probability margin around the threshold to declare 'Neutral'.")

# Store in session state
if 'loaded_ticker' not in st.session_state:
    st.session_state['loaded_ticker'] = None

if sidebar.button("Load Ticker"):
    st.session_state.loaded_ticker = selected_ticker

if st.session_state.loaded_ticker:
    ticker = st.session_state.loaded_ticker
    with st.spinner(f"Loading System for {ticker}..."):
        # Unpack 5 values now
        loader, model, stats, status, metrics_all = load_system(ticker, use_best_auc=use_best_auc)
        
    if loader is None:
        st.error(status)
    else:
        st.success(f"System Ready: {status}")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Forecast & Drift", "Diagnostics", "Performance & Metrics"])
        
        # Load Raw Data for Viz
        raw_df = loader._load_raw_data()
        df_analyzed = loader._compute_indicators(raw_df.copy())
        df_analyzed.reset_index(drop=True, inplace=True)
        

        with tab1:
            st.header(f"{ticker} Market Overview")
            st.markdown(f"**Description**: Historical price and volume data for {ticker}. The candlesticks show the Open, High, Low, and Close prices for each day.")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=raw_df['date'], open=raw_df['open'], high=raw_df['high'], 
                                         low=raw_df['low'], close=raw_df['price'], name="OHLC"), row=1, col=1)
            fig.add_trace(go.Bar(x=raw_df['date'], y=raw_df['volume'], name="Volume"), row=2, col=1)
            st.plotly_chart(fig, width="stretch")
            
        with tab2:
            st.header("Probabilistic Forecast")
            st.markdown("""
            **How to read this:**
            *   **P50 (Median)**: The model's "best guess" for the return.
            *   **Confidence Band**: The range (P10 to P90) where the model is 80% sure the actual return will fall. 
            *   **Direction**: Uses the *Optimal Threshold* from validation to decide Up/Down, with a configurable "Neutral" buffer.
            """)
            
            # Select a date for inference
            dates = df_analyzed['date'].dt.date
            # Only allow selecting from dates where we have enough history
            valid_dates = dates.iloc[config.MAX_SEQ_LEN:].values
            selected_date = st.select_slider("Select Date for Prediction", options=valid_dates, value=valid_dates[-1])
            
            # Find index
            idx = df_analyzed[df_analyzed['date'].dt.date == selected_date].index[0]
            
            # Get window 
            feature_slice = df_analyzed.iloc[idx-config.MAX_SEQ_LEN : idx][config.FEATURE_COLS]
            # Scale
            feature_slice_scaled = stats['scaler'].transform(feature_slice)
            
            # Run Inference
            ret_pred, dir_pred, reg_pred_logits, lat = perform_inference(model, feature_slice_scaled)
            
            st.metric("Inference Latency", f"{lat:.2f} ms", help="Time taken for the CPU to generate this prediction. Must be <100ms.")
            if lat > 100:
                st.error("Latency Constraint Violated (>100ms)")
            
            # Parse Outputs
            q10, q50, q90 = ret_pred[0].numpy()
            direction_prob = torch.sigmoid(dir_pred[0]).item()
            regime_class = torch.argmax(reg_pred_logits[0]).item()
            regime_name = config.REGIME_CLASSES[regime_class]
            
            # Metrics for Interpretation
            best_thresh = 0.5
            if metrics_all and "deep_model" in metrics_all:
                best_thresh = metrics_all["deep_model"].get("Best_Threshold", 0.5)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Return Prediction")
                st.write(f"P50 (Median): **{q50:.4f}**")
                st.write(f"Uncertainty (P90-P10): {q90-q10:.4f}")
                st.info(f"Band: [{q10:.4f}, {q90:.4f}]")
            
            with col2:
                st.subheader("5-Day Trend")
                st.write(f"Probability UP: **{direction_prob:.2%}**")
                st.caption(f"Threshold: {best_thresh:.2f} ± {neutral_tolerance:.2f}")
                
                # Neutral Zone Logic
                upper = best_thresh + neutral_tolerance
                lower = best_thresh - neutral_tolerance
                
                if direction_prob > upper:
                    st.success("BULLISH Trend")
                elif direction_prob < lower:
                    st.error("BEARISH Trend")
                else:
                    st.warning("NEUTRAL / UNCERTAIN")
                    
            with col3:
                st.subheader("Regime")
                st.write(f"Detected: **{regime_name}**")
                st.caption(f"Market Condition: {regime_name}")
                
            # Drift Warning
            vol_idx = config.FEATURE_COLS.index("volatility_20")
            current_vol_z = feature_slice_scaled[-1, vol_idx]
            
            if np.abs(current_vol_z) > 3.0:
                 st.warning(f"⚠️ DRIFT DETECTED: Volatility Z-Score {current_vol_z:.2f} > 3.0. Prediction Low Confidence.")
            
        with tab3:
            st.header("Diagnostics")
            st.markdown("""
            **What is this?**
            Diagnostics help you trust the model by showing where it fails.
            *   **Residuals**: The difference between the Actual Return and the Predicted Return. Ideally, these should be close to zero and randomly scattered.
            *   **Stress Test**: How the model would have performed during the 2020 Market Crash.
            """)
            
            # 1. Stress Test: COVID-19
            st.subheader("⚠️ Stress Test: COVID-19 Crash (March 2020)")
            
            # hardcoded date range for COVID
            covid_start = pd.Timestamp("2020-02-01")
            covid_end = pd.Timestamp("2020-04-30")
            
            # Check if we have data for this period
            mask = (df_analyzed['date'] >= covid_start) & (df_analyzed['date'] <= covid_end)
            covid_df = df_analyzed.loc[mask]
            
            if covid_df.empty:
                st.warning("No data available for March 2020 (COVID-19 Period) for this ticker.")
            else:
                stress_X = []
                stress_dates = []
                experiment_indices = covid_df.index
                
                valid_indices = []
                for idx in experiment_indices:
                    if idx - config.MAX_SEQ_LEN >= 0:
                        qs = df_analyzed.iloc[idx-config.MAX_SEQ_LEN : idx][config.FEATURE_COLS]
                        qs_scaled = stats['scaler'].transform(qs)
                        stress_X.append(qs_scaled)
                        stress_dates.append(df_analyzed.loc[idx, 'date'])
                        valid_indices.append(idx)

                if not stress_X:
                     st.error("Not enough history before March 2020 to run inference.")
                else:
                    stress_X = np.array(stress_X) 
                    stress_tensor = torch.tensor(stress_X, dtype=torch.float32)
                    
                    with torch.no_grad():
                        s_ret, s_dir, s_reg = model(stress_tensor)
                        
                    s_p50 = s_ret[:, 1].numpy()
                    s_true = df_analyzed.loc[valid_indices, 'log_returns'].values
                    
                    # Plot Stress Test
                    fig_stress = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    
                    stress_prices = df_analyzed.loc[valid_indices, 'price']
                    fig_stress.add_trace(go.Scatter(x=stress_dates, y=stress_prices, name="Price", line=dict(color='gray', width=1)), row=1, col=1)
                    
                    fig_stress.add_trace(go.Scatter(x=stress_dates, y=s_true, name="Actual Return", line=dict(color='blue')), row=2, col=1)
                    fig_stress.add_trace(go.Scatter(x=stress_dates, y=s_p50, name="Predicted Return", line=dict(color='red', dash='dot')), row=2, col=1)
                    
                    fig_stress.update_layout(title="COVID-19 Crash Performance", height=500)
                    st.plotly_chart(fig_stress, width="stretch")
                    
                    stress_mse = np.mean((s_true - s_p50)**2)
                    st.write(f"Stress Period MSE: **{stress_mse:.5f}**")

            st.markdown("---")
            
            # 2. Residual Analysis (Last 1 Year)
            st.subheader("Residual Analysis (Last 1 Year)")
            if len(df_analyzed) > 365:
                recent_indices = df_analyzed.index[-365:]
            else:
                recent_indices = df_analyzed.index[config.MAX_SEQ_LEN:]
                
            rec_X = []
            rec_dates = []
            valid_rec = []
             
            for idx in recent_indices:
                if idx - config.MAX_SEQ_LEN >= 0:
                    qs = df_analyzed.iloc[idx-config.MAX_SEQ_LEN : idx][config.FEATURE_COLS]
                    rec_X.append(stats['scaler'].transform(qs))
                    rec_dates.append(df_analyzed.loc[idx, 'date'])
                    valid_rec.append(idx)
                    
            if rec_X:
                rec_tensor = torch.tensor(np.array(rec_X), dtype=torch.float32)
                with torch.no_grad():
                    r_ret, _, _ = model(rec_tensor)
                
                r_p50 = r_ret[:, 1].numpy()
                r_true = df_analyzed.loc[valid_rec, 'log_returns'].values
                residuals = r_true - r_p50
                
                # Plot Residuals
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=rec_dates, y=residuals, mode='markers', name="Residuals", marker=dict(color='orange', size=4)))
                fig_res.add_hline(y=0, line_dash="dash", line_color="white")
                fig_res.update_layout(title="Model Residuals (Errors) Over Time", yaxis_title="True - Predicted")
                st.plotly_chart(fig_res, width="stretch")
                
                # Distribution
                fig_dist = go.Figure(data=[go.Histogram(x=residuals, nbinsx=50, name='Residuals')])
                fig_dist.update_layout(title="Residual Distribution (Normality Check)")
                st.plotly_chart(fig_dist, width="stretch")
            
        with tab4:
             st.header("Model Performance & Metrics")
             
             if metrics_all:
                 # 1. Create Comparison DataFrame
                 data = []
                 
                 # Baselines
                 if "baselines" in metrics_all:
                     for model_name, m_dict in metrics_all["baselines"].items():
                         row = {"Model": model_name}
                         row.update(m_dict)
                         data.append(row)
                 
                 # Deep Model
                 if "deep_model" in metrics_all and metrics_all["deep_model"]:
                     dm = metrics_all["deep_model"]
                     row = {"Model": "Hybrid_Deep_Model"}
                     # Explicitly map keys if needed, or update
                     row.update(dm)
                     data.append(row)
                 
                 if data:
                     df_metrics = pd.DataFrame(data)
                     
                     # Reorder columns for clarity
                     cols = ["Model", "RMSE", "Dir_AUC", "Best_Threshold", "Dir_Acc", "Dir_BalAcc"]
                     # Filter existing stats only
                     existing_cols = [c for c in cols if c in df_metrics.columns]
                     # Add others if present
                     other_cols = [c for c in df_metrics.columns if c not in existing_cols and c != "valid_model" and c != "checkpoint" and c != "checkpoint_auc"]
                     
                     final_cols = existing_cols + other_cols
                     df_display = df_metrics[final_cols].copy()
                     
                     st.subheader("🏆 Model Leaderboard")
                     st.caption("Sorting by Directional Signal (AUC) and Regression Error (RMSE).")
                     
                     # Highlight best metrics
                     st.dataframe(df_display.style.highlight_min(subset=["RMSE"], color='lightgreen').highlight_max(subset=["Dir_AUC", "Dir_Acc"], color='lightblue'), use_container_width=True)
                     
                     # 2. Key Insights
                     st.subheader("Key Insights")
                     
                     # Find Deep Model and Rolling Mean
                     deep_row = df_metrics[df_metrics["Model"] == "Hybrid_Deep_Model"]
                     base_row = df_metrics[df_metrics["Model"] == "Rolling_Mean"]
                     
                     if not deep_row.empty:
                         d_auc = deep_row.iloc[0].get("Dir_AUC", 0.5)
                         d_rmse = deep_row.iloc[0]["RMSE"]
                         
                         col_i1, col_i2 = st.columns(2)
                         col_i1.metric("Deep Model AUC", f"{d_auc:.3f}")
                         col_i1.caption("Target > 0.55")
                         
                         if not base_row.empty:
                             b_rmse = base_row.iloc[0]["RMSE"]
                             impr = (1 - d_rmse/b_rmse) * 100
                             col_i2.metric("RMSE Improvement", f"{impr:.2f}%")
                         
                         if d_auc > 0.55:
                             st.success(f"✅ Strong Directional Signal (AUC {d_auc:.3f}). Model is detecting trends.")
                         elif d_auc < 0.50:
                             st.error(f"❌ Model is anti-correlated (AUC {d_auc:.3f}). Something is wrong.")
                         else:
                             st.warning(f"⚠️ Weak Signal (AUC {d_auc:.3f}). Model is guessing close to random.")
                             
                 else:
                     st.warning("Metrics file exists but contains no data.")
             else:
                 st.warning("No metrics_all.json found. Run experiments first.")

else:
    st.info("Please enter a ticker and load the system.")
