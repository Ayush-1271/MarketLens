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

st.warning("""
**Important:** This system provides a **risk-aware outlook**, not investment advice. 
It focuses on uncertainty, market conditions, and data reliability rather than exact price predictions.
""")

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

with st.sidebar.expander("How to Read This Dashboard"):
    st.markdown("""
    *   **Outlook ≠ Certainty**: Ranges (P10-P90) are more important than single numbers.
    *   **Confidence Matters**: 'Neutral' means the signal is too weak to trust.
    *   **Market Condition**: A volatile market makes all predictions less reliable.
    *   **Diagnostics**: Check the 'Stability' tab to see if the model is behaving correctly.
    """)

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
        tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "5-Day Market Outlook", "Model Stability & Diagnostics", "Model Quality Checks"])
        
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
            st.header("5-Day Market Outlook")
            st.markdown("""
            **What this shows:**
            This section estimates how the asset might behave over the next 5 trading days.
            *   **Expected Outcome**: The model's "best guess" (Median) for the return.
            *   **Likely Range**: In 8 out of 10 scenarios, the return is expected to fall within this band. Wider bands = Higher Uncertainty.
            *   **Trend Confidence**: We only flag a trend if the model is statistically confident. Otherwise, we stay Neutral.
            """)
            
            # Select a date for inference
            dates = df_analyzed['date'].dt.date
            # Only allow selecting from dates where we have enough history
            valid_dates = dates.iloc[config.MAX_SEQ_LEN:].values
            selected_date = st.select_slider("Select Date for Outlook", options=valid_dates, value=valid_dates[-1])
            
            # Find index
            idx = df_analyzed[df_analyzed['date'].dt.date == selected_date].index[0]
            
            # Get window 
            feature_slice = df_analyzed.iloc[idx-config.MAX_SEQ_LEN : idx][config.FEATURE_COLS]
            # Scale
            feature_slice_scaled = stats['scaler'].transform(feature_slice)
            
            # Run Inference
            ret_pred, dir_pred, reg_pred_logits, lat = perform_inference(model, feature_slice_scaled)
            
            st.metric("Analysis Latency", f"{lat:.2f} ms", help="Time taken for the CPU to analyze the data. This is NOT execution latency.")
            if lat > 100:
                st.caption("Note: Latency > 100ms. This is fine for analysis, but would be slow for HFT.")
            
            # Parse Outputs
            q10, q50, q90 = ret_pred[0].numpy()
            direction_prob = torch.sigmoid(dir_pred[0]).item()
            regime_class = torch.argmax(reg_pred_logits[0]).item()
            regime_name = config.REGIME_CLASSES[regime_class]
            
            # Metrics for Interpretation
            best_thresh = 0.5
            if metrics_all and "deep_model" in metrics_all:
                best_thresh = metrics_all["deep_model"].get("Best_Threshold", 0.5)
            
            # Trust Logic
            uncertainty = q90 - q10
            is_high_uncertainty = uncertainty > 0.05
            is_volatile = regime_name == "High_Vol"
            
            trust_level = "Reliable"
            trust_color = "green"
            trust_reason = "Conditions are calm and model is confident."
            
            if is_volatile:
                trust_level = "Do Not Trust"
                trust_color = "red"
                trust_reason = "Market is too volatile for accurate predictions."
            elif is_high_uncertainty:
                trust_level = "Uncertain"
                trust_color = "orange"
                trust_reason = "Model predicts a wide range of possible outcomes."
                
            st.subheader("🤖 What the Model says")
            st.info(f"**Verdict**: {trust_level} ({trust_reason})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 1. Best Guess")
                st.write(f"Expected Return: **{q50:.4f}**")
                st.caption("If you made this trade 100 times, this is the average outcome.")
                
            with col2:
                st.markdown("#### 2. Range")
                st.write(f"Between **{q10:.4f}** and **{q90:.4f}**")
                st.caption("In 8 out of 10 cases, the price lands here.")
            
            with col3:
                st.markdown("#### 3. Market State")
                
                # Simple mapping
                condition_map = {
                    "Low_Vol": "Calm",
                    "Medium_Vol": "Normal",
                    "High_Vol": "Unstable"
                }
                friendly_condition = condition_map.get(regime_name, regime_name)
                
                st.write(f"Condition: **{friendly_condition}**")
                if regime_name == "High_Vol":
                    st.error("⚠️ Unstable: Predictions are likely wrong.")
                else:
                    st.caption("Calm markets are easier to predict.")

            st.markdown("---")
            st.subheader("🚦 Trend Confidence")
            
            # Neutral Zone Logic
            upper = best_thresh + neutral_tolerance
            lower = best_thresh - neutral_tolerance
            
            if direction_prob > upper:
                st.success(f"**Bullish ({direction_prob:.0%} Confidence)**")
                st.write("Why? The model sees strong patterns usually followed by a price rise.")
            elif direction_prob < lower:
                st.error(f"**Bearish ({1-direction_prob:.0%} Confidence)**")
                st.write("Why? The patterns resemble past drops. Note: This doesn't guarantee a crash.")
            else:
                st.warning("**Neutral / Unclear**")
                st.write("Why? The signals are mixed. The model cannot confidently say Up or Down.")
                
            # Drift Warning
            vol_idx = config.FEATURE_COLS.index("volatility_20")
            current_vol_z = feature_slice_scaled[-1, vol_idx]
            
            if np.abs(current_vol_z) > 3.0:
                 st.error(f"⚠️ **Stability Check Failed**: Market is moving abnormally fast (Z-Score {current_vol_z:.2f}). Ignore this forecast.")
            
        with tab3:
            st.header("Model Stability & Diagnostics")
            st.markdown("""
            **Why this matters:**
            This section tests whether the model is behaving safely.
            *   **Behavior During Crisis**: We simulate the 2020 crash to ensure the model doesn't "panic" or produce impossible values.
            *   **Stability**: We check if the model's errors are random (good) or biased (bad).
            """)
            
            # 1. Stress Test: COVID-19
            st.subheader("🛡️ Behavior During Market Crises (Scenario: COVID-19)")
            st.write("Simulating model performance during the March 2020 crash...")
            
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
                    
                    fig_stress.update_layout(title="Did the model panic?", height=500)
                    st.plotly_chart(fig_stress, width="stretch")
                    
                    stress_mse = np.mean((s_true - s_p50)**2)
                    
                    st.info("""
                    **Professor's Note:**
                    *   **Did it panic?** Look at the red dotted line. Did it jump around wildly, or follow the blue line loosely?
                    *   **Realistic?** During a crash, NO model is perfect. We want to see it react, not freeze.
                    """)
                    st.caption(f"Error during this period (MSE): {stress_mse:.5f}")

            st.markdown("---")
            
            st.markdown("---")
            
            # 2. Residual Analysis (Last 1 Year)
            st.subheader("Stability Analysis (Residuals)")
            st.caption("Checking for systematic bias. Errors should be centered around zero.")
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
                fig_dist.update_layout(title="Are errors random? (Bell Curve check)")
                st.plotly_chart(fig_dist, width="stretch")
                
                st.info("""
                **Verdict:** 
                *   If the dots (residuals) are scattered randomly around the white line, the model is **Healthy**.
                *   If they follow a curve or line, the model is **Biased (Unreliable)**.
                """)
            
        with tab4:
             st.header("Model Quality Checks")
             st.markdown("These metrics are used internally to validate that the model is performing better than random chance and simple baselines.")
             
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
                         col_i1.metric("Predictive Power (AUC)", f"{d_auc:.2f}")
                         col_i1.caption("0.50 = Random Guessing. 0.55+ is Good.")
                         
                         if not base_row.empty:
                             b_rmse = base_row.iloc[0]["RMSE"]
                             impr = (1 - d_rmse/b_rmse) * 100
                             col_i2.metric("Error Reduction", f"{impr:.1f}%")
                             col_i2.caption("How much better than a simple average?")
                         
                         if d_auc > 0.55:
                             st.success(f"✅ **Good Model**: AUC ({d_auc:.2f}) is consistently better than a coin flip.")
                         elif d_auc < 0.50:
                             st.error(f"❌ **Bad Model**: AUC ({d_auc:.2f}) is worse than random. Do not use.")
                         else:
                             st.warning(f"⚠️ **Weak Model**: AUC ({d_auc:.2f}) is barely better than random.")
                             
                         st.markdown("""
                         **Professor's Explanation:**
                         *   **AUC (Area Under Curve)**: Measures how often the model gets the *direction* right. 0.50 is random (like a coin flip). Anything above 0.55 is hard to achieve in finance.
                         *   **RMSE (Error)**: Measures how far off the price predictions are. Lower is better. We compare it to a "Rolling Mean" (simple average) to see if the complex model is actually adding value.
                         """)
                              
                     else:
                         st.warning("Metrics file exists but contains no data.")
             else:
                 st.warning("No metrics_all.json found. Run experiments first.")

else:
    st.info("Please enter a ticker and load the system.")
