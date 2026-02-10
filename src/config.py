import os
import torch
import numpy as np
import random

# -----------------------------------------------------------------------------
# REPRODUCIBILITY & SYSTEM CONSTRAINTS
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
MAX_SEQ_LEN = 128            # Constraint: T <= 128
MAX_MODEL_PARAMS = 2_000_000 # Constraint: ~2M parameters
LATENCY_BUDGET_MS = 100      # Constraint: < 100ms inference per ticker

def get_device():
    """Detects best available hardware (CUDA > DirectML > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for DirectML (AMD/Intel GPU/NPU on Windows)
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
        
    return torch.device("cpu")

DEVICE = get_device() # Auto-detect and Assign

def set_seed(seed=RANDOM_SEED):
    """Enforce strict reproducibility settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# DATA PATHS (Auto-Detect)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

NSE_DIR = os.path.join(DATA_DIR, "nse")
NASDAQ_DIR = os.path.join(DATA_DIR, "nasdaq")

# Ensure directories exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# MODEL HYPERPARAMETERS (Constraint Compliant)
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50 
PATIENCE = 5 # Early stopping

# Feature Definition
FEATURE_COLS = [
    'returns', 'log_returns', 'volatility_20', 
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'close_loc', 'high_loc', 'low_loc' 
]

# -----------------------------------------------------------------------------
# LOSS WEIGHTS & OBJECTIVES
# -----------------------------------------------------------------------------
# Primary Task: Quantile Regression (Risk/Return)
# Primary Task: Quantile Regression (Risk/Return)
RETURN_LOSS_WEIGHT = 1.0 

# Auxiliary Task: Direction Prediction (Classification)
# Increased weight to stabilize training as per plan
DIRECTION_LOSS_WEIGHT = 0.5 

# Auxiliary Task: Regime Identification (Weak Supervision)
REGIME_LOSS_WEIGHT = 0.1 

# Direction Training Settings
DIRECTION_HORIZON = 5 # Predict 5-day cumulative return direction
NEUTRAL_ZONE_SCALE = 0.3 # Threshold = 0.3 * Volatility (Avoids noise)
MAX_POS_WEIGHT = 3.0 # Cap for dynamic class weighting

# -----------------------------------------------------------------------------
# REGIME DEFINITIONS
# -----------------------------------------------------------------------------
REGIME_CLASSES = ["Low_Vol", "Medium_Vol", "High_Vol"]
# Fixed thresholds for volatility discretization (can be tuned based on training data distribution)
VOL_PERCENTILES = [33, 66] 

# -----------------------------------------------------------------------------
# DATA SUFFICIENCY GATES
# -----------------------------------------------------------------------------
MIN_ROWS = 600
MIN_TRAIN_SAMPLES = 300
MIN_VAL_SAMPLES = 80
MIN_DIR_VALID = 20
MIN_DIR_PER_CLASS = 5 



MIN_DIR_VALID_STRICT = 100 # Relaxed from 200 to allow more tickers (e.g. PJT)
MIN_DIR_PER_CLASS_STRICT = 20 # Relaxed from 30

# -----------------------------------------------------------------------------
# DATA SPLIT RATIOS
# -----------------------------------------------------------------------------
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
