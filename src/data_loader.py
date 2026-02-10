import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
from . import config

class FinancialDataset(Dataset):
    def __init__(self, X, y_return, y_direction, y_regime):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_return = torch.tensor(y_return, dtype=torch.float32).unsqueeze(1)
        self.y_direction = torch.tensor(y_direction, dtype=torch.float32).unsqueeze(1)
        self.y_regime = torch.tensor(y_regime, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'target_return': self.y_return[idx],
            'target_direction': self.y_direction[idx],
            'target_regime': self.y_regime[idx]
        }

class UnifiedDataLoader:
    def __init__(self, ticker, market_type="auto"):
        """
        Args:
            ticker: Stock symbol (e.g., 'RELIANCE', 'AAPL')
            market_type: 'nse' or 'nasdaq'
        """
        self.ticker = ticker.upper()
        self.market_type = market_type.lower()
        self.data = None
        self.train_stats = {} # Stores mean/std/thresholds from training set
        self.scaler = StandardScaler()
        

    def _load_raw_data(self):
        """Loads and standardizes raw CSV data."""
        paths_to_check = []
        
        if self.market_type == "auto":
            # Add all possible locations
            paths_to_check.append(("nse", os.path.join(config.NSE_DIR, f"{self.ticker}.csv")))
            paths_to_check.append(("nasdaq", os.path.join(config.NASDAQ_DIR, "stocks", f"{self.ticker}.csv")))
            paths_to_check.append(("nasdaq", os.path.join(config.NASDAQ_DIR, "etfs", f"{self.ticker}.csv")))
        elif self.market_type == "nse":
            paths_to_check.append(("nse", os.path.join(config.NSE_DIR, f"{self.ticker}.csv")))
        elif self.market_type == "nasdaq":
             paths_to_check.append(("nasdaq", os.path.join(config.NASDAQ_DIR, "stocks", f"{self.ticker}.csv")))
             paths_to_check.append(("nasdaq", os.path.join(config.NASDAQ_DIR, "etfs", f"{self.ticker}.csv")))
        else:
            raise ValueError("Invalid market type")

        found_path = None
        for m_type, p in paths_to_check:
            if os.path.exists(p):
                found_path = p
                self.market_type = m_type # Update to actual found type
                break
        
        if not found_path:
             raise FileNotFoundError(f"Data for {self.ticker} not found in checked locations.")
             
        path = found_path
        
        # Load CSV
        try:
             df = pd.read_csv(path)
        except Exception as e:
             raise ValueError(f"Failed to read CSV for {self.ticker}: {e}")
        
        # Standardize Columns
        # NSE: Date, Open, High, Low, Close, Adj Close, Volume (Check actual, usually Date,Open,High,Low,Close,Shares Traded, Turnover)
        # NASDAQ: Date, Open, High, Low, Close, Adj Close, Volume
        
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        
        # Handle NSE specific column names often found (e.g. 'series', 'turnover')
        # We need: date, open, high, low, close, volume. Use adj_close if available.
        
        required_cols = {"date", "open", "high", "low", "volume"}
        if not required_cols.issubset(df.columns) and not ("close" in df.columns or "adj_close" in df.columns):
             raise ValueError(f"Missing required columns in {self.ticker}")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Prefer Adj Close, fallback to Close
        if 'adj_close' in df.columns:
            df['price'] = df['adj_close']
        elif 'close' in df.columns:
             df['price'] = df['close']
        else:
            raise ValueError("No price column found")
        # Data Integrity: Drop invalid rows immediately
        # Price <= 0 causes log errors; Volume <= 0 is suspicious
        df = df[(df['price'] > 0) & (df['volume'] > 0)].copy()
        
        # Drop initial NaNs if any (though usually cleaner in raw data)
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError(f"No valid data for {self.ticker} after cleaning (Price > 0 check)")
            
        return df[['date', 'open', 'high', 'low', 'price', 'volume']].copy()

    def _compute_indicators(self, df):
        """Computes technical indicators matching config.FEATURE_COLS.
           Includes hardening against infinite/NaN values.
        """
        # 1. Returns
        df['returns'] = df['price'].pct_change()
        
        # Safe Log Return
        price_ratio = df['price'] / df['price'].shift(1)
        price_ratio = price_ratio.where(price_ratio > 0) 
        df['log_returns'] = np.log(price_ratio)
        
        # 2. Volatility
        df['volatility_20'] = df['log_returns'].rolling(window=20).std()
        # Also keep Rolling_Vol_20 for label generation (legacy support if needed)
        df['Rolling_Vol_20'] = df['volatility_20'] 
        
        # 3. RSI (14)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (12, 26, 9)
        exp1 = df['price'].ewm(span=12, adjust=False).mean()
        exp2 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 5. Price Location / Shape Features
        # Normalized relative to recent range or simply OHLC relationships
        # simple: (X - MA20) / MA20
        ma20 = df['price'].rolling(window=20).mean()
        df['close_loc'] = (df['price'] - ma20) / ma20
        df['high_loc'] = (df['high'] - ma20) / ma20
        df['low_loc'] = (df['low'] - ma20) / ma20
        
        # Fill entries
        df.fillna(0, inplace=True) # Naive fill for early nans to ensure existence
        
        # ---------------------------------------------------------
        # SANITIZATION
        # ---------------------------------------------------------
        # 1. Replace infs with NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Drop rows with NaNs (caused by rolling windows, log of invalid, etc.)
        df.dropna(inplace=True)
        
        # 3. Data Sufficiency Check (Global)
        if len(df) < config.MIN_ROWS:
             raise ValueError(f"Insufficient data after cleaning: {len(df)} rows < {config.MIN_ROWS}")
             
        return df

    def _compute_direction_labels(self, prices, volatility, horizon, neutral_scale):
        """Computes direction labels based on RAW prices and volatility."""
        # Calculate Forward Returns for Direction
        # Fwd_Ret_K = (Close[t+K] / Close[t]) - 1
        
        # Shift prices to get future price
        future_prices = np.roll(prices, -horizon)
        
        # Vectorized forward return calculation
        # Note: The last 'horizon' elements will be invalid wrap-arounds
        with np.errstate(divide='ignore', invalid='ignore'):
            fwd_returns = (future_prices / prices) - 1
        
        # Dynamic Threshold calculation
        thresholds = neutral_scale * volatility
        
        # Generate Labels
        direction_labels = np.full(len(prices), -1, dtype=int)
        
        # Up trend
        direction_labels[fwd_returns > thresholds] = 1
        
        # Down trend
        direction_labels[fwd_returns < -thresholds] = 0
        
        # The last 'horizon' labels are invalid 
        direction_labels[-horizon:] = -1
        
        return direction_labels

    def _create_sequences(self, data, features, seq_len=config.MAX_SEQ_LEN):
        """Creates sliding window sequences (X) and next-step targets (y)."""
        X, y_ret, y_dir, y_reg = [], [], [], []
        
        # Data is already scaled and labelled
        data_values = data[features].values
        returns = data['log_returns'].values
        regimes = data['regime_label'].values
        
        # Use precomputed labels from dataframe
        direction_labels = data['direction_label'].values
        
        num_samples = len(data) - seq_len
        
        if num_samples <= 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        for i in range(num_samples):
            # Input window
            X.append(data_values[i : i + seq_len])
            
            # Target (Next Step)
            # 1. Return: Log return of the *next* day (i+seq_len)
            if i + seq_len < len(returns):
                y_ret.append(returns[i + seq_len])
                y_reg.append(regimes[i + seq_len]) 
                
                # 2. Direction: Trend starting from the *end* of the input window
                # The input window ends at index (i + seq_len - 1).
                # We want the direction label calculated at that timestamp.
                y_dir.append(direction_labels[i + seq_len - 1])
            else:
                break
            
        return np.array(X), np.array(y_ret), np.array(y_dir), np.array(y_reg)

    def get_data_loaders(self):
        """Main pipeline execution."""
        
        # 1. Load Clean Data
        df = self._load_raw_data()
        self.df = df # Persist for access by run_experiment.py
        
        # 2. Engineer Features
        df = self._compute_indicators(df)
        
        # 3. Chronological Split indices
        n = len(df)
        train_end = int(n * config.TRAIN_SPLIT_RATIO)
        val_end = int(n * (config.TRAIN_SPLIT_RATIO + config.VAL_SPLIT_RATIO))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # --- Regime labels should be based on RAW volatility (before scaling) ---
        vol_col = 'volatility_20'
        train_vol_raw = train_df[vol_col].values
        th1 = np.percentile(train_vol_raw, config.VOL_PERCENTILES[0])
        th2 = np.percentile(train_vol_raw, config.VOL_PERCENTILES[1])
        
        self.train_stats['vol_thresholds'] = (th1, th2)
        
        def assign_regime_raw(v):
            if v <= th1: return 0
            elif v <= th2: return 1
            else: return 2
            
        train_df['regime_label'] = train_df[vol_col].apply(assign_regime_raw)
        val_df['regime_label'] = val_df[vol_col].apply(assign_regime_raw)
        test_df['regime_label'] = test_df[vol_col].apply(assign_regime_raw)
        
        # --- Direction labels must use RAW volatility + RAW prices ---
        h = config.DIRECTION_HORIZON
        ns = config.NEUTRAL_ZONE_SCALE
        
        train_df['direction_label'] = self._compute_direction_labels(
            train_df['price'].values, train_df['volatility_20'].values, h, ns
        )
        val_df['direction_label'] = self._compute_direction_labels(
            val_df['price'].values, val_df['volatility_20'].values, h, ns
        )
        test_df['direction_label'] = self._compute_direction_labels(
            test_df['price'].values, test_df['volatility_20'].values, h, ns
        )
        
        # 4. Fit Scaler ONLY on Training Features
        self.scaler.fit(train_df[config.FEATURE_COLS])
        self.train_stats['scaler'] = self.scaler # Save provided scaler
        
        train_df[config.FEATURE_COLS] = self.scaler.transform(train_df[config.FEATURE_COLS])
        val_df[config.FEATURE_COLS] = self.scaler.transform(val_df[config.FEATURE_COLS])
        test_df[config.FEATURE_COLS] = self.scaler.transform(test_df[config.FEATURE_COLS])
        
        # 5. Create Sequences (Labels already in DF)
        X_train, y_ret_train, y_dir_train, y_reg_train = self._create_sequences(train_df, config.FEATURE_COLS)
        X_val, y_ret_val, y_dir_val, y_reg_val = self._create_sequences(val_df, config.FEATURE_COLS)
        X_test, y_ret_test, y_dir_test, y_reg_test = self._create_sequences(test_df, config.FEATURE_COLS)
        
        X_test, y_ret_test, y_dir_test, y_reg_test = self._create_sequences(test_df, config.FEATURE_COLS)
        
        # 6. Create Datasets & Loaders
        train_ds = FinancialDataset(X_train, y_ret_train, y_dir_train, y_reg_train)
        val_ds = FinancialDataset(X_val, y_ret_val, y_dir_val, y_reg_val)
        test_ds = FinancialDataset(X_test, y_ret_test, y_dir_test, y_reg_test)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader, self.train_stats
    
    def check_drift(self, recent_data):
        """
        Checks for distribution drift in recent data relative to training statistics.
        recent_data: DataFrame or array of recent features (raw/unscaled preferred? No, we have scaled stats)
        """
        # This is a placeholder for the inference logic
        pass

    @staticmethod
    @staticmethod
    def get_available_tickers():
        """Scans data directories and returns a list of available tickers (NSE first)."""
        nse_tickers = set()
        nasdaq_tickers = set()
        
        # NSE
        if os.path.exists(config.NSE_DIR):
            for f in glob.glob(os.path.join(config.NSE_DIR, "*.csv")):
                nse_tickers.add(os.path.basename(f).replace(".csv", "").upper())
        
        # NASDAQ
        if os.path.exists(config.NASDAQ_DIR):
            for subdir in ["stocks", "etfs"]:
                path = os.path.join(config.NASDAQ_DIR, subdir)
                if os.path.exists(path):
                    for f in glob.glob(os.path.join(path, "*.csv")):
                        nasdaq_tickers.add(os.path.basename(f).replace(".csv", "").upper())
        
        # Prioritize NSE: Return sorted NSE list + sorted NASDAQ list (excluding duplicates already in NSE)
        sorted_nse = sorted(list(nse_tickers))
        sorted_nasdaq = sorted(list(nasdaq_tickers - nse_tickers))
        
        return sorted_nse + sorted_nasdaq

