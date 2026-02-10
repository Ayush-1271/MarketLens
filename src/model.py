import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        # Multi-scale kernels: 3, 5, 7
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        # Fix: ensure total out_channels matches request roughly
        rem = out_channels - 2 * (out_channels // 3)
        self.conv7 = nn.Conv1d(in_channels, rem, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection if needed
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        
        out = torch.cat([x3, x5, x7], dim=1)
        out = self.bn(F.relu(out))
        out = self.dropout(out)
        
        res = self.residual(x)
        return out + res

class HybridModel(nn.Module):
    def __init__(self, num_features, seq_len=config.MAX_SEQ_LEN, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        
        # 1. Multi-scale CNN Encoder
        # Input: (Batch, Time, Features) -> Permute to (Batch, Features, Time) for Conv1d
        self.cnn_encoder = MultiScaleCNN(num_features, d_model)
        
        # 2. Transformer Encoder (Shallow)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Heads
        # Flatten or Global Pooling? 
        # Plan says "Transformer Encoder (limited depth)". 
        # Typically we pool or take last token.
        
        self.flat_dim = d_model * seq_len
        
        # Regressor/Quantile Head
        # Outputs 3 values: q10, q50, q90
        # Use a small MLP
        self.return_head = nn.Sequential(
            nn.Linear(self.flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # P10, P50, P90
        )
        
        # Direction Head (Binary)
        self.direction_head = nn.Sequential(
            nn.Linear(self.flat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Logits for BCE
        )
        
        # Regime Head (Multi-class: 3 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(self.flat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Logits for CE
        )
        
    def forward(self, x):
        # x: (Batch, Time, Features)
        
        # 1. CNN
        x_cnn = x.permute(0, 2, 1) # (B, F, T)
        x_cnn = self.cnn_encoder(x_cnn) # (B, d_model, T)
        x_cnn = x_cnn.permute(0, 2, 1) # (B, T, d_model) for Transformer
        
        # 2. Transformer
        x_trans = self.transformer(x_cnn) # (B, T, d_model)
        
        # 3. Heads
        x_flat = x_trans.reshape(x_trans.size(0), -1) # (B, T*d_model)
        
        ret_pred = self.return_head(x_flat)
        dir_pred = self.direction_head(x_flat)
        reg_pred = self.regime_head(x_flat)
        
        return ret_pred, dir_pred, reg_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Latency & Parameter Check
    import time
    
    model = HybridModel(num_features=7)
    params = count_parameters(model)
    print(f"Total Parameters: {params}")
    
    if params > config.MAX_MODEL_PARAMS:
        print("WARNING: Model exceeds parameter budget!")
        
    # Latency Check
    dummy_input = torch.randn(1, config.MAX_SEQ_LEN, 7)
    model.eval()
    
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_input)
    avg_lat = (time.time() - start_time) * 10 # ms per 100 iters * 10? No.
    # Total time for 100 iters in seconds. 
    # Avg time per iter in seconds = Total / 100.
    # Avg time in ms = (Total / 100) * 1000 = Total * 10.
    
    print(f"Average Inference Latency (CPU): {avg_lat:.2f} ms")
    
    if avg_lat > config.LATENCY_BUDGET_MS:
        print("FAIL: Latency budget exceeded!")
    else:
        print("PASS: Latency test passed.")
