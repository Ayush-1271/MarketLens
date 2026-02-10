import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from . import config
from .metrics import QuantileLoss, calculate_metrics
from sklearn.metrics import mean_squared_error

class Trainer:
    def __init__(self, model, train_loader, val_loader, scaler=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.device = device if device else config.get_device()
        self.model.to(self.device)
        
        # Calculate Class Weighting (Pos Weight)
        print("Calculating class weights from training set...")
        n_up, n_down, n_neutral = 0, 0, 0
        
        # Iterate partially if huge, or full if small. Data is small (<10k).
        # We need to be careful not to consume the loader if it's an iterator, but DataLoader is reusable.
        for batch in train_loader:
             y = batch['target_direction']
             n_up += (y == 1).sum().item()
             n_down += (y == 0).sum().item()
             n_neutral += (y == -1).sum().item()
             
        # Handling edge cases
        if n_up > 0 and n_down > 0:
            ratio = n_down / n_up
            # Clip weight to prevent instability
            clipped_ratio = min(ratio, config.MAX_POS_WEIGHT)
            self.pos_weight = torch.tensor([clipped_ratio]).to(self.device)
            print(f"  -> Found {n_up} Up, {n_down} Down. Ratio={ratio:.2f}. Setting pos_weight={clipped_ratio:.2f} (Capped)")
        else:
            self.pos_weight = None
            print("  -> Warning: Could not calculate class weights (missing classes). Defaulting to 1.0")
            
        # Losses
        self.criterion_ret = QuantileLoss()
        self.criterion_dir = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) 
        self.criterion_reg = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4) # L2 Regularization
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Checkpointing
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.best_auc = 0.0
        self.best_model_auc_state = None
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_ret_loss = 0
        total_dir_loss = 0
        total_reg_loss = 0
        
        for batch in self.train_loader:
            x = batch['features'].to(self.device)
            y_ret = batch['target_return'].to(self.device)
            y_dir = batch['target_direction'].to(self.device)
            y_reg = batch['target_regime'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_ret, pred_dir, pred_reg = self.model(x)
            
            # 1. Return Loss (Quantile Regression)
            loss_ret = self.criterion_ret(pred_ret, y_ret)

            # 2. Direction Loss (Binary Classification)
            # Mask out neutral samples (-1)
            # y_dir shape: (batch_size, 1)
            dir_mask = (y_dir != -1).flatten()
            
            if dir_mask.sum() > 0:
                # Select only valid samples
                p_dir_valid = pred_dir[dir_mask]
                y_dir_valid = y_dir[dir_mask]
                loss_dir = self.criterion_dir(p_dir_valid, y_dir_valid)
            else:
                loss_dir = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 3. Regime Loss (Multi-class Classification)
            loss_reg = self.criterion_reg(pred_reg, y_reg)
            
            # Total Loss
            loss = (config.RETURN_LOSS_WEIGHT * loss_ret) + \
                   (config.DIRECTION_LOSS_WEIGHT * loss_dir) + \
                   (config.REGIME_LOSS_WEIGHT * loss_reg)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_ret_loss += loss_ret.item()
            total_dir_loss += loss_dir.item()
            total_reg_loss += loss_reg.item()
            
        # Logit Diagnostics (Last Batch)
        with torch.no_grad():
             logits_mean = pred_dir.mean().item()
             logits_std = pred_dir.std().item()
             
        return {
            'loss': total_loss / len(self.train_loader),
            'ret_loss': total_ret_loss / len(self.train_loader),
            'dir_loss': total_dir_loss / len(self.train_loader),
            'reg_loss': total_reg_loss / len(self.train_loader),
            'logits_mean': logits_mean,
            'logits_std': logits_std
        }

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_ret_pred, all_ret_true = [], []
        all_dir_pred, all_dir_true = [], []
        all_reg_pred, all_reg_true = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['features'].to(self.device)
                y_ret = batch['target_return'].to(self.device)
                y_dir = batch['target_direction'].to(self.device)
                y_reg = batch['target_regime'].to(self.device)
                
                pred_ret, pred_dir, pred_reg = self.model(x)
                
                loss_ret = self.criterion_ret(pred_ret, y_ret)
                # Primary Metric for Rollback is Return Loss
                total_loss += loss_ret.item() 
                
                all_ret_pred.append(pred_ret)
                all_ret_true.append(y_ret)
                all_dir_pred.append(pred_dir)
                all_dir_true.append(y_dir)
                all_reg_pred.append(pred_reg)
                all_reg_true.append(y_reg)
                
        metrics = calculate_metrics(
            torch.cat(all_ret_pred), torch.cat(all_ret_true),
            torch.cat(all_dir_pred), torch.cat(all_dir_true),
            torch.cat(all_reg_pred), torch.cat(all_reg_true)
        )
        metrics['val_ret_loss'] = total_loss / len(self.val_loader)
        
        # Calculate Unscaled RMSE for Human Readability
        if self.scaler:
             # P50 Preds (Index 1)
             y_pred_scaled = torch.cat(all_ret_pred)[:, 1].cpu().numpy().flatten()
             y_true_scaled = torch.cat(all_ret_true).cpu().numpy().flatten()
             
             # Unscale
             from . import config
             idx = config.FEATURE_COLS.index('log_returns')
             mu = self.scaler.mean_[idx]
             sigma = self.scaler.scale_[idx]
             
             y_pred_unscaled = (y_pred_scaled * sigma) + mu
             y_true_unscaled = (y_true_scaled * sigma) + mu
             
             mse_unscaled = mean_squared_error(y_true_unscaled, y_pred_unscaled)
             metrics['RMSE_Unscaled'] = np.sqrt(mse_unscaled)
        else:
             metrics['RMSE_Unscaled'] = metrics['RMSE'] # Fallback to scaled
        
        # Diagnostics Logging
        # Re-calculate binary preds for diagnostics (using same logic as metrics.py)
        d_true_all = torch.cat(all_dir_true).cpu().numpy().flatten()
        d_pred_logits = torch.cat(all_dir_pred).cpu().numpy().flatten()
        
        # Mask Neutral
        valid_mask = (d_true_all != -1)
        n_valid = valid_mask.sum()
        
        if n_valid > 0:
            d_true_valid = d_true_all[valid_mask]
            d_prob_valid = 1 / (1 + np.exp(-d_pred_logits[valid_mask])) # Sigmoid
            d_pred_class = (d_prob_valid > 0.5).astype(int)
            
            n_up = (d_true_valid == 1).sum()
            n_down = (d_true_valid == 0).sum()
            
            # Confusion Matrix
            # TP: True=1, Pred=1
            tp = ((d_true_valid == 1) & (d_pred_class == 1)).sum()
            tn = ((d_true_valid == 0) & (d_pred_class == 0)).sum()
            fp = ((d_true_valid == 0) & (d_pred_class == 1)).sum()
            fn = ((d_true_valid == 1) & (d_pred_class == 0)).sum()
            
            print(f"  [Diagnostics] Val Samples: {n_valid} (Up: {n_up}, Down: {n_down})")
            print(f"  [Diagnostics] Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
        else:
            print("  [Diagnostics] No valid direction samples in validation set.")

        return metrics

    def fit(self, epochs=config.EPOCHS):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            current_val_loss = val_metrics['val_ret_loss']
            self.scheduler.step(current_val_loss)
            
            dir_bal_acc = val_metrics['Dir_BalAcc']
            dir_acc_str = f"{dir_bal_acc:.4f}" if not np.isnan(dir_bal_acc) else "N/A"
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Ret Loss: {current_val_loss:.4f} | "
                  f"Val RMSE (Unscaled): {val_metrics.get('RMSE_Unscaled', val_metrics['RMSE']):.4f} | "
                  f"Val Dir BalAcc: {dir_acc_str} (AUC: {val_metrics.get('Dir_AUC', 0.5):.2f}) | "
                  f"Thresh: {val_metrics.get('Best_Threshold', 0.5):.2f} | "
                  f"Logits: {train_metrics['logits_mean']:.2f} +/- {train_metrics['logits_std']:.2f}")
            
            # Rollback / Early Stopping Logic
            # Goal: Minimize Val Return Loss.
            # If Aux tasks cause degradation, we rely on the fact that we save the model 
            # ONLY when Val Return Loss improves.
            
            # Primary Checkpoint: Best Return Loss (Conservative)
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                print("  -> New Best Loss Model Saved!")
            else:
                self.patience_counter += 1
                
            # Secondary Checkpoint: Best Directional AUC (Targeted)
            current_auc = val_metrics.get('Dir_AUC', 0)
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.best_model_auc_state = copy.deepcopy(self.model.state_dict())
                print(f"  -> New Best AUC Model Saved! (AUC: {current_auc:.4f})")
                
            if self.patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break
                
        # Restore best model (Rollback) - Default to Best Loss model for safety
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best model state (Rollback to best Return Loss).")
            
        return self.best_val_loss, self.best_auc
 