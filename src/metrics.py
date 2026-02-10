import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, r2_score, matthews_corrcoef, roc_auc_score, mean_absolute_error

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

def calculate_metrics(ret_pred, ret_true, dir_pred, dir_true, reg_pred, reg_true):
    """
    Calculates metrics for Return (Regression), Direction (Binary Class), Regime (Multi-class).
    """
    metrics = {}
    
    # Ensure CPU for sklearn
    ret_pred = ret_pred.detach().cpu()
    ret_true = ret_true.detach().cpu()
    dir_pred = dir_pred.detach().cpu()
    dir_true = dir_true.detach().cpu()
    reg_pred = reg_pred.detach().cpu()
    reg_true = reg_true.detach().cpu()
    
    # ---------------------------------------------------------
    # 1. Return Metrics (P10, P50, P90) -> Using P50 for Point Estimate
    # ---------------------------------------------------------
    # ret_pred shape: (N, 3) -> [P10, P50, P90]
    y_pred = ret_pred[:, 1].numpy().flatten()
    y_true = ret_true.numpy().flatten()
    
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # ---------------------------------------------------------
    # 2. Direction Metrics (Binary Classification)
    # ---------------------------------------------------------
    # dir_true has -1 for Neutral. We mask those out.
    dir_mask = (dir_true != -1)
    
    if dir_mask.sum() > 0:
        d_true = dir_true[dir_mask].numpy().flatten()
        d_prob = torch.sigmoid(dir_pred[dir_mask]).numpy().flatten()
        
        n_up = (d_true == 1).sum()
        n_down = (d_true == 0).sum()
        n_valid = len(d_true)
        
        metrics['n_valid'] = n_valid
        metrics['n_up'] = n_up
        metrics['n_down'] = n_down
        
        # Enforce data sufficiency to ensure meaningful metrics
        if n_valid < 20 or n_up < 2 or n_down < 2:
            metrics['Dir_BalAcc'] = np.nan
            metrics['Dir_MCC'] = np.nan
            metrics['Dir_AUC'] = np.nan
            metrics['Best_Threshold'] = 0.5
            
            # Simple Accuracy is still definable
            d_pred_simple = (d_prob > 0.5).astype(int)
            metrics['Dir_Acc'] = np.mean(d_true == d_pred_simple)
        else:
            # Metrics with Optimal Threshold
            # AUC
            try:
                 metrics['Dir_AUC'] = roc_auc_score(d_true, d_prob)
                 
                 # Find Optimal Threshold (Maximize J = TPR - FPR)
                 from sklearn.metrics import roc_curve
                 fpr, tpr, thresholds = roc_curve(d_true, d_prob)
                 j_scores = tpr - fpr
                 best_idx = np.argmax(j_scores)
                 best_threshold = thresholds[best_idx]
                 metrics['Best_Threshold'] = best_threshold
                 
                 # Re-calculate Accuracy with Best Threshold
                 d_pred_opt = (d_prob > best_threshold).astype(int)
                 
                 metrics['Dir_BalAcc'] = balanced_accuracy_score(d_true, d_pred_opt)
                 metrics['Dir_MCC'] = matthews_corrcoef(d_true, d_pred_opt)
                 metrics['Dir_Acc'] = np.mean(d_true == d_pred_opt)
                 
            except ValueError:
                 # Likely single class present despite checks
                 metrics['Dir_AUC'] = np.nan
                 metrics['Best_Threshold'] = 0.5
                 metrics['Dir_BalAcc'] = np.nan
                 metrics['Dir_MCC'] = np.nan
                 metrics['Dir_Acc'] = np.mean(d_true == (d_prob > 0.5).astype(int))
    else:
        metrics['Dir_BalAcc'] = np.nan
        metrics['Dir_MCC'] = np.nan
        metrics['Dir_AUC'] = np.nan
        metrics['Dir_Acc'] = 0.0
        metrics['Best_Threshold'] = 0.5
        
    # 3. Regime Metrics (Multi-class)
    r_true = reg_true.numpy().flatten()
    r_pred = torch.argmax(reg_pred, dim=1).numpy().flatten()
    
    metrics['Reg_BalAcc'] = balanced_accuracy_score(r_true, r_pred)
    
    return metrics
