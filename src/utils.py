import os
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def validate_and_normalize_config(config, yaml_cfg=None):
    """
    Cast common config fields to the expected Python types (int/float/bool).
    Call this after applying YAML values and CLI overrides, before using config.
    """
    # cast ints
    int_fields = ['batch_size', 'num_epochs', 'num_workers', 'n_boot', 'seed',
                  'kmer_size', 'max_seq_length', 'num_layers', 'pretrain_num_classes']
    for k in int_fields:
        if hasattr(config, k):
            v = getattr(config, k)
            if v is not None:
                try:
                    setattr(config, k, int(v))
                except Exception:
                    raise ValueError(f"Config.{k} must be int-like (got {v!r})")

    # cast floats
    float_fields = ['learning_rate', 'weight_decay', 'lr_min', 'start_factor', 'end_factor', 'max_grad_norm', 'dropout_rate']
    for k in float_fields:
        if hasattr(config, k):
            v = getattr(config, k)
            if v is not None:
                try:
                    setattr(config, k, float(v))
                except Exception:
                    raise ValueError(f"Config.{k} must be float-like (got {v!r})")

    # bools from strings (e.g. "False" -> False)
    bool_fields = ['use_class_weights']
    for k in bool_fields:
        if hasattr(config, k):
            v = getattr(config, k)
            if isinstance(v, str):
                lower = v.strip().lower()
                if lower in ('true', '1', 'yes'):
                    setattr(config, k, True)
                elif lower in ('false', '0', 'no', ''):
                    setattr(config, k, False)
                else:
                    raise ValueError(f"Config.{k} must be boolean-like (got {v!r})")

    return config

def save_config(config_obj, output_folder, save_name):
    """
    Save config object to JSON. Works if config is a dataclass or a plain object.
    Uses default=str in json.dump to guard against non-serializable types.
    """
    config_path = os.path.join(output_folder, save_name)
    try:
        if is_dataclass(config_obj):
            config_dict = asdict(config_obj)
        else:
            config_dict = {k: v for k, v in vars(config_obj).items() if not k.startswith("_") and not callable(v)}
    except Exception:
        config_dict = {k: str(v) for k, v in vars(config_obj).items() if not k.startswith("_")}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    print(f"Training configuration saved to: {config_path}")

def safe_index_or_max(arr, idx):
    """
    Safely return arr[idx] if available and not NaN, otherwise return max(arr).
    This helper avoids index or empty-array errors when extracting 'best epoch' metrics.
    """
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.nan
    if idx is not None and 0 <= idx < arr.size and not np.isnan(arr[int(idx)]):
        return float(arr[int(idx)])
    try:
        return float(np.nanmax(arr))
    except ValueError:
        return np.nan

def expected_calibration_error(true_binary, prob_pos, n_bins=8, strategy='quantile'):
    """
    Compute ECE (expected calibration error).
    Supports two strategies for bin edges:
    - 'uniform' : equal-width bins on [0,1]
    - 'quantile' : quantile-based bins so each bin has ~equal number of samples
    """
    true_binary = np.asarray(true_binary)
    prob_pos = np.asarray(prob_pos)
    # ​​Return NaN for empty predictions (safeguard)​
    if prob_pos.size == 0:
        return float(np.nan)

    # Choose bin edges
    if strategy == 'quantile':
        try:
            bins = np.quantile(prob_pos, np.linspace(0.0, 1.0, n_bins + 1))
            # If quantile edges collapsed (e.g. constant probabilities), fallback to uniform bins
            if np.any(np.diff(bins) <= 0):
                bins = np.linspace(0.0, 1.0, n_bins + 1)
        except Exception:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # default: uniform / equal-width bins
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Map probabilities to bin ids (0..n_bins-1)
    binids = np.digitize(prob_pos, bins) - 1
    # Clip to valid range in case of edge values
    binids = np.clip(binids, 0, n_bins - 1)

    ece = 0.0
    total = len(prob_pos)
    for b in range(n_bins):
        mask = binids == b
        if mask.sum() > 0:
            acc = true_binary[mask].mean()
            conf = prob_pos[mask].mean()
            ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)

def bootstrap_ci(y_true, y_score, metric=None, n_bootstrap=None, seed=None):
    """
    Compute bootstrap 95% CI for a given metric on (y_true, y_score).
    Returns: (point_estimate, lower_95, upper_95, n_valid_boots)
    metric is one of 'auroc', 'auprc', 'brier'.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    # Return NaNs if no samples
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    # point estimate  
    try:
        if metric == 'auroc':
            pt = float(roc_auc_score(y_true, y_score))
        elif metric == 'auprc':
            pt = float(average_precision_score(y_true, y_score))
        elif metric == 'brier':
            pt = float(brier_score_loss(y_true, y_score))
        else:
            raise ValueError("metric must be 'auroc'|'auprc'|'brier'")
    except Exception:
        pt = np.nan

    boots = []
    n_boot = int(n_bootstrap)
    # Generate bootstrap samples with replacement
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)  # Uniform random sampling 
        yt = y_true[idx]; ys = y_score[idx]
        try:
            if metric == 'auroc':
                val = float(roc_auc_score(yt, ys))
            elif metric == 'auprc':
                val = float(average_precision_score(yt, ys))
            else:
                val = float(brier_score_loss(yt, ys))
        except Exception:
            val = np.nan
        boots.append(val)
    
    # Exclude bootstrap samples that resulted in NaN  
    boots = np.array(boots, dtype=float)
    boots = boots[~np.isnan(boots)]
    # Return NaNs if no valid bootstrap samples
    if boots.size == 0:
        return pt, np.nan, np.nan, 0
    # Compute 95% CI via percentile method
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)
    return pt, lower, upper, boots.size

def _align_histories(hist_list, key):
    """
    Pad variable-length history arrays to uniform length with NaNs.
    
    Args:
        hist_list: List of history dicts
        key: Key to extract from each history
    
    Returns:
        np.ndarray: NaN-padded 2D array (n_histories x max_length)
        Empty array (0,0) if input is empty
    """
    # Return empty 2D array when no histories
    if not hist_list:
        return np.empty((0, 0), dtype=float)
    
    arrays = []
    for h in hist_list:
        # Safely convert missing entries to empty 1-D arrays
        arr = np.array(h.get(key, []), dtype=float)
        arrays.append(arr)
    
    # Determine maximum length among arrays (0 if all arrays empty)
    max_len = max((a.shape[0] for a in arrays), default=0)

    # Create output array filled with NaN and copy per-row values
    out = np.full((len(arrays), max_len), np.nan, dtype=float)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out