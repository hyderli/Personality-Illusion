"""
Plot ROC curves for sycophancy probing across both models and probing points.

Generates:
1. Qwen vs Llama prompt-level probing comparison
2. Qwen prompt vs </think> comparison (if mid-gen data available)
3. Linear vs MLP comparison

Usage:
    python plot_roc_curves.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from pathlib import Path


def load_and_prepare(model_key, layer=14):
    """Load activations and prepare within-dilemma centered held-out split."""
    data = np.load(f'reasoning_extension/results/probing/{model_key}_activations.npz', allow_pickle=True)
    labels = data['labels']
    items = pd.read_csv(f'reasoning_extension/results/behavioral_tasks/{model_key}_sycophancy_items.csv')
    valid = items[(~items.step1_parse_failed) & (~items.step2_parse_failed)].reset_index(drop=True)
    d_ids = valid.dilemma_id.values

    # Train/test split by dilemma
    unique_d = np.unique(d_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_d)
    split = int(0.7 * len(unique_d))
    train_d = set(unique_d[:split])
    test_d = set(unique_d[split:])

    train_mask = np.array([d in train_d for d in d_ids])
    test_mask = np.array([d in test_d for d in d_ids])

    X = data[f'layer_{layer}'].astype(np.float32)

    # Within-dilemma centering
    X_tr = X[train_mask].copy()
    X_te = X[test_mask].copy()
    d_tr = d_ids[train_mask]
    d_te = d_ids[test_mask]

    for d_id in np.unique(d_tr):
        mask = d_tr == d_id
        X_tr[mask] -= X_tr[mask].mean(axis=0)
    for d_id in np.unique(d_te):
        mask = d_te == d_id
        X_te[mask] -= X_te[mask].mean(axis=0)

    y_tr = labels[train_mask]
    y_te = labels[test_mask]

    return X_tr, X_te, y_tr, y_te


def load_midgen(model_key, layer=27):
    """Load mid-generation activations with held-out split."""
    data = np.load(f'reasoning_extension/results/mid_gen_probing/{model_key}_mid_gen_activations.npz')
    labels = data['labels']
    d_ids = data['dilemma_ids']

    unique_d = np.unique(d_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_d)
    split = int(0.7 * len(unique_d))
    train_d = set(unique_d[:split])
    test_d = set(unique_d[split:])

    train_mask = np.array([d in train_d for d in d_ids])
    test_mask = np.array([d in test_d for d in d_ids])

    results = {}
    for source in ['prompt', 'think_end']:
        key = f'{source}_layer_{layer}'
        X = data[key].astype(np.float32)

        X_tr = X[train_mask].copy()
        X_te = X[test_mask].copy()
        d_tr = d_ids[train_mask]
        d_te = d_ids[test_mask]

        for d_id in np.unique(d_tr):
            mask = d_tr == d_id
            X_tr[mask] -= X_tr[mask].mean(axis=0)
        for d_id in np.unique(d_te):
            mask = d_te == d_id
            X_te[mask] -= X_te[mask].mean(axis=0)

        results[source] = (X_tr, X_te)

    y_tr = labels[train_mask]
    y_te = labels[test_mask]

    return results, y_tr, y_te


def train_and_get_roc(X_tr, X_te, y_tr, y_te, model_type='mlp'):
    """Train probe and return ROC curve data."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if model_type == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                           random_state=42, early_stopping=True, validation_fraction=0.15)
    else:
        clf = LogisticRegression(C=1.0, class_weight='balanced',
                                max_iter=1000, random_state=42)

    clf.fit(X_tr_s, y_tr)
    y_prob = clf.predict_proba(X_te_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# =========================================================================
# Plot 1: Both models, prompt-level, linear vs MLP
# =========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Panel A: Qwen vs Llama, MLP, prompt-level ---
ax = axes[0]
colors = {'qwen': '#2196F3', 'llama': '#FF5722'}

for model_key, label, color in [
    ('deepseek-r1-distill-qwen-7b', 'Qwen-7B', colors['qwen']),
    ('deepseek-r1-distill-llama-8b', 'Llama-8B', colors['llama']),
]:
    try:
        X_tr, X_te, y_tr, y_te = load_and_prepare(model_key, layer=14)
        fpr, tpr, roc_auc = train_and_get_roc(X_tr, X_te, y_tr, y_te, 'mlp')
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} MLP (AUC={roc_auc:.3f})')

        fpr_l, tpr_l, auc_l = train_and_get_roc(X_tr, X_te, y_tr, y_te, 'linear')
        ax.plot(fpr_l, tpr_l, color=color, lw=1.5, linestyle='--',
                label=f'{label} Linear (AUC={auc_l:.3f})')
    except Exception as e:
        print(f'  Skipping {model_key}: {e}')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('A) Prompt-Level Probing\n(Held-Out Dilemmas, Layer 14)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# --- Panel B: Qwen prompt vs </think>, MLP ---
ax = axes[1]
try:
    midgen_data, y_tr, y_te = load_midgen('deepseek-r1-distill-qwen-7b', layer=27)

    X_tr_p, X_te_p = midgen_data['prompt']
    fpr_p, tpr_p, auc_p = train_and_get_roc(X_tr_p, X_te_p, y_tr, y_te, 'mlp')
    ax.plot(fpr_p, tpr_p, color='#9E9E9E', lw=2,
            label=f'Prompt (AUC={auc_p:.3f})')

    X_tr_t, X_te_t = midgen_data['think_end']
    fpr_t, tpr_t, auc_t = train_and_get_roc(X_tr_t, X_te_t, y_tr, y_te, 'mlp')
    ax.plot(fpr_t, tpr_t, color='#4CAF50', lw=2,
            label=f'</think> (AUC={auc_t:.3f})')
except Exception as e:
    print(f'  Skipping mid-gen: {e}')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('B) Prompt vs </think> Probing\n(Qwen-7B, MLP, Layer 27)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

# --- Panel C: Layer comparison, Qwen, MLP ---
ax = axes[2]
layer_colors = {7: '#FFC107', 14: '#2196F3', 21: '#9C27B0', 27: '#F44336'}
for layer in [7, 14, 21, 27]:
    try:
        X_tr, X_te, y_tr, y_te = load_and_prepare('deepseek-r1-distill-qwen-7b', layer=layer)
        fpr, tpr, roc_auc = train_and_get_roc(X_tr, X_te, y_tr, y_te, 'mlp')
        ax.plot(fpr, tpr, color=layer_colors[layer], lw=2,
                label=f'Layer {layer} (AUC={roc_auc:.3f})')
    except Exception as e:
        print(f'  Skipping layer {layer}: {e}')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('C) Layer Comparison\n(Qwen-7B, MLP, Held-Out)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/roc_curves_sycophancy_probing.png', dpi=150, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/roc_curves_sycophancy_probing.pdf', bbox_inches='tight')
print('Saved roc_curves_sycophancy_probing.png and .pdf')
