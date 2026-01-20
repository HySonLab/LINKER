import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)

# ======================================================
# Load metadata
# ======================================================
protein_names = [os.path.split(x)[1].split('.')[0] for x in glob('evaluation/best_logits/*')]

# ======================================================
# Containers for micro-averaging
# ======================================================
y_true_all     = []
y_score_my_all = []

n = len(protein_names)

# ======================================================
# Iterate over proteins
# ======================================================
for idx, protein_name in enumerate(protein_names):
    print(f"{protein_name} | progress: {(idx / n) * 100:.2f}%")

    try:
        data = torch.load(
            f"evaluation/best_logits/{protein_name}.pt",  # TODO: fill path
            weights_only=True
        )
        logits = data["logits"]
        label = data["label"]
        mask = data["mask"]
    except Exception:
        print(f"Skip {protein_name}")
        continue

    # --------------------------------------------------
    # Shape normalization
    # logits: [residue, FG, 1] -> [R, F]
    # label : [residue, FG, 1] -> [R, F]
    # mask  : [residue, FG]
    # --------------------------------------------------
    logits = logits.permute(1, 0, 2).squeeze(-1)   # [R, F]
    label = label.permute(1, 0, 2).squeeze(-1)     # [R, F]
    mask = mask.permute(1, 0).bool()               # [R, F]

    length = int(torch.sum(mask[:, 0]).item())

    # --------------------------------------------------
    # Ground-truth per residue
    # residue = 1 if at least one FG is true
    # --------------------------------------------------
    trim_gt = torch.zeros(length)

    label_sum = label.sum(dim=2).sum(dim=1)  # sum over FG & interaction types
    indices = torch.nonzero(label_sum, as_tuple=True)[0]
    trim_gt[indices] = 1

    y_true = trim_gt  # [R_valid]

    # --------------------------------------------------
    # Model scores
    # sigmoid -> mask -> aggregate over FG
    # --------------------------------------------------
    proba = torch.sigmoid(logits)
    proba = proba * mask.unsqueeze(-1).float()  # [R, F, 7]

    mask_dup = mask.unsqueeze(-1).expand(-1, -1, 7)
    valid_proba = proba.clone()
    valid_proba[~mask_dup] = 0

    # Aggregate: FG max -> interaction max
    residue_scores = (
        valid_proba
        .max(dim=1).values      # over FG
        .max(dim=1).values      # over interaction types
    )

    my_scores = residue_scores[:length]

    # --------------------------------------------------
    # Collect
    # --------------------------------------------------
    y_true_all.append(y_true.cpu().numpy())
    y_score_my_all.append(my_scores.cpu().numpy())

# ======================================================
# Save raw logits (per protein)
# ======================================================
results = {
    "y_true_all": y_true_all,
    "y_score_my_all": y_score_my_all,
}

# with open("evaluation/score.pkl", "wb") as f:
#     pickle.dump(results, f)

# ======================================================
# Micro-averaging over all residues
# ======================================================
y_true_all = np.concatenate(y_true_all)
y_score_my_all = np.concatenate(y_score_my_all)

# ======================================================
# Precision–Recall / Average Precision
# ======================================================
prec_my, rec_my, _ = precision_recall_curve(
    y_true_all,
    y_score_my_all
)
ap_my = average_precision_score(
    y_true_all,
    y_score_my_all
)

# ======================================================
# ROC / AUC
# ======================================================
fpr_my, tpr_my, _ = roc_curve(
    y_true_all,
    y_score_my_all
)
roc_auc_my = auc(fpr_my, tpr_my)

print(f"MY -> AP: {ap_my:.4f} | ROC-AUC: {roc_auc_my:.4f}")

# ======================================================
# Save evaluation results
# ======================================================
results = {
    "y_true_all": y_true_all,
    "rec_my": rec_my,
    "prec_my": prec_my,
    "ap_my": ap_my,
    "fpr_my": fpr_my,
    "tpr_my": tpr_my,
    "roc_auc_my": roc_auc_my,
}

# with open("evaluation/results.pkl", "wb") as f:
#     pickle.dump(results, f)

