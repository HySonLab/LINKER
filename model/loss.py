# =========================
# Standard library imports
# =========================
import os
import sys
import pickle
from glob import glob

# =========================
# Third-party imports
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem

# =========================
# PyTorch imports
# =========================
import torch
import torch.nn.functional as F
import torch.nn.functional as fn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =========================
# PyTorch Geometric imports
# =========================
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.nn.dense import DenseGINConv

# =========================
# Project path setup
# =========================
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# =========================
# Project-specific imports
# =========================
from utils import *
from model.base import *
from model.modules import *
import random

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=1, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits, targets: [B,R,F,C]
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        loss = alpha_factor * modulating * ce_loss
        return loss  # shape [B,R,F,C]
    

class LatentAlignmentLoss(nn.Module):
    """
    InfoNCE + Uniformity loss trên latent vector.
    Inputs:
        - z: [B, D] latent vector đã normalize
        - binding_scores: [B, score_dim] dùng để tìm positive pair
    """
    def __init__(self, tau=0.1, uniform_weight=0.1, topk=5):
        super().__init__()
        self.tau = tau
        self.uniform_weight = uniform_weight
        self.topk = topk

    def find_positive_indices(self, scores: torch.Tensor):
        """
        Cho mỗi sample, chọn positive index trong top-k gần nhất.
        scores: [B, score_dim]
        """
        B = scores.size(0)
        pos_idx = []
        for i in range(B):
            dist = torch.norm(scores[i] - scores, dim=-1)  # [B]
            dist[i] = float("inf")  # bỏ chính nó
            # top-k index (gần nhất)
            topk_idx = torch.topk(dist, k=min(self.topk, B-1), largest=False).indices
            # random chọn 1 trong top-k
            j = random.choice(topk_idx.tolist())
            pos_idx.append(j)
        return pos_idx

    @staticmethod
    def uniformity_loss(z: torch.Tensor):
        dist_sq = torch.cdist(z, z, p=2) ** 2
        exp_term = torch.exp(-2 * dist_sq)
        return torch.log(exp_term.mean() + 1e-8)

    @staticmethod
    def infonce_loss(z: torch.Tensor, pos_indices, tau=0.1):
        sim_matrix = z @ z.T  # cosine sim (z đã normalize)
        B = z.size(0)
        loss = 0
        for i in range(B):
            numerator = torch.exp(sim_matrix[i, pos_indices[i]] / tau)
            denominator = torch.sum(torch.exp(sim_matrix[i] / tau))
            loss += -torch.log(numerator / (denominator + 1e-8))
        return loss / B

    def forward(self, z: torch.Tensor, binding_scores: torch.Tensor):
        z = F.normalize(z, dim=-1)
        pos_indices = self.find_positive_indices(binding_scores)
        L_info = self.infonce_loss(z, pos_indices, tau=self.tau)
        L_unif = self.uniformity_loss(z)
        return L_info + self.uniform_weight * L_unif