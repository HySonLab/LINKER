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
from torch_geometric.nn import global_mean_pool
# =========================
# PyTorch Geometric imports
# =========================
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.nn.dense import DenseGINConv
from torch_geometric.nn import GCNConv

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

    
class FINGER_ID(nn.Module):
    def __init__(
        self,
        node_feat_dim = 5,           # Dimension of input node features
        num_fg_types=205,        # Total number of FG types
        fg_embed_dim=512,        # Embedding dim for static FG type
        pos_embed_dim=128,        # Positional embedding dim (per FG in molecule)
        hidden_dim=960,          # GNN hidden dimension
        max_tokens = 160):
        """
        FINGER_ID
        ------------------
        Integrates functional-group (FG) information with molecular graph representations.

        Main components:
            1. GNN over molecular graph (atom-level)
            2. Global graph pooling
            3. Static FG type embedding + positional embedding
            4. FG-specific node pooling
            5. Feature fusion and FG-level self-attention
        """
        super(FINGER_ID, self).__init__()

        # Step 1: GNN → node embeddings
        self.gnn1 = GCNConv(node_feat_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)

        # Step 2: global pool → global_emb
        self.global_pool = global_mean_pool

        # Step 3: static FG embedding + positional FG embedding
        self.static_fg_embedding = nn.Embedding(num_fg_types, fg_embed_dim)
        self.pos_embedding = nn.Embedding(max_tokens, pos_embed_dim)  # up to 64 FG per mol
        # Linear layer for node pooled FG embedding
        self.fg_node_pool_proj = nn.Linear(hidden_dim, hidden_dim)

        # Step 5: fuse [static_emb, pos_emb, node_pool_emb, global_emb]
        fused_dim = fg_embed_dim + pos_embed_dim + hidden_dim + hidden_dim
        self.fuse_fg = nn.Linear(fused_dim, hidden_dim)
        self.count_global = 0
        
    def forward(self, batched_graph, fg_type_tensor, fg_indices_tensor):
        
        fg_mask = (fg_indices_tensor >= 0).any(dim=-1).float()  # [B, F_max]
        graph_x, graph_edge_index, graph_batch = batched_graph.x, batched_graph.edge_index, batched_graph.batch
        graph_x = F.relu(self.gnn1(graph_x, graph_edge_index))
        graph_x = self.gnn2(graph_x, graph_edge_index)
        # print('graph_x: ', graph_x.shape)
        # # Step 2: global pooling
        global_emb = self.global_pool(graph_x, graph_batch)  # [B, hidden_dim]
        # print('global_emb: ', global_emb.shape)
        B, F_max = fg_type_tensor.shape
        # print(fg_type_tensor)
        device = graph_x.device
        # print('device: ', device)
        # print(F_max)
        # # Step 3: static + positional embedding
        fg_static = self.static_fg_embedding(fg_type_tensor)  # [B, F_max, fg_embed_dim]
        positions = torch.arange(F_max, device=device).unsqueeze(0).expand(B, -1)
        fg_pos = self.pos_embedding(positions)  # [B, F_max, pos_embed_dim]
        # Step 4: pool node embeddings per FG group
        fg_node_repr = []
        fg_valid_mask = []  # [B, F_max] = 1 nếu FG hợp lệ, 0 nếu là padding
        
        for b in range(B):
            fg_repr_per_mol = []
            valid_per_mol = []
            for f in range(F_max):
                atom_ids = fg_indices_tensor[b, f]
                
                mask = atom_ids >= 0
                atom_ids = atom_ids[mask]
                if atom_ids.numel() == 0:
                    pooled_feat = torch.zeros((graph_x.size(1),), device=device)
                    valid_per_mol.append(0)
                else:
                    pooled_feat = graph_x[batched_graph.ptr[b] + atom_ids].mean(dim=0)
                    valid_per_mol.append(1)

                fg_repr_per_mol.append(pooled_feat)

            fg_node_repr.append(torch.stack(fg_repr_per_mol, dim=0))  # [F_max, hidden_dim]
            fg_valid_mask.append(torch.tensor(valid_per_mol, device=device))  # [F_max]

        fg_node_repr = torch.stack(fg_node_repr, dim=0)          # [B, F_max, hidden_dim]
        fg_valid_mask = (torch.stack(fg_valid_mask, dim=0) > 0)    # [B, F_max] boolean mask

        # Step 4b: Project node-level FG representations
        fg_node_proj = self.fg_node_pool_proj(fg_node_repr)       # [B, F_max, hidden_dim]

        # Step 5: Fuse static + pos + node_proj + global
        global_expanded = global_emb.unsqueeze(1).expand(-1, F_max, -1)  # [B, F_max, hidden_dim]
        fg_fused = torch.cat([fg_static, fg_pos, fg_node_proj, global_expanded], dim=-1)  # [B, F_max, fused_dim]
        fg_fused = self.fuse_fg(fg_fused)  # [B, F_max, hidden_dim]
        
        # print("fg_type_tensor: ",fg_type_tensor)
        # print("fg_indices_tensor: ",fg_indices_tensor)
        # print("fg_valid_mask: ",fg_valid_mask)
        # split = 'test'
        # os.makedirs(f"/home/phuc.phamhuythienai@gmail.com/Desktop/ExplainabilityInteraction/src/model/{split}_tsne_pdbbind", exist_ok=True)
        # for i in range(len(fg_valid_mask)):
        #     mask = fg_valid_mask[i]
        #     dict_save = {'fg_type_tensor': fg_type_tensor[i][mask], 'attn_out':attn_out[i][mask]}
        #     with open(f"/home/phuc.phamhuythienai@gmail.com/Desktop/ExplainabilityInteraction/src/model/{split}_tsne_pdbbind/{self.count_global}.pkl", "wb") as f:
        #         pickle.dump(dict_save, f)
        #     self.count_global += 1
        return fg_fused, fg_valid_mask


class SCAT(nn.Module):
    def __init__(self, embedding_dim=960, num_heads=8):
        super(SCAT, self).__init__()
        self.prot_self_attn    = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.ligand_self_attn  = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        
        self.arkmab            = ARKMAB(embedding_dim, num_heads)
    def forward(self, prot_vector, prot_mask, fg_embedded, fg_mask):
        prot_ctx = self.prot_self_attn(prot_vector, src_key_padding_mask=~prot_mask.bool())
        ligand_ctx = self.ligand_self_attn(fg_embedded, src_key_padding_mask=~fg_mask.bool())
        
        prot_ctx = prot_ctx.masked_fill(~prot_mask.unsqueeze(-1).bool(), 0.0)
        ligand_ctx = ligand_ctx.masked_fill(~fg_mask.unsqueeze(-1).bool(), 0.0)

        protein_output_ark, protein_mask_ark, protein_attn_weights_ark = self.arkmab(prot_ctx,prot_mask.float(),ligand_ctx,fg_mask.float())
        ligand_output_ark, ligand_mask_ark, ligand_attn_weights_ark = self.arkmab(ligand_ctx,fg_mask.float(),prot_ctx,prot_mask.float())

        masked_output_protein = protein_output_ark * protein_mask_ark.unsqueeze(-1)  # [B, Lp, D]
        masked_output_ligand = ligand_output_ark* ligand_mask_ark.unsqueeze(-1)  # [B, Lp, D]
        
        return masked_output_protein, masked_output_ligand
    
class ARKMAB(nn.Module):
    """
    ARKMAB: ARK Multihead Attention Block

    This module models asymmetric interactions between protein residues
    and ligand elements using pooling-based multi-head cross-attention.

    Key design choices:
    - Cross-attention from residues (queries) to ligand elements (keys/values)
    - Pooling-based attention to stabilize learning under ligand sparsity
    - Pseudo inactive ligand token to absorb non-interacting residues
    """
    def __init__(self, embedding_dim = 960, num_heads = 4, attn_option = 'additive', analysis_mode= False):
        super(ARKMAB, self).__init__()
        pmx_args      = (embedding_dim, num_heads, RFF(embedding_dim), attn_option, False, analysis_mode, False)
        self.pmx      = PoolingMultiheadCrossAttention(*pmx_args)
        self.inactive = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.fillmask = nn.Parameter(torch.ones(1,1), requires_grad=False)
        self.representations = []
        if analysis_mode: pass
        self.apply(initialization)

    def forward(self, residue_features, residue_masks, ligelem_features, ligelem_masks):
        '''
            residue_features:  batch size x residues x H
            residue_masks: batch size x residues x H
            ligelem_features:  batch size x ecfpsubs x H
            ligelem_masks: batch size x ecfpsubs x H
        '''
        
        b = residue_features.shape[0]
        pseudo_substructure = self.inactive.repeat(residue_features.size(0),1,1)
        pseudo_masks        = self.fillmask.repeat(residue_features.size(0),1)
        
        ligelem_features  = torch.cat([ligelem_features,  pseudo_substructure], 1)
        ligelem_masks = torch.cat([ligelem_masks, pseudo_masks], 1)
        

        residue_features_out, attention = self.pmx(Y=ligelem_features, Ym=ligelem_masks, X=residue_features, Xm=residue_masks)
        
        x, y, z = attention.size()
        attention = attention.view(x//b, b, y,z)
        
        return residue_features_out , residue_masks, attention 


class PairwiseUNet(nn.Module):
    """
    UNet module learning pairwise interactions with upsampling + conv and
    explicit interpolation sizes to ensure spatial dimensions always match.

    Input:
        - prot_ctx: [B, R, D]
        - lig_ctx:  [B, F, D]
    Output:
        - logits:   [B, R, F, num_classes]
    """
    def __init__(self, embedding_dim: int, num_classes: int = 7, base_channels: int = 256):
        super().__init__()
        D = embedding_dim
        in_ch = 2 * D

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))  # Pool only on protein dimension (R)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))  # Pool only on protein dimension (R)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 6, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final classifier conv
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, prot_ctx: torch.Tensor, lig_ctx: torch.Tensor) -> torch.Tensor:
        B, R, D = prot_ctx.shape
        Fg = lig_ctx.size(1)

        # Pairwise feature map [B, 2D, R, Fg]
        prot_exp = prot_ctx.unsqueeze(1).expand(B, Fg, R, D)
        lig_exp = lig_ctx.unsqueeze(2).expand(B, Fg, R, D)
        x = torch.cat([prot_exp, lig_exp], dim=-1)      # [B, Fg, R, 2D]
        x = x.permute(0, 3, 2, 1)                        # [B, 2D, R, Fg]

        # Encoder
        e1 = self.enc1(x)                                # [B, C, R, Fg]
        p1 = self.pool1(e1)                              # [B, C, R//2, Fg]

        e2 = self.enc2(p1)                               # [B, 2C, R//2, Fg]
        p2 = self.pool2(e2)                              # [B, 2C, R//4, Fg]

        # Bottleneck
        b = self.bottleneck(p2)                          # [B, 4C, R//4, Fg]

        # Decoder level 2
        u2 = fn.interpolate(b, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([u2, e2], dim=1)                  # [B, 6C, R//2, Fg]
        d2 = self.dec2(d2)                               # [B, 2C, R//2, Fg]

        # Decoder level 1
        u1 = fn.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([u1, e1], dim=1)                  # [B, 3C, R, Fg]
        d1 = self.dec1(d1)                               # [B, C, R, Fg]

        # Final classifier
        logits = self.final_conv(d1)                     # [B, num_classes, R, Fg]
        logits = logits.permute(0, 2, 3, 1)              # [B, R, Fg, num_classes]
        return logits
    
    
class Bidirectional_Interaction_Type_Attention(nn.Module):
    """
    Aggregate ligand <-> protein information using interaction logits with
    learnable interaction-type weights and bidirectional attention pooling.

    Inputs:
        - ligand_emb: [B, F, D]   (F = num functional groups)
        - protein_emb: [B, R, D]  (R = num residues)
        - logits: [B, R, F, K]     (K = num interaction types)
        - fg_mask: [B, F] (1 real / 0 pad) optional
        - prot_mask: [B, R] optional

    Output:
        - pred: [B, 1]
        - aux: dict with intermediate tensors (S, edge_strength, contexts...)
    """
    def __init__(self, emb_dim, num_types=7, hidden=512, nonneg=True, use_sigmoid=False):
        super().__init__()
        self.D = emb_dim
        self.K = num_types
        self.nonneg = nonneg
        self.use_sigmoid = use_sigmoid

        # learnable scalar weight per interaction type
        self.type_weight = nn.Parameter(torch.ones(self.K))
        # optional small transforms for contexts
        self.proj_r = nn.Linear(emb_dim, emb_dim)
        self.proj_f = nn.Linear(emb_dim, emb_dim)

        # final MLP
        mlp_in = 2 * emb_dim + self.K  # prot_pool, lig_pool, type_strength
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, ligand_emb, protein_emb, logits, fg_mask=None, prot_mask=None):
        """
        ligand_emb: [B, F, D]
        protein_emb: [B, R, D]
        logits: [B, R, F, K]
        fg_mask: [B, F] float (1/0)
        prot_mask: [B, R] float (1/0)
        """
        B, F, D = ligand_emb.shape
        _, R, _ = protein_emb.shape
        device = ligand_emb.device
        eps = 1e-8
        # default masks
        if fg_mask is None:
            fg_mask = torch.ones(B, F, device=device)
        if prot_mask is None:
            prot_mask = torch.ones(B, R, device=device)

        # 1) interaction probabilities per type
        if self.use_sigmoid:
            # multi-label style: independent probabilities per type
            p = torch.sigmoid(logits)  # [B,R,F,K]
            # optionally normalize? we'll not normalize here
        else:
            # softmax over interaction types for each (b,r,f)
            p = torch.nn.functional.softmax(logits, dim=-1)  # [B,R,F,K]

        # 2) type weights (non-negative if requested)
        if self.nonneg:
            w = torch.nn.functional.softplus(self.type_weight)  # [K]
        else:
            w = self.type_weight  # can be negative

        # 3) weighted probabilities and collapse K -> scalar edge strength
        # weighted: [B,R,F,K]
        weighted = p * w.view(1,1,1,self.K)
        # edge_strength S: [B,R,F]
        S = weighted.sum(dim=-1)

        # mask out pad positions
        S = S * prot_mask.view(B, R, 1) * fg_mask.view(B, 1, F)
        
        # 4) Compute bidirectional attention:
        #  - residue -> ligand: for each residue r, distribution over F
        #  - ligand  -> residue: for each ligand f, distribution over R
        # We'll compute softmax with masks and numerical stability

        # residue -> ligand attention weights (normalize across F)
        S_r = S.clone()
        S_r = S_r + (1.0 - fg_mask.view(B,1,F)) * (-1e9)  # mask F by -inf
        attn_res2lig = torch.nn.functional.softmax(S_r, dim=2)  # [B,R,F]

        # ligand -> residue attention weights (normalize across R)
        S_l = S.clone()
        S_l = S_l + (1.0 - prot_mask.view(B,R,1)) * (-1e9)
        attn_lig2res = torch.nn.functional.softmax(S_l, dim=1)  # [B,R,F] softmax over R

        # 5) contexts computed by weighted sum
        # ligand_context_per_residue: for each residue r, aggregated ligand vector: [B,R,D]
        ligand_exp = ligand_emb.unsqueeze(1).expand(B, R, F, D)
        ligand_context_per_res = (attn_res2lig.unsqueeze(-1) * ligand_exp).sum(dim=2)  # [B,R,D]

        # protein_context_per_ligand: for each ligand f, aggregated protein vector: [B,F,D]
        protein_exp = protein_emb.unsqueeze(2).expand(B, R, F, D)
        prot_context_per_lig = (attn_lig2res.unsqueeze(-1) * protein_exp).sum(dim=1)  # [B,F,D]

        # 6) combine contexts with original embeddings to get enriched node embeddings
        # For protein: combine original protein_emb with ligand-derived context (pool across residues later)
        prot_enriched = protein_emb + self.proj_r(ligand_context_per_res)  # [B,R,D]
        lig_enriched = ligand_emb + self.proj_f(prot_context_per_lig)      # [B,F,D]

        # 7) compute importance scores per node (sum of edge strengths)
        prot_score = S.sum(dim=2) * prot_mask  # [B,R]
        lig_score = S.sum(dim=1) * fg_mask     # [B,F]

        # normalize to get pooling weights
        prot_weights = prot_score / (prot_score.sum(dim=1, keepdim=True) + eps)  # [B,R]
        lig_weights = lig_score / (lig_score.sum(dim=1, keepdim=True) + eps)    # [B,F]

        # pooled vectors
        prot_pool = (prot_weights.unsqueeze(-1) * prot_enriched).sum(dim=1)  # [B,D]
        lig_pool = (lig_weights.unsqueeze(-1) * lig_enriched).sum(dim=1)    # [B,D]

        # 8) also compute type_strength global vector (sum of weighted probs over R,F)
        type_strength = weighted.sum(dim=(1,2))  # [B,K]

        # 9) final feature and prediction
        final_feat = torch.cat([prot_pool, lig_pool, type_strength], dim=-1)  # [B, 2D+K]
        final_feat_norm = torch.nn.functional.normalize(final_feat, dim=-1)  # just use for latent_loss
        pred = self.mlp(final_feat)  # [B,1]
        return pred, final_feat_norm
    
class LINKER(nn.Module): 
    def __init__(self, num_fg_types = 205, embedding_dim=960, num_heads=8):
        super().__init__()
        self.scat              = SCAT(embedding_dim=embedding_dim, num_heads=num_heads)
        self.unet_pair         = PairwiseUNet(
            embedding_dim=embedding_dim,
            num_classes=7,
            base_channels=256  # có thể điều chỉnh
        )
        self.finger_id      = FINGER_ID(num_fg_types = num_fg_types)
    def forward(self, prot_vector, prot_mask, batched_graph, fg_indices_tensor, fg_type_tensor):
        B, R, D = prot_vector.shape
        F = fg_indices_tensor.shape[1]
        
        fg_embedded, fg_mask = self.finger_id(batched_graph, fg_type_tensor, fg_indices_tensor)
        
        masked_output_protein, masked_output_ligand = self.scat(prot_vector, prot_mask, fg_embedded, fg_mask)
        
        logits = self.unet_pair(masked_output_protein, masked_output_ligand)
        
        
        return logits.permute(0, 2, 1, 3)