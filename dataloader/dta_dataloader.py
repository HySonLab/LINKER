# =========================
# Standard library imports
# =========================
import os
import sys
import argparse
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =========================
# PyTorch Geometric imports
# =========================
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch as GeometricBatch
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


class DTA_Dataloader(Dataset):
    def __init__(self, df, protein_emb_path=None, fg_instance_path=None, ligand_graph_path=None):
        """
        df: pandas DataFrame
        ligand_featurizer: function SMILES
        protein_featurizer: function sequence
        """
        self.df                 = df
        self.protein_emb_dict   = pickle.load(open(protein_emb_path, "rb"))
        self.fg_instance_dict  = pickle.load(open(fg_instance_path, "rb"))
        self.ligand_graph_dict  = pickle.load(open(ligand_graph_path, "rb"))
        print("DTA_Dataloader initialized with {} samples.".format(len(df)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_id = row["protein_id"]
        ligand_id  = row["ligand_id"]
        label      = row["label"]
        
        fg_group_info  = self.fg_instance_dict[ligand_id]
        fg_node_map, fg_type_list, fg_indices = fg_group_info['fg_node_map'], fg_group_info['fg_type_list'], fg_group_info['fg_indices']
        graph_data = self.ligand_graph_dict[ligand_id]
        prot_tensor  = self.protein_emb_dict[protein_id]
        prot_mask = torch.ones(prot_tensor.shape[0], dtype=torch.long)
        return (prot_tensor, prot_mask, graph_data, fg_node_map, fg_type_list, fg_indices, label)



def collate_fn_DTA(batch):
    # Unpack batch
    prot_tensors, prot_masks, graph_datas, fg_node_maps, fg_type_lists, fg_indices_list, labels = zip(*batch)
    
    

    # FG indices tensor: pad each [F_i, A_i] to (B, F_max, A_max)
    max_atoms = max([fgi.size(1) for fgi in fg_indices_list])
    max_F = max([len(t) for t in fg_type_lists])
    
    
    # Protein features
    prot_tensors = pad_sequence(prot_tensors, batch_first=True)
    prot_masks = pad_sequence(prot_masks, batch_first=True)

    # FG type list: padding to max F
    fg_type_tensor = torch.full((len(batch), max_F), fill_value=0, dtype=torch.long)

    for i, fgs in enumerate(fg_type_lists):
        fg_type_tensor[i, :len(fgs)] = torch.tensor(fgs, dtype=torch.long)
        
    padded_fg_indices = []
    
    for fgi in fg_indices_list:
        # fgi: (F_i, A_i)
        pad_width = (0, max_atoms - fgi.size(1))  # Pad to right
        padded = F.pad(fgi, pad_width, value=-1)
        if padded.size(0) < max_F:
            padded = F.pad(padded, (0, 0, 0, max_F - padded.size(0)), value=-1)
        padded_fg_indices.append(padded)
    fg_indices_tensor = torch.stack(padded_fg_indices, dim=0)  # (B, F_max, A_max)
    
    # Graph batch
    batched_graph = GeometricBatch.from_data_list(graph_datas)
    labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)

    return prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor, labels

def main(args):
    df = pd.read_csv(args.csv_path)
    folds = ["Train_1", "Train_2", "Train_3", "Train_4", "Train_5"]
        
    for val_fold in folds:
        
        print(f"Running fold: {val_fold}")
        
        # Train
        train_df = df[(df["split"].isin(folds)) & (df["split"] != val_fold)]

        # Validation
        val_df = df[df["split"] == val_fold]
        
        train_dataset = DTA_Dataloader(train_df, args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path)
        val_dataset   = DTA_Dataloader(val_df, args.protein_emb_path, args.fg_instance_path, args.ligand_graph_path)
        
        
        # ===== DataLoader =====
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_DTA)
        val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_DTA)

        
        for x in train_loader:
            print(x)
            break
        
        for x in val_loader:
            print(x)
            break
        
        print("Train size:", len(train_df))
        print("Val size:", len(val_df))
        break

    test_df = df[df["split"] == "Test"]
    print("Test size:", len(test_df))
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DataLoader for DTA dataset")

    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file (data_splits_with_score.csv)')
    parser.add_argument('--protein_emb_path', type=str, required=True, help='Path to protein embedding folder')
    parser.add_argument('--fg_instance_path', type=str, required=True, help='Path to functional group instance folder')
    parser.add_argument('--ligand_graph_path', type=str, required=True, help='Path to ligand graph folder')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')

    args = parser.parse_args()
    main(args)




