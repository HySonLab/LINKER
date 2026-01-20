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


def scale_affinity(x, min_val, max_val):
    return 2 * (x - min_val) / (max_val - min_val) - 1
def unscale_affinity(x_scaled, min_val, max_val):
    return ((x_scaled + 1) / 2) * (max_val - min_val) + min_val


class LINKER_Dataloader(Dataset):
    def __init__(self, dataframe_path, split, protein_dir, fg_info_dir, graph_dir, label_dir):
        df = pd.read_csv(dataframe_path)
        self.protein_dir = protein_dir
        self.fg_info_dir = fg_info_dir
        self.graph_dir = graph_dir
        self.label_dir = label_dir
        self.df = df[df['new_split'] == split] 
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row                     = self.df.iloc[idx]
        protein_name            = row['protein']
        prot_tensor             = torch.load(os.path.join(self.protein_dir,  f'{protein_name}.pt'), weights_only=False)
        
        
        fg_group_info           = pickle.load(open(os.path.join(self.fg_info_dir,  f'{protein_name}.pkl'), 'rb'))
        fg_node_map, fg_type_list, fg_indices = fg_group_info['fg_node_map'], fg_group_info['fg_type_list'], fg_group_info['fg_indices']
        
        graph_data              = torch.load(os.path.join(self.graph_dir,  f'{protein_name}.pt'), weights_only=False)
        
        
        complex_label           = torch.load(os.path.join(self.label_dir, f'{protein_name}.pt'), weights_only=False)
        value                   = torch.tensor(row['value'])
        prot_mask               = torch.ones(prot_tensor.shape[0],      dtype=torch.long)  # real positions
        complex_label_mask      = torch.ones((complex_label.shape[0],complex_label.shape[1]),   dtype=torch.long)

        
        pydata                  = (protein_name, prot_tensor, prot_mask, graph_data, fg_node_map, fg_type_list, fg_indices, complex_label, complex_label_mask, value)
        
        return pydata



def collate_fn_LINKER(batch):
    # Unpack batch
    names, prot_tensors, prot_masks, graph_datas, fg_node_maps, fg_type_lists, fg_indices_list, complex_labels, complex_label_masks, values = zip(*batch)
    
    

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

    # Giả sử complex_labels là list các tensor [F_i, R_i, C]
    max_f = max(label.shape[0] for label in complex_labels)
    max_r = max(label.shape[1] for label in complex_labels)

    padded_labels = torch.stack([
        F.pad(label, (0, 0, 0, max_r - label.shape[1], 0, max_f - label.shape[0]), value=0)
        for label in complex_labels
    ])  # [B, F_max, R_max, C]

    padded_label_masks = torch.stack([
        F.pad(mask, (0, max_r - mask.shape[1], 0, max_f - mask.shape[0]), value=0)
        for mask in complex_label_masks
    ])  # [B, F_max, R_max]
    values = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)

    return names,prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor,padded_labels,padded_label_masks, values

class BindingAffinityPrediction_Dataset(Dataset):
    def __init__(self, base_path = 'evaluation/extracted_features' , split = 'train'):
        self.path = glob(os.path.join(base_path, f'{split}/*'))

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        sample = torch.load(self.path[idx], weights_only=False)
        return sample

def collate_fn_binding_affinity(batch):
    # Mỗi phần tử trong batch là 1 dict
    protein_names = [item['protein_name'] for item in batch]
    values = torch.tensor([item['value'] for item in batch], dtype=torch.float32)

    # Padding các tensor biến động theo chiều 0 (FG, prot_ctx, ligand_ctx,...)
    fg_embedded = pad_sequence([item['fg_embedded'] for item in batch], batch_first=True)
    fg_mask = pad_sequence([item['fg_mask'] for item in batch], batch_first=True)

    prot_tensors = pad_sequence([item['prot_tensors'] for item in batch], batch_first=True)
    prot_ctx = pad_sequence([item['prot_ctx'] for item in batch], batch_first=True)
    prot_mask = pad_sequence([item['prot_mask'] for item in batch], batch_first=True)
    ligand_ctx = pad_sequence([item['ligand_ctx'] for item in batch], batch_first=True)

    protein_output_ark = pad_sequence([item['protein_output_ark'] for item in batch], batch_first=True)
    ligand_output_ark = pad_sequence([item['ligand_output_ark'] for item in batch], batch_first=True)


    # Giả sử complex_labels là list các tensor [F_i, R_i, C]
    max_f = max([item['logits'].shape[0] for item in batch])
    max_r = max([item['logits'].shape[1] for item in batch])

    padded_logits = torch.stack([
        F.pad(item['logits'], (0, 0, 0, max_r - item['logits'].shape[1], 0, max_f - item['logits'].shape[0]))
        for item in batch
    ])  #
    
    return {
        'protein_names': protein_names,
        'fg_embedded': fg_embedded,
        'prot_tensors': prot_tensors,
        'fg_mask': fg_mask,
        'prot_ctx': prot_ctx,
        'prot_mask': prot_mask,
        'ligand_ctx': ligand_ctx,
        'protein_output_ark': protein_output_ark,
        'ligand_output_ark': ligand_output_ark,
        'logits': padded_logits,
        'values': values.unsqueeze(1)  # [B, 1]
    }


def main(args):
    dataset = LINKER_Dataloader(
        args.csv_path,
        'train',
        args.protein_emb_path,
        args.fg_instance_path,
        args.ligand_graph_path,
        args.label_path
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    for names,prot_tensors, prot_masks, batched_graph, fg_indices_tensor, fg_type_tensor,padded_labels,padded_label_masks, values in train_loader:
        print("names:", names)
        print("prot_tensors:", prot_tensors)
        print("prot_masks:", prot_masks)
        print("batched_graph:", batched_graph)
        print("fg_indices_tensor:", fg_indices_tensor)
        print("fg_type_tensor:", fg_type_tensor)
        print("padded_labels:", padded_labels)
        print("padded_label_masks:", padded_label_masks)
        print("values:", values)
        break  # chỉ in phần tử đầu tiên

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DataLoader for PLIP dataset")

    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file (data_splits_with_score.csv)')
    parser.add_argument('--protein_emb_path', type=str, required=True, help='Path to protein embedding folder')
    parser.add_argument('--fg_instance_path', type=str, required=True, help='Path to functional group instance folder')
    parser.add_argument('--ligand_graph_path', type=str, required=True, help='Path to ligand graph folder')
    parser.add_argument('--label_path', type=str, required=True, help='Path to label folder')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader')

    args = parser.parse_args()
    main(args)





