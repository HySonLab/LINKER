import os
import sys
import argparse
import warnings
import pickle as pkl
from glob import glob
from collections import defaultdict
import ast

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from rdkit import Chem
from openbabel import openbabel as ob
from pymol import cmd
from plip.structure.preparation import PDBComplex

# -----------------------------
# Set up environment
# -----------------------------
warnings.filterwarnings("ignore")
cmd.feedback("disable", "all", "everything")
handler = ob.OBMessageHandler()
handler.SetOutputLevel(0)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CheckMol

# -----------------------------
# Helper functions
# -----------------------------
def parse_atom_groups(atom_str):
    raw_groups = atom_str.strip().split(',')
    groups = []
    for group in raw_groups:
        if not group:
            continue
        atom_ids = [int(x) - 1 for x in group.strip('-').split('-') if x]
        if atom_ids:
            groups.append(atom_ids)
    return groups

def atom_feature(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetHybridization().real,
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
    ]

def bond_feature(bond):
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC
    ]

def mol_to_graph(mol, atom2fg_mask):
    atom_features = [atom_feature(atom) for atom in mol.GetAtoms()]
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_feat = bond_feature(bond)
        edge_attr.extend([bond_feat, bond_feat])

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, atom2fg_mask=atom2fg_mask)

def extract_fg_info_from_graph(graph: Data):
    mask = graph.atom2fg_mask
    fg_ids = torch.nonzero(mask.sum(dim=0) > 0, as_tuple=False).view(-1).tolist()
    fg_node_map = [torch.nonzero(mask[:, j], as_tuple=False).view(-1).tolist() for j in fg_ids]
    return fg_ids, fg_node_map

def build_fg_info(df, mol=None, max_fg_id=205):
    fg_node_map, fg_type_list = [], []
    assigned_atoms, atom_to_groups = set(), {}

    for idx, row in df.iterrows():
        fg_id = int(row['Functional Group Number'])
        atom_groups = parse_atom_groups(row['Atom Position'])
        for group in atom_groups:
            group_idx = len(fg_node_map)
            fg_node_map.append(group)
            fg_type_list.append(fg_id)
            for atom in group:
                assigned_atoms.add(atom)
                atom_to_groups.setdefault(atom, []).append(group_idx)

    # Assign remaining atoms
    num_atoms = mol.GetNumAtoms()
    unassigned_atoms = [i for i in range(num_atoms) if i not in assigned_atoms]
    for atom_idx in unassigned_atoms:
        min_dist, nearest_atom = float('inf'), None
        for assigned_atom in assigned_atoms:
            dist = abs(atom_idx - assigned_atom)
            if dist < min_dist:
                min_dist = dist
                nearest_atom = assigned_atom
        if nearest_atom is not None and nearest_atom in atom_to_groups:
            for group_idx in atom_to_groups[nearest_atom]:
                fg_node_map[group_idx].append(atom_idx)

    # Padded indices
    fg_indices = [torch.tensor(group, dtype=torch.long) for group in fg_node_map]
    fg_indices = pad_sequence(fg_indices, batch_first=True, padding_value=-1)
    return fg_node_map, fg_type_list, fg_indices

def load_mol_and_convert_to_graph(mol_path):
    mol = Chem.MolFromMolFile(mol_path, sanitize=False)
    if mol is None:
        raise ValueError(f"Cannot read molecule from {mol_path}")

    cm = CheckMol()
    res = cm.functionalGroups(file=mol_path, justFGcode=False, returnDataframe=True)
    atom2fg_dict = defaultdict(list)
    for idx, row in res.iterrows():
        fg_id = int(row['Functional Group Number'])
        atom_str = row['Atom Position'].replace('-', ',').replace(',,', ',')
        atom_list = [int(a) for a in atom_str.strip(',').split(',') if a]
        for atom in atom_list:
            atom2fg_dict[atom].append(fg_id)

    max_fg_id = 205
    atom2fg_mask = torch.zeros((mol.GetNumAtoms(), max_fg_id), dtype=torch.float)
    for atom_idx, fg_list in atom2fg_dict.items():
        for fg_id in fg_list:
            atom2fg_mask[atom_idx - 1, fg_id] = 1.0

    fg_ids, fg_node_map = extract_fg_info_from_graph(mol_to_graph(mol, atom2fg_mask))
    return mol_to_graph(mol, atom2fg_mask)

# -----------------------------
# BindingDB
# -----------------------------
class BindingDBDrugFeaturizer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.interaction_dict = {
            'hbond': 0, 'hydroph_interaction': 1, 'pistack': 2,
            'pication': 3, 'saltbridge': 4, 'water_bridges': 5,
            'halogenbond': 6
        }

        os.makedirs(os.path.join(output_dir, 'label'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fg_instance'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ligand_graph'), exist_ok=True)

    def process_file(self, file_path):
        sample = pkl.load(open(file_path, 'rb'))
        protein_name = os.path.basename(sample['protein path']).replace('_protein.pdb', '')
        ligand_path = sample['protein path'].replace('_protein.pdb', '_ligand.sdf')
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)

        if mol is None:
            print(f"[ERROR] Cannot read ligand: {ligand_path}")
            return False

        # Create interaction labels
        label = np.zeros((mol.GetNumAtoms(), len(sample['protein_sequence']), 7), dtype=int)
        for protein_res, interaction in sample['interactions']:
            protein_idx = int(protein_res.split(':')[1])
            inter_type = self.interaction_dict[interaction.split('-')[0]]
            ligand_part = interaction.split(':')[1]

            try:
                atom_ids = [int(ligand_part) - 1]
            except:
                atom_ids = [int(a) - 1 for a in ast.literal_eval(ligand_part)]

            for atom_id in atom_ids:
                label[atom_id, protein_idx, inter_type] = 1

        # Functional groups
        cm = CheckMol()
        res = cm.functionalGroups(file=ligand_path, justFGcode=False, returnDataframe=True)
        if len(res) == 0:
            print(f"[ERROR] No functional groups for {protein_name}")
            return False

        fg_node_map, fg_type_list, fg_indices = build_fg_info(res, mol)

        # Build Label_fg
        Label_plip = torch.tensor(label, dtype=torch.float)
        num_fg, num_res, num_types = len(fg_node_map), Label_plip.size(1), Label_plip.size(2)
        Label_fg = torch.zeros((num_fg, num_res, num_types), dtype=torch.float)
        for i, atom_ids in enumerate(fg_node_map):
            if len(atom_ids) == 0:
                continue
            idx_tensor = torch.tensor(atom_ids, dtype=torch.long)
            Label_fg[i] = Label_plip[idx_tensor].max(dim=0).values

        # Save outputs
        torch.save(Label_fg, os.path.join(self.output_dir, 'label', f'{protein_name}.pt'))
        with open(os.path.join(self.output_dir, 'fg_instance', f'{protein_name}.pkl'), 'wb') as f:
            pkl.dump({'fg_node_map': fg_node_map, 'fg_type_list': fg_type_list, 'fg_indices': fg_indices}, f)

        # Save ligand graph
        graph = load_mol_and_convert_to_graph(ligand_path)
        torch.save(graph, os.path.join(self.output_dir, 'ligand_graph', f'{protein_name}.pt'))

        return True

    def process_all(self):
        files = glob(os.path.join(self.input_dir, '*.pkl'))
        wrong_files = []
        for idx, file_path in enumerate(files):
            print(f'[{idx+1}/{len(files)}] Processing: {file_path}')
            success = self.process_file(file_path)
            if not success:
                wrong_files.append(idx)
        print("Done processing all files. Wrong indices:", wrong_files)
        
        
# -----------------------------
# PDBBind
# -----------------------------
class PDBBindDrugFeaturizer:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'label'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fg_instance'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'ligand_graph'), exist_ok=True)

    def process_file(self, file_path):
        sample = pkl.load(open(file_path, 'rb'))
        print(file_path)
        ligand_path = sample['mol_path']
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        # Create interaction labels
        label = sample['interactions']
        
        protein_name = sample['protein_name']
        # Functional groups
        cm = CheckMol()
        res = cm.functionalGroups(file=ligand_path, justFGcode=False, returnDataframe=True)
        if len(res) == 0:
            print(f"[ERROR] No functional groups for {protein_name}")
            return False

        fg_node_map, fg_type_list, fg_indices = build_fg_info(res, mol)

        # Build Label_fg
        Label_plip = torch.tensor(label, dtype=torch.float)
        num_fg, num_res, num_types = len(fg_node_map), Label_plip.size(1), Label_plip.size(2)
        Label_fg = torch.zeros((num_fg, num_res, num_types), dtype=torch.float)
        for i, atom_ids in enumerate(fg_node_map):
            if len(atom_ids) == 0:
                continue
            idx_tensor = torch.tensor(atom_ids, dtype=torch.long)
            Label_fg[i] = Label_plip[idx_tensor].max(dim=0).values
        # Save outputs
        torch.save(Label_fg, os.path.join(self.output_dir, 'label', f'{protein_name}.pt'))
        with open(os.path.join(self.output_dir, 'fg_instance', f'{protein_name}.pkl'), 'wb') as f:
            pkl.dump({'fg_node_map': fg_node_map, 'fg_type_list': fg_type_list, 'fg_indices': fg_indices}, f)

        # Save ligand graph
        graph = load_mol_and_convert_to_graph(ligand_path)
        torch.save(graph, os.path.join(self.output_dir, 'ligand_graph', f'{protein_name}.pt'))

        return True

    def process_all(self):
        files = glob(os.path.join(self.input_dir, '*.pkl'))
        wrong_files = []
        for idx, file_path in enumerate(files):
            print(f'[{idx+1}/{len(files)}] Processing: {file_path}')
            success = self.process_file(file_path)
            if not success:
                wrong_files.append(idx)
        print("Done processing all files. Wrong indices:", wrong_files)
        
        

# -----------------------------
# Main CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize BindingDB/PDBBind pkl files")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with pkl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    args = parser.parse_args()

    featurizer = PDBBindDrugFeaturizer(args.input_dir, args.output_dir)
    featurizer.process_all()
