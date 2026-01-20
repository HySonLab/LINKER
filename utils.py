import numpy as np
import pandas as pd
import os
# import torch
import shutil
from glob import glob
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from Bio import PDB
from Bio.SeqUtils import seq1
from collections import defaultdict
from difflib import SequenceMatcher
from pyCheckmol import *
from collections import defaultdict
import subprocess
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ABSOLUT_PATH = os.path.dirname(os.path.realpath(__file__))
checkmol_dir = "pyCheckmol/pyCheckmol/bin"
os.environ["PATH"] += os.pathsep + checkmol_dir

# LP_PDBBind_path = '../data/PDBBind/LP_PDBBind.csv'

res_short = {       
        'ALA': 'A',
        'ARG': 'R',
        'ASN': 'N',
        'ASP': 'D',
        'CYS': 'C',
        'GLN': 'Q',
        'GLU': 'E',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LEU': 'L',
        'LYS': 'K',
        'MET': 'M',
        'PHE': 'F',
        'PRO': 'P',
        'SER': 'S',
        'THR': 'T',
        'TRP': 'W',
        'TYR': 'Y',
        'VAL': 'V',
        'HER': 'H'
        }


# Remove leading/trailing spaces from names
df = pd.read_csv('pyCheckmol/data/fg_list.csv', sep = ';')
df["FG_name"] = df["FG_name"].str.strip()
df = df[df["FG_number"].notna()]  # loại bỏ hàng có NaN
df["FG_number"] = df["FG_number"].astype(int)
# Create dictionaries
number_to_name = dict(zip(df["FG_number"], df["FG_name"]))
name_to_number = {name: number for number, name in number_to_name.items()}


class CheckMol:
    def __init__(self):
        self.information_ = ''

    def functionalGroupASbitvector(self, smiles = ''):
        """This function returns a array of 0's and 1's that
        represents the presence or absence of a functional group.
        The first position of this array is always 0, because python
        start a array as index 0. Here we Following the table 1-204 
        (index) 

        Args:
            smiles (str, optional): Smiles as a string.
        """
        get_list = self.functionalGroupSmiles(smiles=smiles)
        vet = np.zeros(205)
        vet[get_list] = 1
        
        return vet
        
    def functionalGroupSmiles(self, smiles = '', isString=True, generate3D=False,justFGcode=True, returnDataframe=True, deleteTMP=True):
        """
        This funtion returns the Functional groups (FG) information. Each FG is
        labeled with a code. Altogether there are 204 FG labeled. The table with
        this information can be viewed at 
        https://github.com/jeffrichardchemistry/pyCheckmol/blob/master/examples/fgtable.pdf
        
        Arguments
        ---------------------
        smiles
            or a Path to file, file must be a smiles extension: .smiles or .smi.
        isString
            If True a string must be passed in `smiles` argument, otherwise `smiles` argument
            must be a path
        generate3D
            If true the smiles will be converted into a sdf with 3D coordinates.
            Openbabel run in backend. If False will be converted with 2D coordinates.
        justFGcode
            if True return just code of FG, If False return the FG's code,
            number of atoms with these code and each atoms label as a dataframe
            or dict.
        returnDataframe
            Use only justFGcode=False. If returnDataframe=False the result is a
            dictionary, if True result is a dataframe.
        deleteTMP
            If True the temporary file will be deleted. The tmp file is created
            in $HOME/.pycheckmoltmp/
        """

        homedir = './'
        
        if isString:
            f = open(homedir+'smitmp.smiles', 'w')
            f.write(smiles)
            f.close()

            smiles = homedir+'smitmp.smiles'

        if generate3D:
            smi2sdf = subprocess.getoutput('obabel {} -O {}tmp.sdf --gen3D'.format(smiles, homedir))
            fg = CheckMol.functionalGroups(self, file=homedir+'tmp.sdf', justFGcode=justFGcode, returnDataframe=returnDataframe)
            if deleteTMP:
                os.remove(homedir+'tmp.sdf')
            else:
                pass
            return fg
        else:
            smi2sdf = subprocess.getoutput('obabel {} -O {}tmp.sdf --gen2D'.format(smiles, homedir))
            fg = CheckMol.functionalGroups(self, file=homedir+'tmp.sdf', justFGcode=justFGcode, returnDataframe=returnDataframe)
            if deleteTMP:
                os.remove(homedir+'tmp.sdf')
                os.remove(homedir+'smitmp.smiles')
            else:
                pass
            return fg


    def functionalGroups(self, file = '', justFGcode=True, returnDataframe=True):
        """
        This funtion returns the Functional groups (FG) information. Each FG is
        labeled with a code. Altogether there are 204 FG labeled. The table with
        this information can be viewed at 
        https://github.com/jeffrichardchemistry/pyCheckmol/blob/master/examples/fgtable.pdf
        
        Arguments
        ---------------------
        file
            Path to file, must be: .sdf, .mol, .mol2
        justFGcode
            if True return just code of FG, If False return the FG's code,
            number of atoms with these code and each atoms label as a dataframe
            or dict.
        returnDataframe
            Use only justFGcode=False. If returnDataframe=False the result is a
            dictionary, if True result is a dataframe.
        """
        get = subprocess.getoutput('checkmol -p {}'.format(file))
        get = get.replace('#','')
        self.information_ =  os.popen('checkmol -v {}'.format(file)).read()
        if justFGcode:
            get = [int(getline.replace(getline[3:], '')) for getline in get.splitlines()]
            return get
        else:
            getdf = pd.DataFrame([x.split(':') for x in get.splitlines()], columns=['FG_code', 'n_atoms', 'Atoms_label'])
            path2fulltable = '{}/pyCheckmol/data/fg_list.csv'.format(ABSOLUT_PATH)
            dftable = pd.read_csv(path2fulltable,sep=';')
            index2get = np.array(getdf['FG_code'].values).astype(int) - 1
            
            df_filteredTable = dftable.iloc[index2get,:].reset_index(drop=True)
            getdf = pd.concat([getdf,df_filteredTable.iloc[:,[1,2]]],axis=1,ignore_index=True)
            getdf.columns = ['Functional Group Number', 'Frequency', 'Atom Position', 'Functional Group', 'Code']
            getdf = getdf[['Functional Group','Frequency','Atom Position','Functional Group Number','Code']]
            if returnDataframe:
                return getdf
            else:
                return getdf.to_dict('list')
        



def get_res_short(res_long):
    if res_long in list(res_short.keys()):
        re = res_short[res_long]
    else:
        re = 'X'
    return re

def get_seq_str(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    header_lines = []
    for l in lines:
        if l.startswith('SEQRES'):
            header_lines.append(l)
    seqs_lines = {}
    chain_id = []
    for hl in header_lines:
        if not hl[11] in chain_id:
            chain_id.append(hl[11])
            seqs_lines[hl[11]] = []
            seqs_lines[hl[11]].append(hl)
        else:
            seqs_lines[hl[11]].append(hl)

    seqs = {}
    for key in seqs_lines:
        seqs[key] = ''
        for l in seqs_lines[key]:
            resl = l[18:].split(' ')
            for res in resl:
                res = res.strip()
                if len(res) == 3:
                    seqs[key] += get_res_short(res)
    # seqs_str = ''
    # for key in seqs:
    #     seqs_str+=seqs[key]
    #     seqs_str+=':'
    # seqs_str = seqs_str[:-2]

    return seqs




def get_seq_str_with_removed_chains(addr):
    seq_dict = get_seq_str(addr)
    seen = {}
    removed = []
    
    final_seqs = {}
    for chain, seq in seq_dict.items():
        if seq in seen:
            removed.append(chain)  # chain này có cùng sequence với chain trước
        else:
            seen[seq] = chain
            final_seqs[chain] = seq

    print("Removed chains due to duplicate sequence:", removed)
    
    if len(final_seqs) == 1:
        return list(final_seqs.values())[0]
    else:
        return ":".join(final_seqs.values())


def extract_pocket_residues(pdb_file, mol2_file, distance_cutoff=5.0):
    # Parse protein structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    # Parse ligand using RDKit
    ligand = Chem.MolFromMol2File(mol2_file, removeHs=False)
    ligand_conf = ligand.GetConformer()
    ligand_coords = np.array([ligand_conf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())])

    pocket_residues = set()
    first_res = {}
    for chain in model:
        first_res[chain.id] = 10e6
        for residue in chain:
            if not is_aa(residue):
                continue
            for atom in residue:
                if residue.id[1] < first_res[chain.id]:
                    first_res[chain.id] = residue.id[1]
                atom_coord = atom.coord
                distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                if np.any(distances < distance_cutoff):
                    pocket_residues.add((chain.id, residue.id[1], residue.resname))

    # Sort and convert to sequence
    sorted_residues = sorted(pocket_residues, key=lambda x: (x[0], x[1]))
    sequence = ''.join([seq1(resname) for _, _, resname in sorted_residues])

    # Print results
    for chain, res_id, resname in sorted_residues:
        print(f"Chain {chain}, Residue {resname} ({seq1(resname)}), Position {res_id - first_res[chain]}")

    print("\nPocket Sequence:")
    print(sequence)

    return sequence


def similar(a, b, threshold=0.9):
    return SequenceMatcher(None, a, b).ratio() >= threshold

def group_similar_sequences(original_dct, pocket_mask_dct, functional_group_mask):
    chain_pocket = [chain for chain, seq in pocket_mask_dct.items() if "@" in seq]
    visited = set()
    groups = []
    for chainA, seqA in original_dct.items():
        if chainA in visited:
            continue
        group = [chainA]
        visited.add(chainA)
        for chainB, seqB in original_dct.items():
            if chainB in visited:
                continue
            if similar(seqA, seqB):
                group.append(chainB)
                visited.add(chainB)
        groups.append(group)
    keep_lst = [] 
    for element in groups:
        if len(element) == 0:
            keep_lst.append(element[0])
        else:
            for pocket in chain_pocket:
                if pocket in element:
                    keep_lst.append(pocket)
    seq_input_concat = []
    seq_pocket_concat = []
    funtional_drug_concat = []
    for key, value in original_dct.items():
        if key in keep_lst:
            seq_input_concat.append(value)
            seq_pocket_concat.append(pocket_mask_dct[key])
            if len(funtional_drug_concat) > 0:
                funtional_drug_concat += [set()] 
            funtional_drug_concat += functional_group_mask[key]
    seq_concat = '|'.join(seq_input_concat)
    seq_pocket_concat = '|'.join(seq_pocket_concat) 
    return seq_concat, seq_pocket_concat,funtional_drug_concat




# Hàm lấy danh sách functional group theo vị trí nguyên tử
def get_functional_group_list_by_atom(atom_number, df):
    atom_str = str(atom_number)
    filtered = df[df["Atom Position"].str.contains(rf'\b{atom_str}\b')]
    cleaned_list = [item.strip() for item in filtered["Functional Group"].tolist()]
    cleaned_list = [item for item in cleaned_list if "alkyne" not in item.lower()]
    if len(cleaned_list) > 0:
        return [cleaned_list[-1]]
    return []

def get_seq_str(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    header_lines = []
    for l in lines:
        if l.startswith('SEQRES'):
            header_lines.append(l)
    seqs_lines = {}
    chain_id = []
    for hl in header_lines:
        if not hl[11] in chain_id:
            chain_id.append(hl[11])
            seqs_lines[hl[11]] = []
            seqs_lines[hl[11]].append(hl)
        else:
            seqs_lines[hl[11]].append(hl)
    seqs = {}
    for key in seqs_lines:
        seqs[key] = ''
        for l in seqs_lines[key]:
            resl = l[18:].split(' ')
            for res in resl:
                res = res.strip()
                if len(res) == 3:
                    seqs[key] += get_res_short(res)
    # seqs_str = ''
    # for key in seqs:
    #     seqs_str+=seqs[key]
    #     seqs_str+=':'
    # seqs_str = seqs_str[:-2]
    return seqs


def fill_empty_sets(functional_group_mask: dict, index_list: list, keys: str):
    d = functional_group_mask[keys]
    n = len(d)
    for idx in index_list:
        # Bỏ qua nếu đã có giá trị
        if d[idx]:
            continue

        # Tìm gần nhất bên trái
        left = idx - 1
        while left >= 0 and len(d[left]) == 0:
            left -= 1

        # Tìm gần nhất bên phải
        right = idx + 1
        while right < n and len(d[right]) == 0:
            right += 1

        # Chọn gần nhất giữa trái và phải
        left_dist = abs(idx - left) if left >= 0 and d[left] else float('inf')
        right_dist = abs(idx - right) if right < n and d[right] else float('inf')
        if left_dist <= right_dist and left_dist != float('inf'):
            d[idx] = d[left].copy()
        elif right_dist != float('inf'):
            d[idx] = d[right].copy()
    
    return d


def extract_interaction(pdb_file, sdf_file, distance_cutoff=5.0):
    # Parse protein structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    # Parse ligand using RDKit
    ligand = Chem.MolFromMol2File(sdf_file.replace('sdf', 'mol2'), removeHs=False)
    ligand_conf = ligand.GetConformer()
    ligand_coords = np.array([ligand_conf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())])
    pocket_residues = set()
    first_res = {}
    pocket_mask_dct = {}
    original_dct = {}
    functional_group_mask = {}
    cm = CheckMol()
    df = cm.functionalGroups(file=sdf_file, justFGcode=False,returnDataframe=True)
    

    for chain in model:
        first_res[chain.id] = 10e6
        pocket_mask_dct[chain.id] = []
        original_dct[chain.id] = []
        functional_group_mask[chain.id] = []
        for residue in chain:
            functional_group_in_res = []
            residue_name = seq1(residue.resname)
            if not is_aa(residue):
                continue
            original_dct[chain.id].append(residue_name)
            for atom in residue:
                if residue.id[1] < first_res[chain.id]:
                    first_res[chain.id] = residue.id[1]
                atom_coord = atom.coord
                distances = np.linalg.norm(ligand_coords - atom_coord, axis=1)
                if np.any(distances < distance_cutoff):
                    pocket_residues.add((chain.id, residue.id[1], residue.resname))
                    if '@' not in residue_name:
                        residue_name = '@'
                matching_indices = np.where(distances < distance_cutoff)[0] # index của ligand id, atom của protein
                if len(matching_indices) > 0:
                    for indices in matching_indices:
                        functional_group_lst = get_functional_group_list_by_atom(indices, df)
                        if len(functional_group_lst) > 0:
                            # print("Residue name: ", residue.resname, ' , functional group: ', functional_group_lst)
                            functional_group_in_res += functional_group_lst
            functional_group_in_res = set(functional_group_in_res)
            
            # for functional_group in functional_group_in_res:
            #     res_functional_interaction_dict[residue.resname][int(name_to_number[functional_group])] += 1
            functional_group_mask[chain.id].append(functional_group_in_res)
            pocket_mask_dct[chain.id].append(residue_name)
            
        # fill_empty_sets(functional_group_mask, [48,52,55,56,59,60,63,74,77,78,81,82,91,92,94,95,96,98,99,102,118,122], 'A')
            
        original_dct[chain.id] = ''.join(original_dct[chain.id])
        pocket_mask_dct[chain.id] = ''.join(pocket_mask_dct[chain.id])
    # Sort and convert to sequence
    sorted_residues = sorted(pocket_residues, key=lambda x: (x[0], x[1]))
    sequence = ''.join([seq1(resname) for _, _, resname in sorted_residues])
    
    
    pocket_dct = {}
    # print(pocket_mask_dct)
    for chain, res_lst in pocket_mask_dct.items():
        pocket_dct[chain] = [i for i, c in enumerate(res_lst) if c == '@']
            
    # print(pocket_dct)
    for chain,res_lst in pocket_dct.items():
        functional_group_mask[chain] = fill_empty_sets(functional_group_mask, res_lst, chain)
        
    seq_concat, seq_pocket_concat,funtional_drug_concat = group_similar_sequences(original_dct, pocket_mask_dct, functional_group_mask)
    return seq_concat, seq_pocket_concat,funtional_drug_concat 

def extract_gt_attention(seq_pocket_concat, funtional_drug_concat):
    res_pocket = [i for i, c in enumerate(seq_pocket_concat) if c == '@']
    fg_to_residues = defaultdict(list)
    lst_res_fg = []
    lst_fg_res = []
    for k in res_pocket:
        fg_numbers = [name_to_number[name] for name in funtional_drug_concat[k]]
        for fg in fg_numbers:
            fg_to_residues[fg].append(k)
            lst_res_fg.append((k, fg))
            lst_fg_res.append((fg,k))
    return lst_res_fg,lst_fg_res