# ============================================================
# Imports & global config
# ============================================================
import os
import shutil
import pickle
import warnings
from glob import glob
from typing import Dict

import pandas as pd
from pymol import cmd
from rdkit import Chem
from openbabel import openbabel as ob
from plip.structure.preparation import PDBComplex
                
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

cmd.feedback("disable", "all", "everything")
ob_handler = ob.OBMessageHandler()
ob_handler.SetOutputLevel(0)


# ============================================================
# BindingDB Processor
# ============================================================
class BindingDBProcessor:
    """
    Preprocess BindingDB protein–ligand–pocket structures
    into clean, indexed complexes.
    """

    RESIDUE_MAP = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
        "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
        "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
        "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    # ========================================================
    # Init
    # ========================================================
    def __init__(
        self,
        raw_root: str,
        output_root: str,
        pkl_path: str,
        merged_csv: str

    ):
        self.raw_root = raw_root
        self.output_root = output_root
        self.merged_csv = merged_csv
        self.prepare_merged_csv(self.merged_csv) 
        os.makedirs(self.output_root, exist_ok=True)
        self.df = pd.read_csv(self.merged_csv)
        self.pkl_path = pkl_path

    # ========================================================
    # Utilities
    # ========================================================
    @staticmethod
    def sorted_pdb(input_file: str, output_file: str) -> None:
        atoms, connects = [], []

        with open(input_file) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    atoms.append((
                        int(line[6:11]),
                        int(line[22:26]),
                        line[17:20],
                        line[21],
                        line,
                    ))
                elif line.startswith("CONECT"):
                    connects.append(line)

        # residue mapping
        res_order = []
        for _, resid, _, chain, _ in atoms:
            key = (chain, resid)
            if key not in res_order:
                res_order.append(key)
        res_map = {old: new for new, old in enumerate(res_order)}

        old2new, new_lines = {}, []
        for new_idx, (old_idx, resid, _, chain, line) in enumerate(atoms):
            old2new[old_idx] = new_idx
            new_resid = res_map[(chain, resid)]
            new_lines.append(
                line[:6] + f"{new_idx:>5}" +
                line[11:22] + f"{new_resid:>4}" +
                line[26:]
            )

        new_connects = []
        for line in connects:
            ids = [old2new[int(x)] for x in line.split()[1:] if int(x) in old2new]
            if ids:
                new_connects.append(
                    "CONECT" + "".join(f"{i:>5}" for i in ids) + "\n"
                )

        with open(output_file, "w") as f:
            f.writelines(new_lines + new_connects)

    def pdb_to_sequence(self, pdb_file: str) -> str:
        seq, seen = [], set()

        with open(pdb_file) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    resname = line[17:20].strip()
                    key = (line[21], line[22:26])
                    if key not in seen and resname in self.RESIDUE_MAP:
                        seen.add(key)
                        seq.append(self.RESIDUE_MAP[resname])
        return "".join(seq)

    @staticmethod
    def run(cmdline: str) -> None:
        os.system(cmdline)

    # ========================================================
    # Dataset preparation
    # ========================================================
    def prepare_merged_csv(self, save_path: str) -> None:
        files = []
        for path in glob(f"{self.raw_root}/*"):
            files.extend(glob(f"{path}/*.csv"))

        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)[:-4]
            dfs.append(df)

        merged = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
        merged.to_csv(save_path, index=False)

    # ========================================================
    # Main processing
    # ========================================================
    def process_one(self, row: pd.Series) -> Dict:
        structure_file      = row['Structure File']
        source_file         = row['source_file']
        ligand_file         = os.path.split(structure_file)[1]
        folder_file         = structure_file.split('/')[5]
        ligand_new_folder   = ligand_file.replace('-results', '')[:-5]
        new_folder          = os.path.join(self.output_root,ligand_new_folder)
        os.makedirs(new_folder, exist_ok=True)
        
        old_ligand_path     = os.path.join(self.raw_root,folder_file,ligand_file)
        old_protein_path    = os.path.join(self.raw_root,folder_file, source_file + '.mol2')
        ligand_file_mol2 =  os.path.join(new_folder, ligand_new_folder + '_ligand.mol2')
        protein_file_mol2 =  os.path.join(new_folder, ligand_new_folder + '_protein.mol2')
        shutil.copy(old_ligand_path, os.path.join(new_folder, ligand_new_folder + '_ligand.mol2'))
        shutil.copy(old_protein_path, os.path.join(new_folder, ligand_new_folder + '_protein.mol2'))
        ligand_pdb = ligand_file_mol2.replace('mol2', 'pdb')
        ligand_sdf = ligand_file_mol2.replace('mol2', 'sdf')
        protein_pdb = protein_file_mol2.replace('mol2', 'pdb')
        protein_name = os.path.split(protein_pdb)[1].replace('_protein.pdb','')

        complex_pdb = os.path.join(new_folder, 'complex.pdb')
        sorted_pdb_file = protein_pdb.replace('.pdb', 'sorted.pdb')
        
        os.system(f'obabel {ligand_file_mol2} -O {ligand_pdb}')
        os.system(f'obabel {ligand_file_mol2} -O {ligand_sdf}')
        os.system(f'obabel {protein_file_mol2} -O {protein_pdb}')
        self.sorted_pdb(protein_pdb, sorted_pdb_file)
        os.system(f'cat {sorted_pdb_file} {ligand_pdb} > {complex_pdb}')
        protein_sequence = self.pdb_to_sequence(sorted_pdb_file)

        protein_file_pre  = sorted_pdb_file
        ligand_file_pre   = ligand_pdb.replace('pdb', 'mol2')
        complex_file_pre  = complex_pdb.replace('.pdb', 'pymol.pdb')
        cmd.reinitialize()
        cmd.load(protein_file_pre, "prot")
        cmd.load(ligand_file_pre, "ligand")
        cmd.create("complex", "ligand, prot")
        cmd.save(complex_file_pre, "complex")
        
        mol = PDBComplex()
        mol.load_pdb(complex_file_pre)
        mol.analyze()
        longnames = [x.longname for x in mol.ligands]
        bsids = [":".join([x.hetid, x.chain, str(x.position)]) for x in mol.ligands]

        indices = [j for j,x in enumerate(longnames) if x == 'LIG']
        for idx in indices:
            bsid = bsids[idx]
            interactions = mol.interaction_sets[bsid]
            
        tuples = []
        for inter in interactions.all_itypes:
            protein_id = f"{inter.restype}:{inter.resnr}:{inter.reschain}"
   
            if inter.__class__.__name__ == 'hbond':
                ligand_id = f"LIG_atom:{inter.a_orig_idx}" if inter.protisdon else f"LIG_atom:{inter.d_orig_idx}"
            
            elif inter.__class__.__name__ == 'pistack':
                ligand_atoms = [atom_idx for atom_idx in inter.ligandring.atoms_orig_idx]
                ligand_id = f"LIG_ring_atoms:{ligand_atoms}"
                
            elif inter.__class__.__name__ == 'hydroph_interaction':
                ligand_id = f"LIG_atom:{inter.ligatom_orig_idx}"
                
            elif inter.__class__.__name__ == 'pication':
                if 'fgroup' in dir(inter.charge):
                    ligand_atom_indices = inter.charge.atoms_orig_idx
                else:    
                    ligand_atom_indices = inter.ring.atoms_orig_idx
                ligand_id = f"LIG_ring_atoms:{ligand_atom_indices}"
                
            elif inter.__class__.__name__ == 'halogenbond': 
                ligand_atom_indices = inter.don_orig_idx
                ligand_id = f"LIG_ring_atoms:{ligand_atom_indices}"
            
            elif inter.__class__.__name__ == 'saltbridge':
                if 'fgroup' in dir(inter.negative):
                    ligand_atom_indices = inter.negative.atoms_orig_idx
                elif 'fgroup' in dir(inter.positive):
                    ligand_atom_indices = inter.positive.atoms_orig_idx
                ligand_id = f"LIG_atom:{ligand_atom_indices}"
                
            else:
                print()
                print()
                print(inter.__class__.__name__)
                print(inter)
                ligand_id = "LIG_unknown"
                exit()
            tuples.append((protein_id, str(inter.__class__.__name__) + '-' + ligand_id))

        mol = Chem.MolFromMol2File(ligand_file_mol2, sanitize=True)

        dictionary = {
            'protein_sequence': protein_sequence,
            'mol': mol,
            'interactions': tuples,
            'protein path': protein_pdb
        }
        

        os.makedirs(self.pkl_path, exist_ok=True)
        file_path = os.path.join(self.pkl_path, f'{protein_name}.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(dictionary, f)
        
        
    def process_all(self):
        for idx, row in self.df.iterrows():
            try:
                print(f"[{idx}] Processing {row['source_file']}")
                self.process_one(row)
            except Exception as e:
                print(f"[ERROR] {idx}: {e}")
                
    # ========================================================
    # Save processed data to pickle
    # ========================================================
    def save_processed_data(
        self,
        protein_name: str,
        protein_sequence: str,
        mol: Chem.Mol,
        interactions: list,
        protein_pdb: str
    ):
        dictionary = {
            "protein_sequence": protein_sequence,
            "mol": mol,
            "interactions": interactions,
            "protein_path": protein_pdb,
        }

        path = os.path.join(self.output_root, "pkl")
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{protein_name}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(dictionary, f)

        print(f"[Saved] {file_path}")      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BindingDB preprocessing pipeline"
    )

    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Path to raw BindingDB directory"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to output processed BindingDB directory"
    )
    
    parser.add_argument(
        "--pkl_path",
        type=str,
        required=True,
        help="Path to pkl preprocessing file"
    )

    parser.add_argument(
        "--merged_csv",
        type=str,
        required=True,
        help="Path to merged.csv file"
    )

    args = parser.parse_args()

    processor = BindingDBProcessor(
        raw_root        =   args.raw_root,
        output_root     =   args.output_root,
        pkl_path        =   args.pkl_path,
        merged_csv      =   args.merged_csv
    )

    processor.process_all()
