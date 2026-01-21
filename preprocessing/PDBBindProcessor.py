import os
import argparse
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from tqdm import tqdm
class PDBBindPreprocessor:
    def __init__(self, complex_metadata, ligand_metadata, protein_metadata,
                 main_dataset, ligand_folder, complex_folder, output_csv, pkl_folder):
        self.complex_metadata = complex_metadata
        self.ligand_metadata = ligand_metadata
        self.protein_metadata = protein_metadata
        self.main_dataset = main_dataset
        self.ligand_folder = ligand_folder
        self.complex_folder = complex_folder
        self.output_csv = output_csv
        self.pkl_folder = pkl_folder
        os.makedirs(self.pkl_folder, exist_ok=True)

    def merge_metadata(self):
        """Merge complex, ligand, protein metadata with main dataset"""
        df1 = pd.read_csv(self.complex_metadata)
        df2 = pd.read_csv(self.ligand_metadata)
        df3 = pd.read_csv(self.protein_metadata)
        
        df_merged = df1.merge(df2, on="ligand_id", how="left")
        df_merged = df_merged.merge(df3, on="protein_id", how="left")
        
        df_main = pd.read_csv(self.main_dataset)
        if "Unnamed: 0" in df_main.columns:
            df_main.rename(columns={"Unnamed: 0": "pdb_id"}, inplace=True)
        
        df_merged = df_main.merge(df_merged, on="pdb_id", how="left")
        selected_columns = ['complex_id', 'ligand_id', 'protein_id', 'pdb_id', 'fasta', 'new_split', 'value']
        df_merged = df_merged[selected_columns]
        df_merged = df_merged.dropna().reset_index(drop=True)
        if "pdb_id" in df_main.columns:
            df_merged.rename(columns={"pdb_id": "protein"}, inplace=True)
        df_merged.to_csv(self.output_csv, index=False)
        print(f"Merged dataset saved to {self.output_csv}")
        self.df_merged = df_merged

    def process_and_save_pickles(self):
        """Convert MOL files and save protein-ligand interaction data as pickle"""
        for i, row in tqdm(self.df_merged.iterrows(), total=len(self.df_merged), desc="Processing complexes"):
            protein_name = row['protein']
            protein_sequence = row['fasta']
            ligand_id = row['ligand_id']
            complex_id = row['complex_id']
            
            path_mol = os.path.join(self.ligand_folder, ligand_id, ligand_id + '.mol')
            path_complex = os.path.join(self.complex_folder, complex_id, complex_id + '.plip.npy')
            
            # Load mol object
            mol = pickle.load(open(path_mol, 'rb'))
            
            # Chuyển sang text format, tránh kekulize lỗi
            mol_block = Chem.MolToMolBlock(mol, kekulize=False)
            new_path = path_mol.replace('.mol', '_read.mol')
            with open(new_path, 'w') as f:
                f.write(mol_block)
            
            # Tạo dictionary
            dictionary = {
                'protein_sequence': protein_sequence,
                'mol_path': new_path,
                'interactions': np.load(path_complex)[:, :, :7],
                'protein_name': protein_name
            }
            
            file_path = os.path.join(self.pkl_folder, f'{protein_name}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(dictionary, f)
        print(f"All pickle files saved to {self.pkl_folder}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBBind dataset and save pickles")
    parser.add_argument('--complex_metadata', type=str, required=True)
    parser.add_argument('--ligand_metadata', type=str, required=True)
    parser.add_argument('--protein_metadata', type=str, required=True)
    parser.add_argument('--main_dataset', type=str, required=True)
    parser.add_argument('--ligand_folder', type=str, required=True)
    parser.add_argument('--complex_folder', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--pkl_folder', type=str, required=True)
    args = parser.parse_args()

    preprocessor = PDBBindPreprocessor(
        complex_metadata=args.complex_metadata,
        ligand_metadata=args.ligand_metadata,
        protein_metadata=args.protein_metadata,
        main_dataset=args.main_dataset,
        ligand_folder=args.ligand_folder,
        complex_folder=args.complex_folder,
        output_csv=args.output_csv,
        pkl_folder=args.pkl_folder
    )
    
    preprocessor.merge_metadata()
    preprocessor.process_and_save_pickles()


if __name__ == "__main__":
    main()