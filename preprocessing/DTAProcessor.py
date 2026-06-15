import argparse
import pandas as pd
import json
import os
from difflib import get_close_matches
from rdkit import Chem
import ast
from glob import glob
import torch
import os
import pickle

class DTAPreprocessing:
    def __init__(self, data_name, data_path="data"):
        self.data_name = data_name
        self.base_path = f"{data_path}/{data_name}"
        
        self.df = None
        self.protein_dict = None
        self.keys = None

    # -------- Load data --------
    def load_data(self):
        print("Load data !")
        self.df = pd.read_csv(f"{self.base_path}/full.csv")
        
        with open(f"{self.base_path}/proteins.txt", "r") as f:
            self.protein_dict = json.load(f)
        
        self.keys = list(self.protein_dict.keys())

    # -------- Fuzzy matching --------
    def match_protein(self, name, cutoff=0.0):
        matches = get_close_matches(name, self.keys, n=1, cutoff=cutoff)
        print(f"Matching protein: {name}", "->", matches[0] if matches else "No match")
        return matches[0] if matches else None

    # -------- Map protein sequence --------
    def map_proteins(self):
        print("Map protein sequences !")
        # fuzzy match
        self.df["protein_matched"] = self.df["protein"].apply(
            lambda x: self.match_protein(x)
        )
        # map sequence
        self.df["protein_seq"] = self.df["protein_matched"].map(self.protein_dict)

    # -------- Map ligand & protein ID --------
    def map_ids(self):
        print("Map ids !")
        unique_ligands  = self.df["ligand"].unique()
        unique_proteins = self.df["protein"].unique()

        ligand_map  = {lig: f"L{i+1}" for i, lig in enumerate(unique_ligands)}
        protein_map = {pro: f"P{i+1}" for i, pro in enumerate(unique_proteins)}

        self.df["ligand_id"]  = self.df["ligand"].map(ligand_map)
        self.df["protein_id"] = self.df["protein"].map(protein_map)

    # -------- Save --------
    def save(self):
        self.df.to_csv(f"{self.base_path}/{self.data_name}_preprocessed.csv", index=False)

    # -------- Debug --------
    def check_missing(self):
        missing = self.df["protein_seq"].isna().sum()
        print("Missing sequences:", missing)

        if missing > 0:
            print(self.df[self.df["protein_seq"].isna()][
                ["protein", "protein_clean", "protein_matched"]
            ].head())

    def drug_to_sdf(self):
        print("Convert drugs to SDF format !")
        df           = pd.read_csv(f"{self.base_path}/{self.data_name}_preprocessed.csv")
        ligand_pairs = df[["ligand", "ligand_id"]].drop_duplicates()
        os.makedirs(f"data/{self.data_name}/drug_mol", exist_ok=True)
        for idx, row in ligand_pairs.iterrows():
            print(f"Processing ligand: {row['ligand']} with ID: {row['ligand_id']}")
            ligand_id = str(row['ligand_id'])
            if os.path.exists(f"data/{self.data_name}/drug_mol/{ligand_id}.sdf"):
                print(f"File for {ligand_id} already exists, skipping.")
                continue
            writer = Chem.SDWriter(f"data/{self.data_name}/drug_mol/{ligand_id}.sdf")
            mol = Chem.MolFromSmiles(row['ligand'])
            if mol:
                writer.write(mol)
            writer.close()
    
    def split_data(self):
        self.df = pd.read_csv(f"{self.base_path}/{self.data_name}_preprocessed.csv")
        with open(f"data/{self.data_name}/train_folds.txt", "r", encoding="utf-8") as f:
            content = f.read()

        train_folds = ast.literal_eval(content)

        with open(f"data/{self.data_name}/test_fold.txt", "r", encoding="utf-8") as f:
            content = f.read()

        test_fold = ast.literal_eval(content)

        
        # Initialize split
        self.df["split"] = None

        # Train folds
        for i, fold in enumerate(train_folds):
            self.df.loc[fold, "split"] = f"Train_{i+1}"

        # Test
        self.df.loc[test_fold, "split"] = "Test"
        
        self.df.to_csv(f"{self.base_path}/{self.data_name}_preprocessed.csv")

    def save_pickle(self):
        paths = glob(f'datapreprocessed/{self.data_name}Feature/ligand_graph/*')
        dictionary = {}
        for path in paths:
            name = os.path.split(path)[1].split('.')[0]
            dictionary[name] = torch.load(path, weights_only=False)
        with open(f"datapreprocessed/{self.data_name}Feature/ligand_graph.pkl", "wb") as f:
            pickle.dump(dictionary, f)

        paths = glob(f'datapreprocessed/{self.data_name}Feature/protein_embeddings/*')
        dictionary = {}
        for path in paths:
            name = os.path.split(path)[1].split('.')[0]
            dictionary[name] = torch.load(path, weights_only=False)
        with open(f"datapreprocessed/{self.data_name}Feature/protein_embeddings.pkl", "wb") as f:
            pickle.dump(dictionary, f)

        paths = glob(f'datapreprocessed/{self.data_name}Feature/fg_instance/*')
        dictionary = {}
        for path in paths:
            name = os.path.split(path)[1].split('.')[0]
            dictionary[name] = pickle.load(open(path, 'rb'))
        with open(f"datapreprocessed/{self.data_name}Feature/fg_instance.pkl", "wb") as f:
            pickle.dump(dictionary, f)


    # -------- Run full pipeline --------
    def run(self):
        self.load_data()
        self.map_proteins()
        self.map_ids()
        self.check_missing()
        self.save()
        self.drug_to_sdf()


def main():
    parser = argparse.ArgumentParser(description="Preprocess DTA dataset and save CSV")

    parser.add_argument('--data_name', type=str, required=True,
                        help="Dataset name (e.g., Davis)")

    parser.add_argument('--data_path', type=str, default="data",
                        help="Base data folder")

    args = parser.parse_args()

    preprocessor = DTAPreprocessing(
        data_name=args.data_name,
        data_path=args.data_path
    )

    # preprocessor.load_data()
    # preprocessor.map_proteins()
    # preprocessor.map_ids()
    # preprocessor.check_missing()
    # preprocessor.save()
    # preprocessor.drug_to_sdf()
    # preprocessor.split_data()
    # preprocessor.save_pickle()
    preprocessor.run()


if __name__ == "__main__":
    main()
