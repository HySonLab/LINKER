import os
import argparse
import pickle as pkl
from glob import glob
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd

class BindingDBProteinFeaturizer:
    def __init__(self, device="cuda"):
        """
        Args:
            device (str): "cuda" or "cpu"
        """
        self.device = device
        print("Loading ESMC model...")
        self.client = ESMC.from_pretrained("esmc_300m").to(self.device)
        print(f"Model loaded on {device}.")

    def process_protein(self, sample: dict, output_dir: str):
        """Encode a single protein sample and save embeddings."""
        protein_name = os.path.basename(sample['protein path']).replace('_protein.pdb','')
        seq = sample['protein_sequence']

        # Create output folder if not exist
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{protein_name}.pt")

        # ESM requirement: replace ':' with '|'
        seq = seq.replace(':', '|')
        print(f"Processing protein {protein_name} | Sequence length: {len(seq)}")

        protein = ESMProtein(sequence=seq)
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        embeddings = torch.tensor(logits_output.embeddings).squeeze(0)[1:-1]
        print(f"Embeddings shape: {embeddings.shape}")
        torch.save(embeddings, filepath)
        print(f"Saved embeddings to {filepath}\n")

    def process_all(self, pkl_dir: str, output_dir: str):
        """Process all .pkl files in a directory."""
        list_files = glob(os.path.join(pkl_dir, "*.pkl"))
        print(f"Found {len(list_files)} protein files to process.\n")

        for idx, file in enumerate(list_files, start=1):
            print(f"[{idx}/{len(list_files)}] Processing: {file}")
            sample = pkl.load(open(file, 'rb'))
            self.process_protein(sample, output_dir + '/protein_embeddings')



class PDBBindProteinFeaturizer:
    def __init__(self, device="cuda"):
        """
        Args:
            device (str): "cuda" or "cpu"
        """
        self.device = device
        print("Loading ESMC model...")
        self.client = ESMC.from_pretrained("esmc_300m").to(self.device)
        print(f"Model loaded on {device}.")

    def process_protein(self, sample: dict, output_dir: str):
        """Encode a single protein sample and save embeddings."""
        protein_name = sample['protein_name']
        seq = sample['protein_sequence']

        # Create output folder if not exist
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{protein_name}.pt")

        # ESM requirement: replace ':' with '|'
        seq = seq.replace(':', '|')
        print(f"Processing protein {protein_name} | Sequence length: {len(seq)}")

        protein = ESMProtein(sequence=seq)
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        embeddings = torch.tensor(logits_output.embeddings).squeeze(0)[1:-1]
        print(f"Embeddings shape: {embeddings.shape}")
        torch.save(embeddings, filepath)
        print(f"Saved embeddings to {filepath}\n")

    def process_all(self, pkl_dir: str, output_dir: str):
        """Process all .pkl files in a directory."""
        list_files = glob(os.path.join(pkl_dir, "*.pkl"))
        print(f"Found {len(list_files)} protein files to process.\n")

        for idx, file in enumerate(list_files, start=1):
            print(f"[{idx}/{len(list_files)}] Processing: {file}")
            sample = pkl.load(open(file, 'rb'))
            self.process_protein(sample, output_dir + '/protein_embeddings')




class DTAProteinFeaturizer:
    def __init__(self, device="cuda"):
        """
        Args:
            device (str): "cuda" or "cpu"
        """
        self.device = device
        print("Loading ESMC model...")
        self.client = ESMC.from_pretrained("esmc_300m").to(self.device)
        print(f"Model loaded on {device}.")

    def process_protein(self, sample: dict, output_dir: str):
        """Encode a single protein sample and save embeddings."""
        protein_id      = sample['protein_id']
        seq             = sample['protein_seq']

        # Create output folder if not exist
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{protein_id}.pt")
        if os.path.exists(filepath):
            print(f"File for {protein_id} already exists, skipping.")
            return
        # ESM requirement: replace ':' with '|'
        seq = seq.replace(':', '|')
        print(f"Processing protein {protein_id} | Sequence length: {len(seq)}")

        protein = ESMProtein(sequence=seq)
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )

        embeddings = torch.tensor(logits_output.embeddings).squeeze(0)[1:-1]
        print(f"Embeddings shape: {embeddings.shape}")
        torch.save(embeddings, filepath)
        print(f"Saved embeddings to {filepath}\n")

    def process_all(self, csv_file: str, output_dir: str):
        """Process all .pkl files in a directory."""
        df = pd.read_csv(csv_file)
        protein_pairs = df[["protein_id", "protein_seq"]].drop_duplicates()
        print(f"Found {len(protein_pairs)} proteins to process.\n")
        idx = 0
        for _, row in protein_pairs.iterrows():
            print(f"Processing: {row['protein_id']} ({idx}/{len(protein_pairs)})")
            self.process_protein(row, output_dir + '/protein_embeddings')
            idx += 1
            

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein embedding extraction using ESMC.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing BindingDB/PDBBind .pkl files or DTA directory with preprocessed CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save protein embeddings (.pt)')
    parser.add_argument('--data_name', type=str, required=False,
                        help='Dataset name (e.g., "Davis") - only for DTA featurizer')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on: "cuda" or "cpu"')
    args = parser.parse_args()
    
    if args.data_name == "BindingDB":
        featurizer = BindingDBProteinFeaturizer(device=args.device)
        featurizer.process_all(pkl_dir=args.input_dir, output_dir=args.output_dir)
    
    elif args.data_name == "PDBBind":
        featurizer = PDBBindProteinFeaturizer(device=args.device)
        featurizer.process_all(pkl_dir=args.input_dir, output_dir=args.output_dir)
    
    elif args.data_name == "Davis":
        featurizer      = DTAProteinFeaturizer(device=args.device)
        csv_file        = os.path.join(args.input_dir, args.data_name, f"{args.data_name}_preprocessed.csv")
        featurizer.process_all(csv_file=csv_file, output_dir=args.output_dir)
        

