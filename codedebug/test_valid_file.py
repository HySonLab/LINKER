import numpy as np
import torch
from glob import glob
import ast  
import matplotlib.pyplot as plt
import sys
import argparse
import warnings
import pickle
import pandas as pd
import json
from difflib import get_close_matches
import pickle as pkl
from glob import glob
from collections import defaultdict
import ast
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from rdkit import Chem
from openbabel import openbabel as ob
from pymol import cmd
from plip.structure.preparation import PDBComplex
warnings.filterwarnings("ignore")
cmd.feedback("disable", "all", "everything")

handler = ob.OBMessageHandler()
handler.SetOutputLevel(0)
import sys
sys.path.append('../')
from utils import *

for file in glob('data/Davis/drug_mol/*'):
    try:
        mol = Chem.MolFromMolFile(file)
        if mol is None:
            print(f"Invalid molecule file: {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/DavisFeature/fg_instance/*'):
    try:
        with open(file, 'rb') as f:
            instance = pkl.load(f)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/DavisFeature/fg_instance/*'):
    try:
        with open(file, 'rb') as f:
            instance = pkl.load(f)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/DavisFeature/ligand_graph/*'):
    try:
        print(f"Processing ligand graph file: {file}")
        emb = torch.load(file, weights_only=False)
        print(f"Successfully loaded ligand graph: {file} | Graph data: {emb}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/DavisFeature/protein_embeddings/*'):
    try:
        print(f"Processing ligand graph file: {file}")
        emb = torch.load(file, weights_only=False)
        print(f"Successfully loaded ligand graph: {file} | Graph data: {emb}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break


for file in glob('data/KIBA/drug_mol/*'):
    try:
        mol = Chem.MolFromMolFile(file)
        if mol is None:
            print(f"Invalid molecule file: {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/KIBAFeature/fg_instance/*'):
    try:
        with open(file, 'rb') as f:
            instance = pkl.load(f)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break

for file in glob('datapreprocessed/KIBAFeature/fg_instance/*'):
    try:
        with open(file, 'rb') as f:
            instance = pkl.load(f)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/KIBAFeature/ligand_graph/*'):
    try:
        print(f"Processing ligand graph file: {file}")
        emb = torch.load(file, weights_only=False)
        print(f"Successfully loaded ligand graph: {file} | Graph data: {emb}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
for file in glob('datapreprocessed/KIBAFeature/protein_embeddings/*'):
    try:
        print(f"Processing ligand graph file: {file}")
        emb = torch.load(file, weights_only=False)
        print(f"Successfully loaded ligand graph: {file} | Graph data: {emb}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        break
