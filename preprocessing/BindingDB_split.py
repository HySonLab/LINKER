import argparse
import os
import pandas as pd
import random
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity
import networkx as nx
from Bio import pairwise2
from Bio.Align import substitution_matrices

# --- Hàm tính Tanimoto ---
def tanimoto_calc(path1, path2):
    supplier1 = Chem.SDMolSupplier(path1, sanitize=False)
    supplier2 = Chem.SDMolSupplier(path2, sanitize=False)
    mol1 = supplier1[0]
    mol2 = supplier2[0]
    gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp1 = gen.GetFingerprint(mol1)
    fp2 = gen.GetFingerprint(mol2)
    return TanimotoSimilarity(fp1, fp2)

# --- Hàm tính sequence identity ---
blosum62 = substitution_matrices.load("BLOSUM62")
def seq_identity(pair):
    id1, seq1, id2, seq2 = pair
    aln = pairwise2.align.globalds(
        seq1, seq2,
        blosum62,
        -10, -0.5,
        one_alignment_only=True
    )[0]
    matches = sum(a==b and a!='-' for a,b in zip(aln.seqA, aln.seqB))
    identity = matches / len(aln.seqA)
    return {"ID1": id1,"ID2": id2,"identity": identity}

# --- Hàm split cluster ---
def cluster_split(df_sequences, df_sim, threshold=0.5):
    df_filtered = df_sim[df_sim['identity']>threshold]
    G = nx.Graph()
    ids_with_sequence = set(df_sequences['ID'])
    G.add_nodes_from(ids_with_sequence)
    edges = df_filtered[['ID1','ID2']].values
    G.add_edges_from(edges)
    clusters = list(nx.connected_components(G))
    all_ids = set(df_sim['ID1']).union(df_sim['ID2'])
    ids_without_sequence = all_ids - ids_with_sequence
    for id in ids_without_sequence:
        clusters.append({id})
    df_clusters = pd.DataFrame({
        'cluster_id': range(len(clusters)),
        'members':[list(c) for c in clusters]
    })
    return df_clusters

# --- Hàm assign split ---
def assign_split(protein, train_ids, val_ids, test_ids):
    protein = protein.split('_')[0]
    if protein in train_ids:
        return "train"
    elif protein in val_ids:
        return "val"
    elif protein in test_ids:
        return "test"
    else:
        return "unknown"

def extract_id(url):
    filename = url.split('/')[-1]           # "1AZ1-results_16415.mol2"
    protein, ligand = filename.split('-results_')
    ligand = ligand.replace('.mol2','')
    return f"{protein}_{ligand}"
    
# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to BindingDB CSV file')
    parser.add_argument('--ligand_folder', type=str, required=True, help='Folder containing ligand sdf files')
    parser.add_argument('--pkl_path', type=str, required=True, help='Folder containing pkl files')
    parser.add_argument('--output', type=str, required=True, help='Output folder for cleaned CSV')
    parser.add_argument('--cpu', type=int, default=12, help='Number of CPUs for parallel computation')
    parser.add_argument('--seq_threshold', type=float, default=0.5, help='Sequence identity threshold for clustering')
    parser.add_argument('--tanimoto_threshold', type=float, default=0.8, help='Tanimoto similarity threshold to remove train proteins')
    args = parser.parse_args()

    # --- Read CSV ---
    df = pd.read_csv(args.csv)

    df['protein'] = df['Structure File'].apply(extract_id)
    seq_list = []
    for f in df['protein']:
        path_pkl = os.path.join(args.pkl_path, f + '.pkl')
        df_temp = pd.read_pickle(path_pkl)
        seq_list.append(df_temp['protein_sequence'])
    df['seq'] = seq_list
    df['prefix'] = df['protein'].apply(lambda x: x.split('_')[0])
    prefix_to_sequence = df.groupby('prefix')['seq'].first().to_dict()
    df_sequences = pd.DataFrame(list(prefix_to_sequence.items()), columns=["ID","sequence"])

    # --- Compute pairwise sequence similarity ---
    from itertools import combinations
    pairs = [(r1.ID,r1.sequence,r2.ID,r2.sequence) for r1,r2 in combinations(df_sequences.itertuples(index=False),2)]
    from multiprocessing import Pool
    from tqdm import tqdm
    with Pool(args.cpu) as pool:
        results = list(tqdm(pool.imap(seq_identity, pairs), total=len(pairs)))
    df_sim = pd.DataFrame(results)

    # --- Cluster and split ---
    df_clusters = cluster_split(df_sequences, df_sim, threshold=args.seq_threshold)
    clusters = df_clusters['members'].tolist()
    random.shuffle(clusters)
    test_ratio = 0.2
    train_val_ratio = 0.8
    train_ratio = 0.75
    n_test_clusters = int(len(clusters)*test_ratio)
    test_clusters = clusters[:n_test_clusters]
    train_val_clusters = clusters[n_test_clusters:]
    test_ids = [i for c in test_clusters for i in c]
    train_val_ids = [i for c in train_val_clusters for i in c]
    random.shuffle(train_val_ids)
    n_train = int(len(train_val_ids)*train_ratio)
    train_ids = train_val_ids[:n_train]
    val_ids = train_val_ids[n_train:]
    df['new_split'] = df['protein'].apply(lambda x: assign_split(x, train_ids, val_ids, test_ids))

    # --- Remove train proteins with high Tanimoto similarity to test ---
    remove_train = []
    train_list = df[df['new_split']=='train']['protein'].tolist()
    test_list = df[df['new_split']=='test']['protein'].tolist()
    for train_sample in train_list:
        for test_sample in tqdm(test_list):
            try:
                train_ligand = os.path.join(args.ligand_folder, f'{train_sample}/{train_sample}_ligand.sdf')
                test_ligand = os.path.join(args.ligand_folder, f'{test_sample}/{test_sample}_ligand.sdf')
                if not os.path.exists(train_ligand) or not os.path.exists(test_ligand):
                    continue
                tanimoto = tanimoto_calc(train_ligand,test_ligand)
                if tanimoto > args.tanimoto_threshold:
                    remove_train.append(train_sample)
                    break
            except:
                continue
    df_clean = (df[~df['protein'].isin(remove_train)]
                [['protein', 'new_split', 'Surflex Score', 'seq']]
                .rename(columns={'Surflex Score': 'value'}))
    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    df_clean.to_csv(args.output,index=False)
    print(f"Saved cleaned CSV to {args.output}")
