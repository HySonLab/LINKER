python preprocessing/PDBBindProcessor.py \
    --complex_metadata ./data/LP_PDBBind/complex_metadata_pdb_2020_general.csv \
    --ligand_metadata ./data/LP_PDBBind/ligand_metadata_pdb_2020_general.csv \
    --protein_metadata ./data/LP_PDBBind/protein_metadata_pdb_2020_general.csv \
    --main_dataset ./data/LP_PDBBind/LP_PDBBind.csv \
    --ligand_folder ./data/LP_PDBBind/ligands \
    --complex_folder ./data/LP_PDBBind/complexes \
    --output_csv ./data/LP_PDBBind/LP_PDBBind_Merged.csv \
    --pkl_folder ./data/PDBBind_PKL
