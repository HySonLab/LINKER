python featurizer/drug_featurizer.py \
        --input_dir data/PDBBind_PKL \
        --output_dir datapreprocessed/PDBBindFeature 

python featurizer/protein_featurizer.py \
        --input_dir data/PDBBind_PKL \
        --output_dir datapreprocessed/PDBBindFeature 

