python featurizer/drug_featurizer.py \
        --input_dir data \
        --output_dir datapreprocessed/DavisFeature \
        --data_name Davis

python featurizer/protein_featurizer.py \
        --input_dir data \
        --data_name Davis \
        --output_dir datapreprocessed/DavisFeature 

