python featurizer/drug_featurizer.py \
        --input_dir data/BindingDB_PKL \
        --output_dir datapreprocessed/BindingDBFeature \
        --data_name BindingDB 

python featurizer/protein_featurizer.py \
        --input_dir data/BindingDB_PKL \
        --output_dir datapreprocessed/BindingDBFeature \
        --data_name BindingDB 

