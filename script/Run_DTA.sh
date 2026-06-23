python run/run_DTA.py \
        --exp_name functional_group_7_gcn \
        --num_seeds 42 \
        --csv_path data/Davis/Davis_preprocessed.csv\
        --protein_emb_path datapreprocessed/DavisFeature/protein_embeddings.pkl \
        --fg_instance_path datapreprocessed/DavisFeature/fg_instance.pkl \
        --ligand_graph_path datapreprocessed/DavisFeature/ligand_graph.pkl \
        --batch_size 64