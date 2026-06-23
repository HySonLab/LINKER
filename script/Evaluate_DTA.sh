python evaluation/evaluation_dta.py \
        --num_seeds 42 \
        --exp_name davis \
        --checkpoint_dir /home/phuc.phamhuythienai@gmail.com/Desktop/LINKER_ABLATION/logs_final/functional_group_7_gcn_20260620_233814_dta_42 \
        --csv_path data/Davis/Davis_preprocessed.csv\
        --protein_emb_path datapreprocessed/DavisFeature/protein_embeddings.pkl \
        --fg_instance_path datapreprocessed/DavisFeature/fg_instance.pkl \
        --ligand_graph_path datapreprocessed/DavisFeature/ligand_graph.pkl \
        --batch_size 64