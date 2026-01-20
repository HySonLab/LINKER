python run/run_save_state.py \
        --csv_path datapreprocessed/BindingDBFeature/Clean_BindingDB.csv \
        --protein_emb_path datapreprocessed/BindingDBFeature/protein_embeddings \
        --fg_instance_path datapreprocessed/BindingDBFeature/fg_instance \
        --ligand_graph_path datapreprocessed/BindingDBFeature/ligand_graph \
        --label_path  datapreprocessed/BindingDBFeature/label \
        --checkpoint  checkpoints/best_model.pt \
        --save_dir evaluation/extracted_features

python run/run_Predictor.py \
        --base_path evaluation/extracted_features \
        --lambda_value 5