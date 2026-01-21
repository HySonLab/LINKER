# python  preprocessing/BindingDB_split.py \
#         --csv data/BindingDB_Merged.csv \
#         --pkl_path data/BindingDB_PKL \
#         --ligand_folder data/BindingDB_Preprocessed \
#         --output datapreprocessed/BindingDBFeature/Clean_BindingDB.csv

# python dataloader/dataloader.py \
#         --csv_path datapreprocessed/BindingDBFeature/Clean_BindingDB.csv \
#         --protein_emb_path datapreprocessed/BindingDBFeature/protein_embeddings \
#         --fg_instance_path datapreprocessed/BindingDBFeature/fg_instance \
#         --ligand_graph_path datapreprocessed/BindingDBFeature/ligand_graph \
#         --label_path  datapreprocessed/BindingDBFeature/label \
#         --batch_size 4

python dataloader/dataloader.py \
        --csv_path ./data/LP_PDBBind/LP_PDBBind_Merged.csv \
        --protein_emb_path datapreprocessed/PDBBindFeature/protein_embeddings \
        --fg_instance_path datapreprocessed/PDBBindFeature/fg_instance \
        --ligand_graph_path datapreprocessed/PDBBindFeature/ligand_graph \
        --label_path  datapreprocessed/PDBBindFeature/label \
        --batch_size 4