
# LINKER: Learning Interactions Between Functional Groups and Residues with Chemical Knowledge-Enhanced Reasoning and Explainability

LINKER is a framework for modeling and explaining protein–ligand interactions by explicitly learning interactions between ligand functional groups and protein residues. The method integrates chemical knowledge, structural information, and deep learning to improve interpretability in structure-based drug discovery.


---

## LINKER Overview

<p align="center">
  <img src="assets/LINKER_architecture.png" width="700"/>
</p>


## Codeflow Overview

<p align="center">
  <img src="assets/LINKER_codeflow.png" width="700"/>
</p>

---

## Dependencies

LINKER relies on the following external tools and libraries:

- **PLIP** – Protein–Ligand Interaction Profiler  
- **pyCheckmol** – Functional group detection  
- **Open Babel (obabel)** – Molecular file conversion and processing  
- **ESMC** – Protein language model embeddings  

Please make sure these tools are installed and accessible in your environment before running the pipeline.

---

## Datasets

We use publicly available protein–ligand complex datasets:

- **BindingDB 3D Complexes**  
  https://www.bindingdb.org/rwd/data/surflex/surflex.tar

- **Leak-Proof PDBBind (LP-PDBBind)**  
  https://github.com/THGLab/LP-PDBBind

---

## Pipeline

### 1. Preprocessing 
Preprocess raw BindingDB 3D complexes, including structure cleaning and filtering.
```bash
bash script/BindingDBPreprocessing.sh
```
### 2. Featurizer 
Extract chemical and structural features from processed protein–ligand complexes, including functional group annotations and residue-level representations.
```bash
bash script/BindingDBFeaturizer.sh
```
### 3. Dataloader 
Construct datasets and dataloaders with batching, masking, and padding strategies for efficient model training.
```bash
bash script/Dataloader.sh
```
### 4. Run 
Train the LINKER model on the prepared dataset and save checkpoints and logs.
```bash
bash script/Run.sh
```