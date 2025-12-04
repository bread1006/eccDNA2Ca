# eccDNA2Ca

**eccDNA2Ca** is an ensemble learning tool for the prediction of cancer-associated extrachromosomal circular DNAs (eccDNAs), combining traditional machine learning and deep learning approaches.

## Overview

eccDNA2Ca integrates:
- **XGBoost** based on manually extracted features.
- **CNN** based on sequence encoding of eccDNA.
- A **soft-voting** ensemble mechanism.
- **Customizable input:** Accepts FASTA files and/or feature matrices.
- **Optional feature extraction module:** Automatically generates feature matrices if none are available.

## Install
```bash
git lfs clone https://github.com/bread1006/eccDNA2Ca.git
cd eccDNA2Ca
```
OR
```bash
git clone https://github.com/bread1006/eccDNA2Ca.git
cd eccDNA2Ca
git lfs install
git lfs pull
```

## Create an environment
```bash
conda create -n eccDNA2Ca python=3.11
conda activate eccDNA2Ca
```

## Install  dependencies
```bash
pip install -r requirements.txt
```

## Feature extraction (optional)
If you do not have an existing XGBoost feature matrix, you can generate one using the provided feature extraction module:

```bash
unzip Feature_Extractor.zip -d Feature_Extractor
cd Feature_Extractor/
python Feature_Extractor.py \
    --bed ExternalValidation.bed \             ## **User-provided**: BED file for your eccDNA regions
    --fasta ExternalValidation_eccDNA.fasta \ ## **User-provided**: FASTA file corresponding to your eccDNA sequences
    --repeat_bed hg19_repeats.bed \           ## **Pre-prepared**: BED file of repetitive regions
    --intron_bed hg19_introns.bed \           ## **Pre-prepared**: BED file of intron regions
    --tfbs_bed hg19_tfbs.bed \                ## **Pre-prepared**: BED file of transcription factor binding sites
    --gtf gencode_genes_only.gtf \            ## **Pre-prepared**: GTF file of gene annotations
    --cancer_list CancerGene_list.csv \       ## **Pre-prepared**: List of cancer-related genes
    --out feature_matrix.csv                   ## Output feature matrix CSV for XGBoost module
```
Notes:
Files marked as User-provided must be supplied by you for your specific eccDNA dataset.
Files marked as Pre-prepared are already included in the repository or available from standard resources, and do not need to be modified.
The output feature_matrix.csv can then be used as input to the XGBoost module of eccDNA2Ca.

## Usage
A pre-trained eccDNA2Ca model has been uploaded, so you don't need to retrain it. You can provide either or both inputs for new predictions:

```bash
python main.py --fasta_file <fasta_path>  #Input only fasta_file for CNN module
python main.py --xgb_features_file <feature_matrix>  #Input only feature_file for XGBoost module
python main.py --fasta_file <fasta_path> --xgb_features_file <feature_matrix>  #Input both 
```
- `--fasta_file`: Path to eccDNA sequences in FASTA format (optional).  
- `--xgb_features_file`: Path to feature matrix Excel file for XGBoost (optional).

## Example usage 
```bash
python main.py --fasta_file ExternalValidation_eccDNA.fasta --xgb_features_file ExternalValidation_XGBoost_Feature_Matrix.xlsx  # Example: run prediction with external validation data
```

## Example Output
**ID     Predictions:**

seq1     0.9722  
seq2     0.9732  
seq3     0.9848  
seq4     0.8627  
seq5     0.9703  
seq6     0.8070  
...

## Notice
This repository contains a large `.h5` model file that are managed using **Git Large File Storage (Git LFS)**.
Please ensure you have **Git LFS** installed and configured before cloning or pulling the repository to correctly handle these large files.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Citation
If you use this code and datasets in your research, please cite our paper:

> Zhou L. et al., *eccDNA2Ca: An Ensemble Deep Learning Framework for Interpretable Prediction of Cancer-Associated Extrachromosomal Circular DNA*. Manuscript in preparation, 2025.
