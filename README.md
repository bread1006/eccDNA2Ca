# eccDNA2Ca

**eccDNA2Ca** is an ensemble learning tool for the prediction of cancer-associated extrachromosomal circular DNAs (eccDNAs), combining traditional machine learning and deep learning approaches.

## Overview

eccDNA2Ca integrates:
- **XGBoost** based on manually extracted features.
- **CNN** based on sequence encoding of eccDNA.
- A **soft-voting** ensemble mechanism.
- **Customizable input:** Accepts FASTA files and/or feature matrices.

## Install
```bash
git clone https://github.com/bread1006/eccDNA2Ca.git
cd eccDNA2Ca
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

## Usage
A pre-trained eccDNA2Ca model has been uploaded, so you don't need to retrain it. You can provide either or both inputs for new prediction:

```bash
python main.py --fasta_file <fasta_path>  #Input only fasta_file for CNN module
python main.py --xgb_features_file <feature_matrix>  #Input only feature_file for XGBoost module
python main.py --fasta_file <fasta_path> --xgb_features_file <feature_matrix>  #Input both 
```
- `--fasta_file`: Path to eccDNA sequences in FASTA format (optional).  
- `--xgb_features_file`: Path to feature matrix Excel file for XGBoost (optional).
## Example usage 
``bash 

