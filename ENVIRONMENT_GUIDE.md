# Reproducible Prediction Environment (Exact)

This project is deterministic **only when the environment matches exactly**. The recommended approach is to use conda **explicit lock files**, which pin every package build and URL.

**Important**
- Lock files are platform-specific. A macOS lock file will not work on Windows, and vice versa.
- Do **not** install `rdkit-pypi` via pip; it can override conda RDKit and change fingerprints.

**Quick Start (Exact, Recommended)**
1. Install Miniconda or Anaconda.
2. Create the environment from the lock file for your platform.
3. Run the prediction script using `conda run`.

**macOS (Apple Silicon, osx-arm64)**
1. Create the env:
```bash
conda create -p .conda_env --file conda-osx-arm64.lock
```
2. Run predictions:
```bash
conda run -p .conda_env python External_validation_all.py \
  --input_csv test_SMILES.csv \
  --model_dirs AMES_mutagenicity,Carcinogenicity,DILI,hERG,Hematotoxicity \
  --model_map "AMES_mutagenicity:ensemble,Carcinogenicity:model_randomforest.joblib,DILI:model_randomforest.joblib,hERG:ensemble,Hematotoxicity:model_randomforest.joblib" \
  --output_csv all_predictions.csv
```

**Windows (win-64)**
1. Create the env:
```bash
conda create -p .conda_env --file conda-win-64.lock
```
2. Run predictions:
```bash
conda run -p .conda_env python External_validation_all.py \
  --input_csv test_SMILES.csv \
  --model_dirs AMES_mutagenicity,Carcinogenicity,DILI,hERG,Hematotoxicity \
  --model_map "AMES_mutagenicity:ensemble,Carcinogenicity:model_randomforest.joblib,DILI:model_randomforest.joblib,hERG:ensemble,Hematotoxicity:model_randomforest.joblib" \
  --output_csv all_predictions.csv
```

**Verify Environment Versions**
Run this in the created environment and compare versions to the expected set:
```bash
conda run -p .conda_env python - <<'PY'
import rdkit, sklearn, numpy, pandas, joblib, catboost, lightgbm
print('rdkit', rdkit.__version__)
print('sklearn', sklearn.__version__)
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)
print('joblib', joblib.__version__)
print('catboost', catboost.__version__)
print('lightgbm', lightgbm.__version__)
PY
```
Expected versions used to produce the reference output:
- rdkit 2024.03.5
- scikit-learn 1.3.2
- numpy 1.24.3
- pandas 2.0.3
- joblib 1.4.2
- catboost 1.2.5
- lightgbm 4.5.0



