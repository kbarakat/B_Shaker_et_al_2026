# Toxicity ADMET Prediction Models

Ensemble and single-model predictors for 5 toxicity endpoints: AMES mutagenicity, Carcinogenicity, DILI, hERG, and Hematotoxicity.

## Models

Each folder contains:
- `ensemble_models_list.json` — weighted ensemble configuration
- `model_randomforest.joblib`, `model_xgboost.joblib`, `model_lightgbm.joblib`, `model_catboost.joblib` — individual models
- `preprocessing_pipeline.joblib` — feature scaling/preprocessing
- `features_list.csv` — feature names
- `Train, Test, and Validation csv filies` — Datasets
  
### Toxicity Endpoints
- **AMES_mutagenicity** — AMES bacterial reverse mutation test
- **Carcinogenicity** — In vivo carcinogenicity
- **DILI** — Drug-induced liver injury
- **hERG** — Human ether-à-go-go related gene inhibition
- **Hematotoxicity** — Blood/hematologic toxicity

## Installation

```bash
git clone https://github.com/kbarakat/B_Shaker_et_al_2026.git
cd toxicity-admet-models
pip install -r requirements.txt
```
## If there is any problem regarding environment creation, as an alternative: follow the instruction in "ENVIRONMENT_GUIDE.md" file.

## Usage

### Single property prediction
```bash
python External_validation.py \
  --input_csv test_SMILES.csv \
  --model_path AMES_mutagenicity/ensemble_models_list.json \
  --preproc_path AMES_mutagenicity/preprocessing_pipeline.joblib \
  --output_csv ames_predictions.csv
```

### All properties together
```bash
python External_validation_all.py \
  --input_csv test_SMILES.csv \
  --model_dirs AMES_mutagenicity,Carcinogenicity,DILI,hERG,Hematotoxicity \
  --model_map "AMES_mutagenicity:ensemble,Carcinogenicity:model_randomforest.joblib,DILI:model_randomforest.joblib,hERG:ensemble,Hematotoxicity:model_randomforest.joblib" \
  --output_csv all_predictions.csv
```

## Input Format

Input CSV must contain a `SMILES` column with valid SMILES strings.

## Output

Output CSV contains:
- Original columns from input
- `<property>_Probability` — predicted probability (0–1)
- `<property>_Label` — binary prediction (0 or 1, threshold 0.5)

## Requirements

scikit-learn==1.3.2

rdkit>=2023.03

pandas>=1.5.0

numpy>=1.23.0

joblib>=1.2.0


See `requirements.txt` for exact versions.

## Model Training Details

All models trained on standardized molecular descriptors + Morgan fingerprints + MACCS keys.

- **AMES**: Ensemble (recommended)
- **Carcinogenicity**: Random Forest
- **DILI**: Random Forest
- **hERG**: Ensemble
- **Hematotoxicity**: Random Forest

## NOTE
In any case if you are uable to download real files of models or any, please use following Google drive link as alternative for download
```bash
https://drive.google.com/file/d/1-jKbAg7n_4ZfuPIIZtHKmBTVg2TbQ79c/view?usp=sharing
```


## Citation

Manuscript is submitted in "Computers in Biology and Medicines"
