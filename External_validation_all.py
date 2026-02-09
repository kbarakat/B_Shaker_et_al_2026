#!/usr/bin/env python3
"""
Run external validation (predictions) across multiple model folders and
save a single CSV with probability and binary label columns per property.

Usage:
  python External_validation_all.py \
  --input_csv test_SMILES.csv \
  --model_dirs AMES_mutagenicity,Carcinogenicity,DILI,hERG,Hematotoxicity \
  --model_map "AMES_mutagenicity:ensemble,Carcinogenicity:model_randomforest.joblib,DILI:model_randomforest.joblib,hERG:model_randomforest.joblib,Hematotoxicity:model_randomforest.joblib" \
  --output_csv out.csv

The script will auto-discover model folders in the current working directory
that contain a `preprocessing_pipeline.joblib` file if `--model_dirs` is not provided.
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys


# ---------------- Descriptor computation (same as training script) ----------------
SELECTED_RDKit = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount",
    "RingCount", "FractionCSP3", "NumAliphaticRings", "NumHeteroatoms"
]


def compute_descriptors_for_smiles(smiles, morgan_nbits=2048, morgan_radius=2):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        basic_vals = []
        for name in SELECTED_RDKit:
            try:
                func = getattr(Descriptors, name)
                basic_vals.append(float(func(mol)))
            except Exception:
                basic_vals.append(0.0)
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_nbits)
        morgan_bits = [int(b) for b in morgan.ToBitString()]
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = [int(b) for b in maccs.ToBitString()]
        return basic_vals + morgan_bits + maccs_bits
    except Exception:
        return None


# ---------------- Core prediction helpers ----------------
def find_model_and_preproc(folder):
    """Return (model_path_or_json, preproc_path) or (None, None) if not found."""
    # Preproc
    preproc_candidates = [
        os.path.join(folder, "preprocessing_pipeline.joblib"),
    ]
    preproc_path = None
    for p in preproc_candidates:
        if os.path.exists(p):
            preproc_path = p
            break
    if preproc_path is None:
        # try any file matching 'preprocessing_pipeline' in folder
        for f in os.listdir(folder):
            if f.startswith("preprocessing_pipeline") and f.endswith(".joblib"):
                preproc_path = os.path.join(folder, f)
                break

    # Ensemble first
    ensemble_json = os.path.join(folder, "ensemble_models_list.json")
    if os.path.exists(ensemble_json):
        return ensemble_json, preproc_path

    # fallback: any .joblib model in folder (prefer common names)
    preferred = ["model_randomforest.joblib", "model_xgboost.joblib", "model_lightgbm.joblib", "model_catboost.joblib"]
    for name in preferred:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p, preproc_path

    # last resort: first .joblib in folder
    for f in os.listdir(folder):
        if f.endswith('.joblib'):
            return os.path.join(folder, f), preproc_path

    return None, preproc_path


def load_features_list(folder, preproc=None):
    features_file = os.path.join(folder, "features_list.csv")
    if os.path.exists(features_file):
        expected_features = pd.read_csv(features_file)["feature"].tolist()
        return expected_features
    # derive from preproc if possible
    if preproc is not None and hasattr(preproc, "n_features_in_"):
        n = int(getattr(preproc, "n_features_in_"))
        return [f"f{i}" for i in range(n)]
    return None


def prepare_X_for_model(X_raw, expected_features):
    """Pad or truncate X_raw (DataFrame) to match expected_features list or count."""
    if expected_features is not None:
        expected_count = len(expected_features)
    else:
        expected_count = None

    X = X_raw.copy()
    if expected_count is not None:
        if X.shape[1] < expected_count:
            for i in range(X.shape[1], expected_count):
                X[i] = 0.0
        elif X.shape[1] > expected_count:
            X = X.iloc[:, :expected_count]
        X.columns = expected_features
    else:
        # no expected list, leave as-is but set generic names
        X.columns = [f"f{i}" for i in range(X.shape[1])]

    return X


def predict_with_model_or_ensemble(model_path, X_proc, folder):
    """Return numpy array of probabilities for the positive class (shape: n_samples)."""
    if model_path.endswith('.json'):
        with open(model_path, 'r') as f:
            ensemble_info = json.load(f)
        preds = np.zeros(X_proc.shape[0], dtype=float)
        total_weight = sum(m.get('weight', 1.0) for m in ensemble_info)
        for m in ensemble_info:
            mpath = m['path']
            # Try several candidate resolutions for relative paths to avoid duplicating folder names
            candidates = []
            if os.path.isabs(mpath):
                candidates.append(os.path.normpath(mpath))
            else:
                candidates.append(os.path.normpath(os.path.join(folder, mpath)))
                candidates.append(os.path.normpath(mpath))
                candidates.append(os.path.normpath(os.path.join(folder, os.path.basename(mpath))))

            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                raise FileNotFoundError(f"Ensemble model path not found for entry '{mpath}'. Tried: {candidates}")

            mdl = joblib.load(found)
            preds += m.get('weight', 1.0) * mdl.predict_proba(X_proc)[:, 1]
        preds = preds / total_weight if total_weight != 0 else preds
        return preds
    else:
        # resolve single-model .joblib path if necessary
        model_candidate = model_path
        if not os.path.isabs(model_candidate) and not os.path.exists(model_candidate):
            cand1 = os.path.normpath(os.path.join(folder, model_candidate))
            cand2 = os.path.normpath(os.path.join(folder, os.path.basename(model_candidate)))
            if os.path.exists(cand1):
                model_candidate = cand1
            elif os.path.exists(cand2):
                model_candidate = cand2

        if not os.path.exists(model_candidate):
            raise FileNotFoundError(f"Model file not found: tried '{model_path}' and resolved '{model_candidate}'")

        model = joblib.load(model_candidate)
        return model.predict_proba(X_proc)[:, 1]


# ---------------- Main multi-folder prediction ----------------
def run_all_predictions(input_csv, model_dirs, output_csv, threshold=0.5, model_map=None):
    df = pd.read_csv(input_csv)
    assert 'SMILES' in df.columns, "Input CSV must contain a 'SMILES' column."

    print(f"Loaded {len(df)} molecules from {input_csv}")

    # compute descriptors once
    descriptors = []
    valid_idx = []
    for i, smi in enumerate(df['SMILES']):
        vec = compute_descriptors_for_smiles(smi)
        if vec is not None:
            descriptors.append(vec)
            valid_idx.append(i)

    if not descriptors:
        raise ValueError('No valid SMILES found in the input file!')

    X_raw = pd.DataFrame(descriptors)
    print(f"Computed descriptors for {len(X_raw)} valid SMILES.")

    # Prepare result frame initialized as original df
    result_df = df.copy()

    for folder in model_dirs:
        folder = folder.strip()
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)
        if not os.path.isdir(folder):
            print(f"Skipping {folder}: not a directory.")
            continue

        model_path, preproc_path = find_model_and_preproc(folder)
        if model_path is None or preproc_path is None:
            print(f"Skipping {folder}: missing model or preprocessing pipeline.")
            continue

        prop_name = os.path.basename(folder)
        # allow CLI-provided mapping to override which model is used for this folder
        if model_map and prop_name in model_map:
            choice = model_map[prop_name]
            if choice == 'ensemble':
                candidate = os.path.join(folder, 'ensemble_models_list.json')
                if os.path.exists(candidate):
                    model_path = candidate
                else:
                    raise FileNotFoundError(f"Requested ensemble for '{prop_name}' but {candidate} not found")
            else:
                # resolve choice as path/filename
                if os.path.isabs(choice) and os.path.exists(choice):
                    model_path = choice
                else:
                    cands = [
                        os.path.join(folder, choice),
                        os.path.join(folder, os.path.basename(choice)),
                        choice,
                    ]
                    found = None
                    for c in cands:
                        if os.path.exists(c):
                            found = c
                            break
                    if found is None:
                        raise FileNotFoundError(f"Model choice '{choice}' for '{prop_name}' not found. Tried: {cands}")
                    model_path = found

        print(f"\nProcessing property: {prop_name}")
        print(f"  model: {model_path}")
        print(f"  preproc: {preproc_path}")

        # load preprocessing first (so we can inspect n_features if available)
        preproc = joblib.load(preproc_path)

        expected_features = load_features_list(folder, preproc=preproc)
        X_prepared = prepare_X_for_model(X_raw, expected_features)

        # transform
        X_proc = preproc.transform(X_prepared)

        # predict on valid rows
        preds_valid = predict_with_model_or_ensemble(model_path, X_proc, folder)

        # map back to full-length array
        preds_full = np.full(len(df), np.nan)
        preds_full[valid_idx] = preds_valid

        labels_full = np.full(len(df), np.nan)
        labels_full[valid_idx] = (preds_valid > threshold).astype(int)

        # add columns
        prob_col = f"{prop_name}_Probability"
        label_col = f"{prop_name}_Label"
        result_df[prob_col] = preds_full
        result_df[label_col] = labels_full

        print(f"Added columns: {prob_col}, {label_col}")

    # save
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ All predictions complete. Results saved to: {output_csv}")


def discover_model_dirs(start_dir):
    dirs = []
    for name in os.listdir(start_dir):
        p = os.path.join(start_dir, name)
        if os.path.isdir(p):
            # consider it a model folder if it contains a preprocessing pipeline
            for f in os.listdir(p):
                if f.startswith('preprocessing_pipeline') and f.endswith('.joblib'):
                    dirs.append(p)
                    break
    return dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run external validation across multiple model folders.')
    parser.add_argument('--input_csv', required=True, help='Input CSV with SMILES column')
    parser.add_argument('--model_dirs', required=False, help='Comma-separated list of model folders to use (default: autodiscover)')
    parser.add_argument('--model_map', required=False, help="Comma-separated map of folder:choice (example: AMES_mutagenicity:ensemble,Carcinogenicity:model_randomforest.joblib)")
    parser.add_argument('--output_csv', default='all_predictions.csv', help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary label')
    args = parser.parse_args()

    if args.model_dirs:
        model_dirs = args.model_dirs.split(',')
    else:
        model_dirs = discover_model_dirs(os.getcwd())
        print(f"Discovered model folders: {[os.path.basename(d) for d in model_dirs]}")

    # parse model_map if provided: format folder:choice,folder2:choice2
    parsed_map = None
    if args.model_map:
        parsed_map = {}
        for item in args.model_map.split(','):
            if ':' not in item:
                continue
            k, v = item.split(':', 1)
            parsed_map[k.strip()] = v.strip()

    run_all_predictions(args.input_csv, model_dirs, args.output_csv, threshold=args.threshold, model_map=parsed_map)
