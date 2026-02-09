import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("out.csv")

# --------------------------------------------------
# Force numeric conversion (critical fix)
# --------------------------------------------------
df['True_Labels'] = pd.to_numeric(df['True_Labels'], errors='coerce')
df['hERG_Label'] = pd.to_numeric(
    df['hERG_Label'], errors='coerce'
)

# --------------------------------------------------
# Remove invalid rows (hidden NaNs / strings)
# --------------------------------------------------
df_eval = df.dropna(subset=['True_Labels', 'hERG_Label']).copy()

# Enforce integer labels (0/1)
y_true = df_eval['True_Labels'].astype(int)
y_pred = df_eval['hERG_Label'].astype(int)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# --------------------------------------------------
# Print results
# --------------------------------------------------
print(f"Accuracy     : {accuracy:.1%}")
print(f"Precision    : {precision:.1%}")
print(f"Recall       : {recall:.1%}")
print(f"F1-Score     : {f1:.1%}")
print(f"Specificity  : {specificity:.1%}")
print(f"Confusion Matrix:\n{cm}")

# --------------------------------------------------
# Append metrics to CSV
# --------------------------------------------------
metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Specificity",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        tn,
        fp,
        fn,
        tp
    ]
})

output_file = "hERG_test_SMILES_Predicted_accuracy.csv"

with open(output_file, "w") as f:
    df.to_csv(f, index=False)
    f.write("\n")
    metrics_df.to_csv(f, index=False)

print(f"\nSaved file: {output_file}")
