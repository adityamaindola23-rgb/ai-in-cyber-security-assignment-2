"""
Lab 2: Detecting Suspicious Login Activity Using Machine Learning
Run from the project root: python src/solve_lab2.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "simple_login_activity.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Q1: Inspect and clean
df = pd.read_csv(DATA)
print("Dataset info:")
print(df.info())
print("\\nMissing values before cleaning:")
print(df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()
for col in ["failed_logins", "download_mb"]:
    df[col] = df[col].fillna(df[col].median())
print("\\nMissing values after cleaning:", int(df.isna().sum().sum()))
print("Duplicate rows after cleaning:", int(df.duplicated().sum()))
print("Final rows:", len(df))
print(df.head())

# Q2: Split
features = ["failed_logins", "login_hour", "new_device", "download_mb"]
X, y = df[features], df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# Q3: Decision Tree and evaluation
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\\nAccuracy: {accuracy * 100:.2f}%")
labels = ["Normal", "Suspicious"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=labels)
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=160)
plt.show()

# Q4: K-Means (unsupervised; labels are not used for fitting)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
cluster_summary = df.groupby("cluster")[features].mean().round(2)
label_summary = pd.crosstab(df["cluster"], df["label"])
print("\\nCluster mean summary:")
print(cluster_summary)
print("\\nCluster vs known label:")
print(label_summary)
cluster_summary.to_csv(OUT / "cluster_summary.csv")
label_summary.to_csv(OUT / "cluster_vs_label.csv")

# Q5: Feature importance and new records
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("\\nFeature importance:")
print(importance)
importance.to_csv(OUT / "feature_importance.csv", index=False)
plt.figure(figsize=(6, 4))
plt.bar(importance["feature"], importance["importance"])
plt.title("Decision Tree Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=160)
plt.show()

new_records = pd.DataFrame([
    {"failed_logins": 0, "login_hour": 10, "new_device": 0, "download_mb": 45},
    {"failed_logins": 8, "login_hour": 2, "new_device": 1, "download_mb": 180},
    {"failed_logins": 2, "login_hour": 23, "new_device": 1, "download_mb": 900},
])
new_records["prediction"] = model.predict(new_records[features])
print("\\nThree new record predictions:")
print(new_records)
new_records.to_csv(OUT / "new_record_predictions.csv", index=False)

# Q6: Error analysis and final result export
df.to_csv(OUT / "Lab2_Login_ML_Result.csv", index=False)
print("\\nFalse positive example: legitimate late-night work on a new device is marked Suspicious.")
print("False negative example: a disguised attack with few obvious indicators is marked Normal.")
print("Limitation: this small synthetic dataset cannot represent the full variety of real login behaviour.")
print("Human review: investigate high-risk alerts and unusual combinations before taking account action.")

metrics = {
    "accuracy": float(accuracy),
    "confusion_matrix_labels_order": labels,
    "confusion_matrix": cm.tolist(),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\\nSaved results in: {OUT}")
