import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",    default="data_mbg_preprocessing/data_mbg_preprocessing.csv")
parser.add_argument("--max_features", type=int,   default=5000)
parser.add_argument("--C",            type=float, default=1.0)
parser.add_argument("--solver",       default="lbfgs")
args = parser.parse_args()

mlflow.set_experiment("MBG_Sentiment_CI")

print(f"[1/4] Memuat data dari: {args.data_path}")
df = pd.read_csv(args.data_path)
df = df.dropna(subset=["text_processed", "label"])
X = df["text_processed"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

with mlflow.start_run(run_name="CI_LogisticRegression"):
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_param("C",            args.C)
    mlflow.log_param("solver",       args.solver)
    mlflow.log_param("test_size",    0.2)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(C=args.C, solver=args.solver, max_iter=300, random_state=42))
    ])

    print("[2/4] Training model...")
    pipeline.fit(X_train, y_train)

    print("[3/4] Evaluasi model...")
    y_pred = pipeline.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy",           accuracy)
    mlflow.log_metric("f1_weighted",        f1)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted",    recall)

    print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

    print("[4/4] Menyimpan artefak...")
    os.makedirs("artifacts", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    cm   = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - MBG Sentiment CI")
    plt.tight_layout()
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)

    report      = classification_report(y_test, y_pred, output_dict=True)
    report_path = "artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path)

    mlflow.sklearn.log_model(pipeline, "model")
    print(f"Run selesai! Run ID: {mlflow.active_run().info.run_id}")

print("Training CI selesai!")
