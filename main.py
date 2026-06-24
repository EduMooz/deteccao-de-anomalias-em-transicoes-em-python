import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==========================
# Carregamento dos dados
# ==========================

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print(df["Class"].value_counts(normalize=True))

# ==========================
# Engenharia de atributos
# ==========================

df["Amount_log"] = np.log1p(df["Amount"])

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

# Remove Amount original
X = df.drop(["Class", "Amount"], axis=1)
y = df["Class"]

# ==========================
# Divisão treino e teste
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.3,
    random_state=42
)

# ==========================
# Regressão Logística
# ==========================

print("\n===== REGRESSÃO LOGÍSTICA =====")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

y_probs = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr)
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))

# Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# ==========================
# Random Forest + SMOTE
# ==========================

print("\n===== RANDOM FOREST + SMOTE =====")

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

# ==========================
# Pipeline Logistic Regression
# ==========================

print("\n===== PIPELINE LOGISTIC =====")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_probs_pipeline = pipeline.predict_proba(X_test)[:, 1]

threshold = 0.25

y_pred_custom = (
    y_probs_pipeline > threshold
).astype(int)

print(classification_report(
    y_test,
    y_pred_custom
))

# ==========================
# XGBoost + SMOTE
# ==========================

print("\n===== XGBOOST + SMOTE =====")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train_res, y_train_res)

# Probabilidades
y_probs_xgb = xgb.predict_proba(X_test)[:, 1]

# Threshold personalizado
threshold = 0.25

y_pred_xgb = (
    y_probs_xgb > threshold
).astype(int)

print(classification_report(
    y_test,
    y_pred_xgb
))

# ==========================
# AUC do XGBoost
# ==========================

auc = roc_auc_score(
    y_test,
    y_probs_xgb
)

print("AUC XGBoost:", auc)

# ==========================
# Teste de Thresholds
# ==========================

print("\n===== TESTE DE THRESHOLDS =====")

for threshold in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:

    y_pred = (
        y_probs_xgb > threshold
    ).astype(int)

    print(f"\nThreshold = {threshold}")

    print(classification_report(
        y_test,
        y_pred
    ))

# ==========================
# Importância das Variáveis
# ==========================

importancias = xgb.feature_importances_

plt.figure(figsize=(10, 4))
plt.bar(
    range(len(importancias)),
    importancias
)
plt.title("Importância das Variáveis")
plt.show()

# ==========================
# Grid Search
# ==========================

print("\n===== GRID SEARCH =====")

param_grid = {
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1]
}

grid = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ),
    param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1
)

grid.fit(
    X_train_res,
    y_train_res
)

print("Melhores parâmetros:")
print(grid.best_params_)

# ==========================
# SHAP
# ==========================

print("\n===== SHAP =====")

explainer = shap.Explainer(xgb)

shap_values = explainer(
    X_test[:100]
)

shap.plots.bar(shap_values)