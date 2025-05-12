#!/usr/bin/env python3
# coding: utf-8
"""
learning_comparison.py – Évaluation multi‑modèles + prédictions 2027
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import sqlite3

from sklearn.model_selection import (
    train_test_split, KFold, cross_validate, GridSearchCV
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Configuration
DB_FILE = Path("data/dataset.sqlite")
TABLE_NAME = "department_stats"

# Paramètres généraux & ligne de commande
parser = argparse.ArgumentParser(
    description="Apprentissage multi‑modèles sur les stats départementales"
)
parser.add_argument("--seed", type=int, default=42,
                    help="Graine aléatoire pour la reproductibilité")
parser.add_argument("--drop-threshold", type=float, default=0.40,
                    metavar="[0‑1]",
                    help="Au‑delà de ce pourcentage de NaN, on supprime la colonne")
parser.add_argument("--print-width", type=int, default=110,
                    help="Largeur (en colonnes) pour les audits NaN")

args = parser.parse_args()

# Constantes dérivées des arguments
DROP_THRESHOLD = args.drop_threshold
PRINT_WIDTH   = args.print_width

# Pour les modèles + CV
np.random.seed(args.seed)

# Lecture du jeu de données depuis SQLite
if not DB_FILE.exists():
    raise FileNotFoundError(
        f"Impossible de trouver {DB_FILE.resolve()} – "
        "assurez‑vous d’avoir exécuté le script de génération d’abord."
    )

with sqlite3.connect(DB_FILE) as conn:
    query = f"SELECT * FROM {TABLE_NAME};"
    df = pd.read_sql_query(query, conn)

df["department_code"] = df["department_code"].astype(str)

print(f"{len(df):,} lignes importées depuis {DB_FILE.name}")
print(df.head())

# Audit des NaN & nettoyage
def audit_nan(frame: pd.DataFrame, title: str) -> None:
    nan_stats = frame.isnull().mean().mul(100).sort_values(ascending=False)
    print(f"\n{title}".center(PRINT_WIDTH, "="))
    print(nan_stats.to_string(float_format=lambda x: f"{x:5.1f}%"))

audit_nan(df, "POURCENTAGE DE NaN AVANT NETTOYAGE")

# Colonnes trop vides → suppression
too_empty = [c for c, pct in df.isnull().mean().items() if pct > DROP_THRESHOLD]
if too_empty:
    print(f"\n[INFO] Suppression des colonnes trop vides (> {DROP_THRESHOLD*100:.0f}% NaN) :\n → {too_empty}")
    df = df.drop(columns=too_empty)

# Imputation
numeric_cols = df.select_dtypes(include="number").columns
for col in numeric_cols:
    if df[col].isnull().any():
        fill_value = df[col].median() if df[col].skew() > 1 else df[col].mean()
        df[col] = df[col].fillna(fill_value)

audit_nan(df, "POURCENTAGE DE NaN APRÈS IMPUTATION")

# Colonnes cibles & features
orientations = [
    "vote_orientation_pct_Gauche",
    "vote_orientation_pct_Droite",
    "vote_orientation_pct_Centre",
]

# Vérifier leur présence
missing_targets = [o for o in orientations if o not in df.columns]
if missing_targets:
    sys.exit(f"[ERREUR] Colonnes cibles manquantes : {missing_targets}")

# Nouveau filtre : on retire toutes les colonnes électorales
def is_electoral(col: str) -> bool:
    return col.startswith("vote_pct_") or col in orientations

feature_cols = [
    c for c in df.select_dtypes(include="number").columns
    if not is_electoral(c) and c not in {"year"}
]
print(f"\nTotal features SOCIO‑ÉCO retenues : {len(feature_cols)}")

# Définition des modèles
model_defs = {
    "RandomForest": RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=args.seed
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.05, subsample=0.9,
        random_state=args.seed
    ),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.0005, max_iter=20_000),
    "SVR": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
    "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
}

# LightGBM / XGBoost si dispo
try:
    from lightgbm import LGBMRegressor
    model_defs["LightGBM"] = LGBMRegressor(
        n_estimators=600, learning_rate=0.05, subsample=0.9,
        random_state=args.seed
    )
except ModuleNotFoundError:
    pass

try:
    from xgboost import XGBRegressor
    model_defs["XGBoost"] = XGBRegressor(
        n_estimators=600, learning_rate=0.05, subsample=0.9,
        random_state=args.seed, tree_method="hist"
    )
except ModuleNotFoundError:
    pass

# Entraînement + validation croisée
scoring = {
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "r2":  make_scorer(r2_score),
}
cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)

all_scores = []
fitted_models = {}

for target in orientations:
    y = df[target]
    X = df[feature_cols]

    best_mse = np.inf
    best_model_name = None
    best_model_fitted = None

    for name, model in model_defs.items():
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=False
        )
        mse_mean = -cv_results["test_mse"].mean()
        mse_std  = cv_results["test_mse"].std()
        rmse     = np.sqrt(mse_mean)
        mae_mean = -cv_results["test_mae"].mean()
        r2_mean  = cv_results["test_r2"].mean()

        all_scores.append({
            "orientation": target,
            "model": name,
            "mse": mse_mean,
            "rmse": rmse,
            "mae": mae_mean,
            "r2": r2_mean,
            "mse_std": mse_std,
        })

        if mse_mean < best_mse:
            best_mse, best_model_name = mse_mean, name

    # GridSearch sur le meilleur algo brut
    print(f"\n[GRID] Affinage hyper‑paramètres pour {target} avec {best_model_name}")
    if best_model_name == "RandomForest":
        param_grid = {
            "n_estimators": [300, 600, 900],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 5],
        }
    elif best_model_name == "GradientBoosting":
        param_grid = {
            "n_estimators": [400, 800],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        }
    else:
        param_grid = {}

    base_model = model_defs[best_model_name]
    if param_grid:
        grid = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid.fit(X, y)
        best_model_fitted = grid.best_estimator_
        print(f"    ↳ meilleur jeu : {grid.best_params_}")
    else:
        best_model_fitted = base_model.fit(X, y)

    fitted_models[target] = best_model_fitted

    # Feature importances si dispo
    if hasattr(best_model_fitted, "feature_importances_"):
        fi = (
            pd.Series(best_model_fitted.feature_importances_, index=feature_cols)
              .sort_values(ascending=False)
        )
        fi.to_csv(f"annexes/feature_importance_{target}_{best_model_name}.csv")
        print(f"[INFO] feature_importance_{target}_{best_model_name}.csv sauvegardé")

# Export des scores comparatifs
scores_df = pd.DataFrame(all_scores).sort_values(
    ["orientation", "mse"]
).reset_index(drop=True)

scores_df.to_csv("annexes/model_scores.csv", index=False)
print("\n=================  COMPARATIF (cross‑val. 5‑fold)  =================")
print(scores_df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
print("\n(✓) Fichier 'model_scores.csv' exporté.")

# Prédictions génériques
def predict_for_year(year: int, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in data.iterrows():
        dept_code = row.get("department_code", idx)

        feat_vec = pd.DataFrame({c: [row[c]] for c in feature_cols})

        pred_row = {"code_departement": dept_code, "annee": year}
        for target, mdl in fitted_models.items():
            pred = mdl.predict(feat_vec)[0]
            pred_row[target] = max(0, min(100, pred))
        rows.append(pred_row)
    return pd.DataFrame(rows)

# Exemple : année 2027
pred_2027 = predict_for_year(2027, df)
pred_2027.to_csv("predictions_2027.csv", index=False)
print("\n---------------- APERÇU PRÉDICTIONS 2027 ----------------")
print(pred_2027.head())
print("\n(✓) Fichier 'predictions_2027.csv' exporté.")

# Génération des figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Corrélation (features + cibles)
corr = df[feature_cols + orientations].corr()
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
ax.set_title("Heatmap de corrélation features vs cibles")
fig.savefig("figures/correlation_heatmap.png", dpi=300)
print("Figure enregistrée → figures/correlation_heatmap.png")

# Importance des variables pour chaque orientation
for target, mdl in fitted_models.items():
    if hasattr(mdl, "feature_importances_"):
        fi = pd.Series(mdl.feature_importances_, index=feature_cols)\
               .sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8,6))
        fi.plot.bar(ax=ax)
        ax.set_ylabel("Importance")
        ax.set_title(f"Feature importances – {target}")
        out = f"figures/feature_importance_{target}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        print(f"Figure enregistrée → {out}")

# Réel vs Prédit (cross-val ou prédictions 2027)
pred_2027 = pd.read_csv("predictions_2027.csv")
for target in orientations:
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df[target], pred_2027[target], alpha=0.6)
    ax.plot([0,100],[0,100], "--", linewidth=1)
    ax.set_xlabel("Valeurs historiques")
    ax.set_ylabel("Prédictions 2027")
    ax.set_title(f"Historique vs Prédit – {target}")
    out = f"figures/history_vs_pred_{target}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Figure enregistrée → {out}")