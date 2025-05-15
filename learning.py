#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, KFold, cross_validate, GridSearchCV
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, make_scorer
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#===================================================#
#               PATH CONFIGURATION                  #
#===================================================#

BASE_PATH_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_PATH_SCRIPT, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration settings
DB_FILE = Path("data/dataset.sqlite")
TABLE_NAME = "department_stats"
RESULTS_DIR = Path("results")
PLOTS_DIR = Path(RESULTS_DIR / "plots")
REPORTS_DIR = Path(RESULTS_DIR / "reports")

# Constants
SEED = 42
DROP_THRESHOLD = 0.40
PRINT_WIDTH = 110

for directory in [RESULTS_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

np.random.seed(SEED)

# Configure plot settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

#===================================================#
#              DATA HANDLING FUNCTIONS              #
#===================================================#

def load_data(db_path: Path, table_name: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Unable to find {db_path.resolve()} – "
            "make sure you have the database."
        )

    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )
        if table_name not in tables["name"].values:
            print(f"Table {table_name} not found. Attempting to merge available tables...")
            return load_and_merge_all_tables(conn)
        
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn)

    if "department_code" in df.columns:
        df["department_code"] = df["department_code"].astype(str)
        
    print(f"{len(df):,} rows imported from {db_path.name}")
    return df


def load_and_merge_all_tables(conn: sqlite3.Connection) -> pd.DataFrame:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )
    data_frames = {}
    excluded_tables = ["sqlite_sequence"]
    
    for table_name in tables["name"].tolist():
        if table_name in excluded_tables:
            continue
            
        print(f"Loading table: {table_name}")
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            if "year" in df.columns:
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
            data_frames[table_name] = df
            print(f"  Loaded {table_name} with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading {table_name}: {e}")
    
    if "elections" in data_frames:
        merged_df = data_frames["elections"].copy()
        
        tables_to_merge = [
            "department_population", "criminality", "immigration", "wealth_per_capita", 
            "average_salary", "natality", "niveau_de_vie_median", "unemployment", "real_estate"
        ]
        
        for table_name in tables_to_merge:
            if table_name in data_frames and not data_frames[table_name].empty:
                df_to_join = data_frames[table_name].copy()
                id_cols_to_drop = [col for col in df_to_join.columns if col.endswith("_id")]
                df_to_join = df_to_join.drop(columns=id_cols_to_drop, errors="ignore")
                
                merged_df = pd.merge(
                    merged_df, df_to_join, 
                    on=["department_code", "year"], how="left", 
                    suffixes=("", f"_{table_name}")
                )
        
        if "departments" in data_frames:
            dept_df = data_frames["departments"].copy()
            merged_df = pd.merge(merged_df, dept_df, on="department_code", how="left")
        
        print(f"Merged data: {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
        return merged_df
    else:
        print("Table 'elections' not found. Unable to merge data.")
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    too_empty = [c for c, pct in df.isnull().mean().items() if pct > DROP_THRESHOLD]
    if too_empty:
        print(f"\n[INFO] Removing columns with too many missing values (> {DROP_THRESHOLD*100:.0f}% NaN):\n → {too_empty}")
        df = df.drop(columns=too_empty)
    
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            fill_value = df[col].median() if df[col].skew() > 1 else df[col].mean()
            df[col] = df[col].fillna(fill_value)
            print(f"Imputing {col} with {'median' if df[col].skew() > 1 else 'mean'}")
    
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            print(f"Imputing {col} with mode: {mode_value}")
    
    return df


def identify_features_targets(df: pd.DataFrame) -> tuple[list, list]:
    orientation_targets = [
        col for col in df.columns 
        if col.startswith("vote_orientation_pct_")
    ]
    
    if not orientation_targets:
        print("[WARNING] No political orientation columns found.")
        potential_targets = [
            col for col in df.columns 
            if col.startswith("vote_pct_") or "gagnant" in col.lower()
        ]
        
        if potential_targets:
            print(f"Potential targets found: {potential_targets}")
            orientation_targets = potential_targets[:3]
    
    exclude_patterns = ["_id", "vote_", "parti_", "tour_", "orientation_"]
    feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in {"election_id", "year"} and 
        not any(pattern in col for pattern in exclude_patterns)
    ]
    
    print(f"\nSelected features ({len(feature_cols)}): {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
    print(f"Selected targets ({len(orientation_targets)}): {orientation_targets}")
    
    return feature_cols, orientation_targets

#===================================================#
#              VISUALIZATION FUNCTIONS              #
#===================================================#

def visualize_data_distribution(df: pd.DataFrame) -> None:
    orientations = [col for col in df.columns if col.startswith("vote_orientation_pct_")]
    if orientations:
        plt.figure(figsize=(12, 8))
        df[orientations].mean().plot(kind="bar", color="seagreen")
        plt.title("Average political orientation distribution", fontsize=14)
        plt.ylabel("Average percentage (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "orientation_distribution.png")
        plt.close()
    
    socio_eco_vars = [
        "unemployment_rate", "average_salary", "immigration_rate", 
        "wealth_per_capita", "criminality_indice"
    ]
    
    existing_vars = [var for var in socio_eco_vars if var in df.columns][:3]
    
    if len(existing_vars) >= 2:
        fig, axes = plt.subplots(len(existing_vars), 1, figsize=(12, 3*len(existing_vars)))
        if len(existing_vars) == 1:
            axes = [axes]
            
        for i, var in enumerate(existing_vars):
            sns.histplot(df[var].dropna(), kde=True, ax=axes[i], color="royalblue")
            axes[i].set_title(f"Distribution of {var}", fontsize=12)
            axes[i].set_ylabel("Frequency")
            
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "socio_eco_distributions.png")
        plt.close()


def visualize_correlations(df: pd.DataFrame, features: list, targets: list) -> None:
    corr = df[features + targets].corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, square=True, linewidths=.5)
    plt.title("Correlation matrix (features and targets)", fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap_full.png")
    plt.close()
    
    if targets:
        target = targets[0]
        plt.figure(figsize=(14, 10))
        target_corrs = df[features].corrwith(df[target]).sort_values(ascending=False)
        top_n = 10
        top_corrs = pd.concat([target_corrs.head(top_n), target_corrs.tail(top_n)]).sort_values(ascending=True)
        colors = ['crimson' if x < 0 else 'forestgreen' for x in top_corrs]
        top_corrs.plot(kind='barh', figsize=(12, 10), color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f"Top correlations with {target}", fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"key_correlations_{target.lower().replace('%', 'pct')}.png")
        plt.close()


def visualize_residuals(X, y, model, target_name, model_name):
    y_pred = model.predict(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.6, color="royalblue")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual values (%)')
    plt.ylabel('Predicted values (%)')
    plt.title(f'{target_name} - Actual vs. Predicted ({model_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"actual_vs_predicted_{target_name.lower().replace('%', 'pct')}_{model_name}.png")
    plt.close()
    

def analyze_feature_importance(model, features, target_name, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print(f"[WARNING] Unable to extract feature importance for {model_name}")
        return None
    
    if len(importances) == len(features):
        feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
        
        csv_path = REPORTS_DIR / f"feature_importance_{target_name.lower().replace('%', 'pct')}_{model_name}.csv"
        feature_importance.to_csv(csv_path)
        print(f"[INFO] Feature importance saved: {csv_path}")
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(10)
        top_features.sort_values().plot.barh(color='darkgreen')
        plt.title(f"Top 10 features for {target_name}", fontsize=14)
        plt.xlabel('Relative importance')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"top10_features_{target_name.lower().replace('%', 'pct')}.png")
        plt.close()
        
        return feature_importance
    else:
        print(f"[ERROR] Incompatible dimensions for importances: {len(importances)} vs {len(features)}")
        return None

def visualize_predictions(predictions_df, year):
    orientation_cols = [
        col for col in predictions_df.columns 
        if col.startswith("vote_orientation_pct_") or col.startswith("vote_pct_")
    ]
    
    if not orientation_cols:
        print("[WARNING] No orientation columns found for visualizations")
        return
    
    plt.figure(figsize=(12, 8))
    
    orientation_means = predictions_df[orientation_cols].mean()
    orientation_means.sort_values(ascending=False).plot(
        kind='bar', 
        color='darkblue', 
        alpha=0.7
    )
    
    plt.title(f"Predicted political orientations for {year}", fontsize=14)
    plt.ylabel("Average predicted percentage (%)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"predictions_{year}_summary.png")
    plt.close()
    
    for orientation in orientation_cols:
        top_depts = predictions_df.sort_values(orientation, ascending=False)[
            ['code_departement', orientation]
        ].head(10)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=top_depts,
            y='code_departement',
            x=orientation,
        )
        plt.title(f"Top 10 departments for {orientation} ({year})", fontsize=14)
        plt.xlabel("Predicted percentage (%)")
        plt.ylabel("Department code")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"predictions_{year}_top_depts_{orientation}.png")
        plt.close()

#===================================================#
#              MODELING FUNCTIONS                   #
#===================================================#

def train_and_evaluate_models(df: pd.DataFrame, features: list, targets: list) -> dict:
    model_defs = {
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=None, random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9, random_state=SEED),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=20_000),
        "SVR": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
    }
    
    try:
        from lightgbm import LGBMRegressor
        model_defs["LightGBM"] = LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9, random_state=SEED)
        print("[INFO] LightGBM available and added to models.")
    except ModuleNotFoundError:
        print("[INFO] LightGBM not available, ignored.")
    
    try:
        from xgboost import XGBRegressor
        model_defs["XGBoost"] = XGBRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9, random_state=SEED, tree_method="hist")
        print("[INFO] XGBoost available and added to models.")
    except ModuleNotFoundError:
        print("[INFO] XGBoost not available, ignored.")
    
    scoring = {
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2":  make_scorer(r2_score),
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    all_scores = []
    fitted_models = {}
    
    for target_idx, target in enumerate(targets):
        print(f"\n{'='*80}\nModeling for {target}\n{'='*80}")
        
        y = df[target]
        X = df[features]
        
        best_mse = np.inf
        best_model_name = None
        best_model_fitted = None
        
        models_results = []
        
        for name, model in model_defs.items():
            print(f"Evaluating {name}...")
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
            
            mse_mean = -cv_results["test_mse"].mean()
            mse_std  = cv_results["test_mse"].std()
            rmse     = np.sqrt(mse_mean)
            mae_mean = -cv_results["test_mae"].mean()
            r2_mean  = cv_results["test_r2"].mean()
            
            model_result = {
                "orientation": target,
                "model": name,
                "mse": mse_mean,
                "rmse": rmse,
                "mae": mae_mean,
                "r2": r2_mean,
                "mse_std": mse_std,
            }
            
            all_scores.append(model_result)
            models_results.append(model_result)
            
            if mse_mean < best_mse:
                best_mse, best_model_name = mse_mean, name
        
        models_df = pd.DataFrame(models_results).sort_values("rmse")
        print("\nModel comparison:")
        print(models_df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
        
        print(f"\n[GRID] Fine-tuning hyperparameters for {target} with {best_model_name}")
        
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
        elif best_model_name == "LightGBM":
            param_grid = {
                "n_estimators": [400, 800],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63],
            }
        elif best_model_name == "XGBoost":
            param_grid = {
                "n_estimators": [400, 800],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6],
            }
        elif best_model_name == "SVR":
            param_grid = {
                "svr__C": [1, 10, 100],
                "svr__epsilon": [0.1, 0.2],
            }
        elif best_model_name == "kNN":
            param_grid = {
                "kneighborsregressor__n_neighbors": [5, 10, 15, 20],
                "kneighborsregressor__weights": ["uniform", "distance"],
            }
        elif best_model_name in ["Ridge", "Lasso"]:
            param_grid = {
                "alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
            }
        else:
            param_grid = {}
        
        base_model = model_defs[best_model_name]
        if param_grid:
            grid = GridSearchCV(base_model, param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            grid.fit(X, y)
            best_model_fitted = grid.best_estimator_
            print(f"    ↳ Best parameters: {grid.best_params_}")
            print(f"    ↳ Optimized MSE: {-grid.best_score_:.4f}")
        else:
            best_model_fitted = base_model.fit(X, y)
            print("    ↳ No hyperparameter optimization for this model.")
        
        fitted_models[target] = best_model_fitted
        
        visualize_residuals(X, y, best_model_fitted, target, best_model_name)
        
        # if (target_idx == 0 or models_df.iloc[0]["r2"] > 0.8) and hasattr(best_model_fitted, "feature_importances_"):
        feature_importance = analyze_feature_importance(best_model_fitted, features, target, best_model_name)
    
    scores_df = pd.DataFrame(all_scores).sort_values(["orientation", "mse"]).reset_index(drop=True)
    
    scores_path = REPORTS_DIR / "model_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"\n(✓) Model comparison exported: {scores_path}")
    
    return fitted_models


def predict_for_future(year, fitted_models, features, departments_data=None, df_clean=None):
    print(f"\n{'='*80}\nGenerating predictions for year {year}\n{'='*80}")
    if departments_data is None:
        if df_clean is not None:
            try:
                last_year = df_clean['year'].max()
                print(f"[INFO] Using data from year {last_year} as base")
                departments_data = df_clean[df_clean['year'] == last_year].copy()
            except (KeyError, ValueError) as e:
                print(f"[ERROR] Problem with dataframe: {e}")
                return pd.DataFrame()
        else:
            print("[ERROR] Cleaned DataFrame not provided")
            return pd.DataFrame()
    if departments_data.empty:
        print("[ERROR] No departmental data available")
        return pd.DataFrame()
    print(f"[INFO] Preparing predictions for {len(departments_data)} departments")
    rows = []
    for idx, row in departments_data.iterrows():
        if 'department_code' in row:
            dept_code = row['department_code']
        elif 'code_departement' in row:
            dept_code = row['code_departement']
        else:
            dept_code = f"Dept_{idx}"
        existing_features = [f for f in features if f in row]
        if len(existing_features) < len(features):
            missing_features = set(features) - set(existing_features)
            print(f"[WARNING] Missing features for {dept_code}: {missing_features}")
            if len(existing_features) < len(features) * 0.7:
                print(f"[ERROR] Too many missing features for {dept_code}, prediction impossible")
                continue
        feat_vec = {}
        for f in features:
            if f in row:
                feat_vec[f] = [row[f]]
            else:
                feat_vec[f] = [departments_data[f].median() if f in departments_data.columns else 0]
        feat_df = pd.DataFrame(feat_vec)
        pred_row = {
            "code_departement": dept_code,
            "annee": year
        }
        for target, model in fitted_models.items():
            try:
                pred = model.predict(feat_df)[0]
                pred_row[target] = max(0, min(100, pred))
            except Exception as e:
                print(f"[ERROR] Unable to predict {target} for {dept_code}: {e}")
                pred_row[target] = None
        rows.append(pred_row)
    if not rows:
        print("[ERROR] No predictions could be generated")
        return pd.DataFrame()
    predictions_df = pd.DataFrame(rows)
    out_path = REPORTS_DIR / f"predictions_{year}.csv"
    predictions_df.to_csv(out_path, index=False)
    print(f"(✓) {year} predictions saved: {out_path}")
    visualize_predictions(predictions_df, year)
    return predictions_df

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

def main():
    
    print(f"{'='*40} MULTI-MODEL ELECTORAL ANALYSIS {'='*40}")
    print(f"Configuration: Seed={SEED}, NaN Threshold={DROP_THRESHOLD*100:.0f}%")
    
    #----------------- DATA LOADING -----------------#
    try:
        df = load_data(DB_FILE, TABLE_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    
    #----------------- DATA CLEANING -----------------#
    df_clean = clean_data(df)
    
    #----------------- EXPLORATORY VISUALIZATION -----------------#
    print("\nCreating exploratory visualizations...")
    visualize_data_distribution(df_clean)
    
    #----------------- FEATURE SELECTION -----------------#
    features, targets = identify_features_targets(df_clean)
    
    if not targets:
        print("[ERROR] No targets identified. Stopping program.")
        return
    
    #----------------- CORRELATION ANALYSIS -----------------#
    print("\nAnalyzing correlations...")
    visualize_correlations(df_clean, features, targets)
    
    #----------------- MODEL TRAINING -----------------#
    print("\nTraining and evaluating models...")
    fitted_models = train_and_evaluate_models(df_clean, features, targets)
    
    #----------------- FUTURE PREDICTIONS -----------------#
    print("\nGenerating predictions...")
    predict_for_future(2027, fitted_models, features, df_clean=df_clean)
    
    print(f"\n{'='*40} ANALYSIS COMPLETED {'='*40}")
    print(f"Results available in directory {RESULTS_DIR}")


if __name__ == "__main__":
    main()