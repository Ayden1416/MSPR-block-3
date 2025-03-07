import pandas as pd
import numpy as np

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------------------
# 1. CHARGEMENT DU DATASET
# -------------------------------------------------------------------
df = pd.read_excel("dataset.xlsx")

# Vérifier les colonnes disponibles
print("Colonnes disponibles :", df.columns)

# -------------------------------------------------------------------
# 2. NETTOYAGE ET PRÉPARATION DES DONNÉES
# -------------------------------------------------------------------
# Check for missing values before dropping
print("Nombre de valeurs manquantes par colonne avant suppression :")
print(df.isnull().sum())

# First fill missing values for specific columns
df['average_price_per_m2'] = df['average_price_per_m2'].fillna(df['average_price_per_m2'].mean())
df['immigration_rate'] = df['immigration_rate'].fillna(df['immigration_rate'].mean())

# For columns with a large number of missing values, decide on a strategy
columns_with_many_nans = [
    'vote_pct_UPR', 'vote_pct_Génération.s', 'vote_pct_SP',
    'vote_pct_PCF', 'vote_pct_Reconquête', 'vote_pct_PS', 'vote_pct_EELV'
]
for col in columns_with_many_nans:
    df[col] = df[col].fillna(0)

# Fill missing values for orientation columns
orientations = ['vote_orientation_pct_Gauche', 'vote_orientation_pct_Droite', 
                'vote_orientation_pct_Centre']
for orientation in orientations:
    if orientation in df.columns:
        df[orientation] = df[orientation].fillna(df[orientation].mean())

# Clean the 'average_salary' column that contains non-breaking spaces
df['average_salary'] = df['average_salary'].astype(str).str.replace('\u202f', '').str.replace(' ', '').astype(float)

# NOW drop rows with missing values in critical columns if necessary
critical_columns = ['criminality_indice', 'unemployment_rate', 'wealth_per_capita']
df = df.dropna(subset=critical_columns)
print(f"Nombre de lignes après suppression des NaN dans les colonnes critiques : {len(df)}")

# Define features based on available columns
features_cols = [
    'criminality_indice', 'childs', 'adults', 'seniors',
    'average_price_per_m2', 'average_salary', 'unemployment_rate',
    'wealth_per_capita', 'immigration_rate', 'abstentions_pct',
]

# Vérification que les colonnes existent
for col in features_cols:
    if col not in df.columns:
        raise ValueError(f"La colonne {col} n'existe pas dans le DataFrame.")

# Check if the DataFrame is empty after cleaning
if df.empty:
    raise ValueError("Le DataFrame est vide après le nettoyage. Vérifiez les étapes de nettoyage.")

# -------------------------------------------------------------------
# 3. ENTRAÎNER DES MODÈLES POUR CHAQUE ORIENTATION POLITIQUE
# -------------------------------------------------------------------
# Créer un dictionnaire pour stocker les modèles
models = {}

# Entraîner un modèle pour chaque orientation politique
for orientation in orientations:
    if orientation in df.columns:
        y = df[orientation]
        X = df[features_cols]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        models[orientation] = model
        
        # Évaluer chaque modèle
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\nModèle pour {orientation}:")
        print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
        print(f"Coefficient de détermination (R²) : {r2:.2f}")

# -------------------------------------------------------------------
# 6. FAIRE DES PRÉDICTIONS POUR CHAQUE DÉPARTEMENT
# -------------------------------------------------------------------
def predict_for_year(year, departements_data):
    """
    Prédit les pourcentages de vote pour chaque orientation politique
    pour tous les départements pour une année donnée.
    
    :param year: L'année pour laquelle faire la prédiction
    :param departements_data: DataFrame contenant les données par département
    :return: DataFrame avec les prédictions
    """
    results = []
    
    for idx, row in departements_data.iterrows():
        dept_code = row['department_code'] if 'department_code' in row else idx
        
        # Créer un DataFrame avec les caractéristiques du département
        dept_features = pd.DataFrame({
            'criminality_indice': [row['criminality_indice']],
            'childs': [row['childs']],
            'adults': [row['adults']],
            'seniors': [row['seniors']],
            'average_price_per_m2': [row['average_price_per_m2']],
            'average_salary': [row['average_salary']],
            'unemployment_rate': [row['unemployment_rate']],
            'wealth_per_capita': [row['wealth_per_capita']],
            'immigration_rate': [row['immigration_rate']],
            'abstentions_pct': [row['abstentions_pct']]
        })
        
        # Prédiction pour chaque orientation
        dept_result = {
            'code_departement': dept_code,
            'annee': year
        }
        
        for orientation, model in models.items():
            prediction = model.predict(dept_features)[0]
            # S'assurer que la prédiction est entre 0 et 100
            prediction = max(0, min(100, prediction))
            dept_result[orientation] = prediction
            
        results.append(dept_result)
    
    return pd.DataFrame(results)

# Exemple: Prédire pour l'année 2027
# Note: On utilise le DataFrame original pour obtenir les données des départements
# Si vous avez besoin d'ajuster les données pour 2027, faites-le ici
predicted_results_2027 = predict_for_year(2027, df)

# Afficher les résultats
print("\nPrédictions pour 2027:")
print(predicted_results_2027.head())

# Ou exporter en CSV
predicted_results_2027.to_csv('predictions_2027.csv', index=False)
print("Résultats exportés dans 'predictions_2027.csv'")