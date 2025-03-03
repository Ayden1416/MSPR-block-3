import pandas as pd
from const import CRIMINALITY_DATASET


def get_cleaned_data():
    # Chargement des données
    df = pd.read_csv(CRIMINALITY_DATASET, sep=';')

    # Filtrer les années 2017 et 2022
    df = df[df['annee'].isin([2017, 2022])]
    
    # Exclure les DOM-TOM (codes > 970)
    # Créer une fonction pour vérifier si le département est en métropole
    def is_metropolitan(code):
        if code in ['2A', '2B']:  # Codes pour la Corse
            return True
        try:
            return int(code) < 970
        except ValueError:
            return False

    df = df[df['Code_departement'].apply(is_metropolitan)]

    # Conversion des colonnes numériques si nécessaire
    df['taux_pour_mille'] = df['taux_pour_mille'].str.replace(',', '.').astype(float)

    # Utiliser le z-score (nombre d'écarts-types par rapport à la moyenne)
    crime_index = {}
    for annee, df_annee in df.groupby('annee'):
        scores_dep = {}
        mean = df_annee['taux_pour_mille'].mean()
        std = df_annee['taux_pour_mille'].std()
        
        for _, row in df_annee.iterrows():
            dep = row['Code_departement']
            taux = row['taux_pour_mille']
            # Transformer le z-score pour avoir des valeurs entre 0 et 100
            z_score = (taux - mean) / std
            score_normalise = 50 + (z_score * 10)  # Centre à 50, écart-type de 10
            # Limiter les valeurs entre 0 et 100
            score_normalise = max(0, min(100, score_normalise))
            scores_dep[dep] = score_normalise
        crime_index[str(annee)] = scores_dep

    return crime_index