import pandas as pd
from const import CRIMINALITY_DATASET


def get_cleaned_data():
    df = pd.read_csv(CRIMINALITY_DATASET, sep=';')

    df = df[df['annee'].isin([2017, 2022])] 
    
    def is_metropolitan(code):
        if code in ['2A', '2B']:
            return True
        try:
            return int(code) < 970
        except ValueError:
            return False

    df = df[df['Code_departement'].apply(is_metropolitan)]

    df['taux_pour_mille'] = df['taux_pour_mille'].str.replace(',', '.').astype(float)

    crime_index = {}
    for annee, df_annee in df.groupby('annee'):
        scores_dep = {}
        mean = df_annee['taux_pour_mille'].mean()
        std = df_annee['taux_pour_mille'].std()
        
        for _, row in df_annee.iterrows():
            dep = row['Code_departement']
            taux = row['taux_pour_mille']
            z_score = (taux - mean) / std
            score_normalise = 50 + (z_score * 10)
            score_normalise = max(0, min(100, score_normalise))
            scores_dep[dep] = score_normalise
        crime_index[str(annee)] = scores_dep

    return crime_index