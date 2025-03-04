import pandas as pd
from const import UNEMPLOYMENT_DATASET

def get_cleaned_data():
    # Lecture du fichier Excel
    df = pd.read_excel(UNEMPLOYMENT_DATASET)
    
    # Sélectionner uniquement les colonnes T3
    cols_2017 = [col for col in df.columns if '2017-T3' in col]
    cols_2022 = [col for col in df.columns if '2022-T3' in col]
    
    # Créer un dictionnaire de renommage pour toutes les colonnes T3
    rename_dict = {col: col.replace('-T3', '') for col in cols_2017 + cols_2022}
    df.rename(columns=rename_dict, inplace=True)
    
    # Mettre à jour les noms des colonnes après le renommage
    cols_2017 = [col.replace('-T3', '') for col in cols_2017]
    cols_2022 = [col.replace('-T3', '') for col in cols_2022]
    
    # Colonnes à conserver
    columns_to_keep = ['Libellé'] + cols_2017 + cols_2022
    
    # Créer un nouveau DataFrame avec les colonnes sélectionnées
    df_clean = df[columns_to_keep]
    
    # Sélectionner uniquement les départements
    df_clean = df_clean.iloc[16:16+96]  # Changer 95 à 96 pour inclure le département 95
    
    # Générer les codes département (01 à 95 + 2A et 2B)
    codes = [f"{i:02d}" for i in range(1, 96)]
    codes.remove('20')  # Remove 20
    codes[19:19] = ['2A', '2B']  # Insert 2A and 2B at position 19 (where 20 was)
    codes = codes[:96]  # Keep only the first 95 codes
    
    df_clean['Departement'] = codes
    
    # Ne garder que les colonnes souhaitées dans le bon ordre
    df_clean = df_clean[['Departement'] + cols_2017 + cols_2022]
    
    # Convertir le DataFrame en dictionnaire avec le format souhaité
    result = {
        "2017": {},
        "2022": {}
    }
    
    # Remplir les données pour 2017 et 2022
    for index, row in df_clean.iterrows():
        dept = row['Departement']
        result["2017"][dept] = float(row[cols_2017[0]])
        result["2022"][dept] = float(row[cols_2022[0]])
    
    return result