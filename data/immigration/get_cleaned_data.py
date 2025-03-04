import pandas as pd
from const import (
    IMMIGRATION_2021_DATASET,
    IMMIGRATION_2017_DATASET
)

HEADER_LINE = 10

def get_cleaned_data():
    # --- Lecture et traitement pour l'année 2017 ---
    df_2017 = pd.read_excel(IMMIGRATION_2017_DATASET, sheet_name="COM", header=HEADER_LINE)
    
    # Définir les tranches d'âge
    age_groups = ['AGE400', 'AGE415', 'AGE425', 'AGE455']
    
    # Pour chaque tranche d'âge, calculer le pourcentage d'immigrants
    for age in age_groups:
        # Total population pour cette tranche d'âge (IMMI1 + IMMI2, les deux sexes)
        total = df_2017[[f'{age}_IMMI1_SEXE1', f'{age}_IMMI1_SEXE2', 
                         f'{age}_IMMI2_SEXE1', f'{age}_IMMI2_SEXE2']].sum(axis=1)
        
        # Total immigrants pour cette tranche d'âge (IMMI1, les deux sexes)
        immigrants = df_2017[[f'{age}_IMMI1_SEXE1', f'{age}_IMMI1_SEXE2']].sum(axis=1)
        
        # Calculer le pourcentage
        df_2017[f'{age}_percent'] = (immigrants / total) * 100
    
    # Calculer la moyenne des pourcentages par tranche d'âge
    df_2017['Immigrant_Percentage'] = df_2017[[f'{age}_percent' for age in age_groups]].mean(axis=1)
    
    # Créer une colonne département
    df_2017['Department'] = df_2017['CODGEO'].str[:2]
    
    # Grouper par département (pour 2017)
    result_2017 = df_2017.groupby('Department').agg({
        'Immigrant_Percentage': 'mean'
    }).reset_index()
    
    # Ajouter la colonne Year
    result_2017["Year"] = 2017
    
    # --- Lecture et préparation des données pour l'année 2021 ---
    df_2021 = pd.read_excel(IMMIGRATION_2021_DATASET, sheet_name="Figure 1", header=5)
    
    # On conserve uniquement les colonnes nécessaires et on les renomme
    result_2021 = df_2021[["Code", "Pourcentage immigrés"]].rename(
        columns={
            "Code": "Department",
            "Pourcentage immigrés": "Immigrant_Percentage"
        }
    )
    
    # Filtrer pour garder uniquement les départements métropolitains (01-95)
    result_2021 = result_2021[result_2021["Department"].str.match(r'^(?:[0-9]{1,2})$')]
    result_2021 = result_2021[result_2021["Department"].str.zfill(2).astype(str).astype(int) <= 95]
    
    # Ajouter la colonne Year pour 2021
    result_2021["Year"] = 2021
    
    # Convertir result_2017 en dictionnaire avec les départements comme clés
    dict_2017 = result_2017.set_index('Department')['Immigrant_Percentage'].to_dict()
    
    # Convertir result_2021 en dictionnaire avec les départements comme clés
    dict_2021 = result_2021.set_index('Department')['Immigrant_Percentage'].to_dict()
    
    # S'assurer que les codes de département sont sur 2 chiffres
    dict_2017 = {k.zfill(2): v for k, v in dict_2017.items()}
    dict_2021 = {k.zfill(2): v for k, v in dict_2021.items()}
    
    return {
        "2017": dict_2017,
        "2022": dict_2021 # Aucune donnée existante pour 2022
    }
