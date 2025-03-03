import pandas as pd
from const import AGE_DATASET

HEADER_LINE = 4  

def get_cleaned_data():
    # Lecture des feuilles 2017 et 2022 en utilisant la bonne ligne d'en-tête
    df_2017 = pd.read_excel(AGE_DATASET, sheet_name='2017', header=HEADER_LINE)
    df_2022 = pd.read_excel(AGE_DATASET, sheet_name='2022', header=HEADER_LINE)

    children_ages = [
        '0 à 4 ans', '5 à 9 ans', '10 à 14 ans', '15 à 19 ans'
    ]
    adult_ages = [
        '20 à 24 ans', '25 à 29 ans', '30 à 34 ans', '35 à 39 ans',
        '40 à 44 ans', '45 à 49 ans', '50 à 54 ans', '55 à 59 ans'
    ]
    senior_ages = [
        '60 à 64 ans', '65 à 69 ans', '70 à 74 ans', '75 à 79 ans',
        '80 à 84 ans', '85 à 89 ans', '90 à 94 ans', '95 ans et plus'
    ]
    
    def sum_categories(df):
        # Filtrer uniquement les départements réels via la colonne 'Unnamed: 0'
        df["dept_numeric"] = pd.to_numeric(df["Unnamed: 0"], errors='coerce')
        df = df[(df["dept_numeric"] >= 1) & (df["dept_numeric"] <= 95)]
      
        # Sélectionner uniquement les colonnes dont le nom correspond exactement aux intitulés attendus
        children_cols = [col for col in df.columns if col in children_ages]
        adult_cols    = [col for col in df.columns if col in adult_ages]
        senior_cols   = [col for col in df.columns if col in senior_ages]

        # Calculer les sommes pour chaque catégorie
        return {
            "CHILDRENS": int(df[children_cols].sum().sum()),
            "ADULTS":    int(df[adult_cols].sum().sum()),
            "SENIORS":   int(df[senior_cols].sum().sum())
        }

    return {
        "2017": sum_categories(df_2017),
        "2022": sum_categories(df_2022)
    }