import pandas as pd
from const import (
    REAL_ESTATE_2017_DATASET,
    REAL_ESTATE_2022_DATASET
)

def get_cleaned_data():
    df_2017 = pd.read_csv(REAL_ESTATE_2017_DATASET)
    df_2022 = pd.read_csv(REAL_ESTATE_2022_DATASET)
    
    result = {
        '2017': {},
        '2022': {}
    }
    
    df_2017['Departement'] = df_2017['INSEE_COM'].astype(str).str[:2]
    dept_means_2017 = df_2017.groupby('Departement')['Prixm2Moyen'].mean()
    result['2017'] = dept_means_2017.round(3).to_dict()
    
    df_2022['Departement'] = df_2022['INSEE_COM'].astype(str).str[:2]
    dept_means_2022 = df_2022.groupby('Departement')['Prixm2Moyen'].mean()
    result['2022'] = dept_means_2022.round(3).to_dict()
            
    return result