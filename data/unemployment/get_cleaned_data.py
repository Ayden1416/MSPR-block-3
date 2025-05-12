import pandas as pd
from const import UNEMPLOYMENT_DATASET

def get_cleaned_data():
    df = pd.read_excel(UNEMPLOYMENT_DATASET)
    
    cols_2017 = [col for col in df.columns if '2017-T3' in col]
    cols_2022 = [col for col in df.columns if '2022-T3' in col]
    
    rename_dict = {col: col.replace('-T3', '') for col in cols_2017 + cols_2022}
    df.rename(columns=rename_dict, inplace=True)
    
    cols_2017 = [col.replace('-T3', '') for col in cols_2017]
    cols_2022 = [col.replace('-T3', '') for col in cols_2022]
    
    columns_to_keep = ['Libellé'] + cols_2017 + cols_2022
    
    df_clean = df[columns_to_keep]
    df_clean = df_clean.iloc[16:16+96]
    
    codes = [f"{i:02d}" for i in range(1, 96)]
    codes.remove('20')
    codes[19:19] = ['2A', '2B']
    codes = codes[:96]
    
    df_clean['Departement'] = codes
    df_clean = df_clean[['Departement'] + cols_2017 + cols_2022]
    
    result = {
        "2017": {},
        "2022": {}
    }
    
    for index, row in df_clean.iterrows():
        dept = row['Departement']
        result["2017"][dept] = float(row[cols_2017[0]])
        result["2022"][dept] = float(row[cols_2022[0]])
    
    return result