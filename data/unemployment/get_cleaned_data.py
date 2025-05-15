# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

#===================================================#
#               PATH CONFIGURATION                  #
#===================================================#

BASE_PATH_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_PATH_SCRIPT, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from const import UNEMPLOYMENT_DATASET

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and cleans unemployment data from an Excel file.

    Returns a dictionary structured as follows:
    {
        '2017': {department_code: unemployment_rate_2017, ...},
        '2022': {department_code: unemployment_rate_2022, ...}
    }
    """

    #----------------- DATA LOADING -----------------#
    df = pd.read_excel(UNEMPLOYMENT_DATASET)
    
    #----------------- COLUMN SELECTION -----------------#
    cols_2017 = [col for col in df.columns if '2017-T3' in col]
    cols_2022 = [col for col in df.columns if '2022-T3' in col]
    
    #----------------- COLUMN RENAMING -----------------#
    rename_dict = {col: col.replace('-T3', '') for col in cols_2017 + cols_2022}
    df.rename(columns=rename_dict, inplace=True)
    
    #----------------- UPDATED COLUMN NAMES -----------------#
    cols_2017 = [col.replace('-T3', '') for col in cols_2017]
    cols_2022 = [col.replace('-T3', '') for col in cols_2022]
    
    #----------------- DATA FILTERING -----------------#
    columns_to_keep = ['Libellé'] + cols_2017 + cols_2022
    
    df_clean = df[columns_to_keep]
    df_clean = df_clean.iloc[16:16+96]
    
    #----------------- DEPARTMENT CODE GENERATION -----------------#
    codes = [f"{i:02d}" for i in range(1, 96)]
    codes.remove('20')
    codes[19:19] = ['2A', '2B']
    codes = codes[:96]
    
    #----------------- DATAFRAME RESTRUCTURING -----------------#
    df_clean['Departement'] = codes
    df_clean = df_clean[['Departement'] + cols_2017 + cols_2022]
    
    #----------------- OUTPUT FORMATTING -----------------#
    result = {
        "2017": {},
        "2022": {}
    }
    
    for index, row in df_clean.iterrows():
        dept = row['Departement']
        result["2017"][dept] = float(row[cols_2017[0]])
        result["2022"][dept] = float(row[cols_2022[0]])
    
    return result

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())