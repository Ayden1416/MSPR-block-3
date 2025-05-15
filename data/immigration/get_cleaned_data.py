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

from const import (
    IMMIGRATION_2017_DATASET,
    IMMIGRATION_2021_DATASET
)

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and cleans immigration data from Excel files for 2017 and 2021.

    Returns a dictionary structured as follows:
    {
        '2017': {department_code: immigrant_percentage_2017, ...},
        '2022': {department_code: immigrant_percentage_2021, ...}
    }
    """
    
    #----------------- DATA LOADING -----------------#
    HEADER_LINE = 10
    df_2017 = pd.read_excel(IMMIGRATION_2017_DATASET, sheet_name="COM", header=HEADER_LINE)
    
    #----------------- 2017 DATA PROCESSING -----------------#
    age_groups = ['AGE400', 'AGE415', 'AGE425', 'AGE455']
    
    for age in age_groups:
        total = df_2017[[f'{age}_IMMI1_SEXE1', f'{age}_IMMI1_SEXE2', 
                         f'{age}_IMMI2_SEXE1', f'{age}_IMMI2_SEXE2']].sum(axis=1)
        
        immigrants = df_2017[[f'{age}_IMMI1_SEXE1', f'{age}_IMMI1_SEXE2']].sum(axis=1)
        
        df_2017[f'{age}_percent'] = (immigrants / total) * 100
    
    df_2017['Immigrant_Percentage'] = df_2017[[f'{age}_percent' for age in age_groups]].mean(axis=1)
    df_2017['Department'] = df_2017['CODGEO'].str[:2]
    
    result_2017 = df_2017.groupby('Department').agg({
        'Immigrant_Percentage': 'mean'
    }).reset_index()
    
    result_2017["Year"] = 2017
    
    #----------------- 2021 DATA PROCESSING -----------------#
    df_2021 = pd.read_excel(IMMIGRATION_2021_DATASET, sheet_name="Figure 1", header=5)
    
    result_2021 = df_2021[["Code", "Pourcentage immigrés"]].rename(
        columns={
            "Code": "Department",
            "Pourcentage immigrés": "Immigrant_Percentage"
        }
    )
    
    result_2021 = result_2021[result_2021["Department"].str.match(r'^(?:[0-9]{1,2})$')]
    result_2021 = result_2021[result_2021["Department"].str.zfill(2).astype(str).astype(int) <= 95]
    
    result_2021["Year"] = 2021
    
    #----------------- OUTPUT FORMATTING -----------------#
    dict_2017 = result_2017.set_index('Department')['Immigrant_Percentage'].to_dict()
    dict_2021 = result_2021.set_index('Department')['Immigrant_Percentage'].to_dict()
    
    dict_2017 = {k.zfill(2): v for k, v in dict_2017.items()}
    dict_2021 = {k.zfill(2): v for k, v in dict_2021.items()}
    
    return {
        "2017": dict_2017,
        "2022": dict_2021
    }

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())