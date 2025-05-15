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
    REAL_ESTATE_2017_DATASET,
    REAL_ESTATE_2022_DATASET
)

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and cleans real estate data from CSV files for 2017 and 2022.
    
    Returns a dictionary structured as follows:
    {
        '2017': {department_code: average_price_per_m2_2017, ...},
        '2022': {department_code: average_price_per_m2_2022, ...}
    }
    """
    
    #----------------- DATA LOADING -----------------#
    df_2017 = pd.read_csv(REAL_ESTATE_2017_DATASET)
    df_2022 = pd.read_csv(REAL_ESTATE_2022_DATASET)
    
    result = {
        '2017': {},
        '2022': {}
    }
    
    #----------------- 2017 DATA PROCESSING -----------------#
    # Extract department code from INSEE commune code
    df_2017['Departement'] = df_2017['INSEE_COM'].astype(str).str[:2]
    
    # Calculate average price per m² by department
    dept_means_2017 = df_2017.groupby('Departement')['Prixm2Moyen'].mean()
    
    # Round to 3 decimal places and convert to dictionary
    result['2017'] = dept_means_2017.round(3).to_dict()
    
    #----------------- 2022 DATA PROCESSING -----------------#
    # Extract department code from INSEE commune code
    df_2022['Departement'] = df_2022['INSEE_COM'].astype(str).str[:2]
    
    # Calculate average price per m² by department
    dept_means_2022 = df_2022.groupby('Departement')['Prixm2Moyen'].mean()
    
    # Round to 3 decimal places and convert to dictionary
    result['2022'] = dept_means_2022.round(3).to_dict()
    
    return result

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())