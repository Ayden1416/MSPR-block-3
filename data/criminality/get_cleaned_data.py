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

from const import CRIMINALITY_DATASET

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and cleans criminality data from a CSV file.
    
    Returns a dictionary structured as follows:
    {
        '2017': {department_code: crime_index_score, ...},
        '2022': {department_code: crime_index_score, ...}
    }
    """
    
    #----------------- DATA LOADING -----------------#
    df = pd.read_csv(CRIMINALITY_DATASET, sep=';')
    
    #----------------- DATA FILTERING -----------------#
    df = df[df['annee'].isin([2017, 2022])]
    
    #----------------- METROPOLITAN FILTERING -----------------#
    def is_metropolitan(code):
        """
        Checks if department code belongs to Metropolitan France.
        """
        if code in ['2A', '2B']:
            return True
        try:
            return int(code) < 970
        except ValueError:
            return False
    
    df = df[df['Code_departement'].apply(is_metropolitan)]
    
    #----------------- DATA CLEANING -----------------#
    df['taux_pour_mille'] = df['taux_pour_mille'].str.replace(',', '.').astype(float)
    
    #----------------- INDEX CALCULATION -----------------#
    crime_index = {
        "2017": {},
        "2022": {}
    }
    
    for annee, df_annee in df.groupby('annee'):
        #----------------- STATISTICS CALCULATION -----------------#
        mean = df_annee['taux_pour_mille'].mean()
        std = df_annee['taux_pour_mille'].std()
        
        #----------------- NORMALIZATION -----------------#
        scores_dep = {}
        for _, row in df_annee.iterrows():
            dep = row['Code_departement']
            taux = row['taux_pour_mille']
            z_score = (taux - mean) / std
            score_normalise = 50 + (z_score * 10)
            score_normalise = max(0, min(100, score_normalise))
            scores_dep[dep] = score_normalise
        
        crime_index[str(annee)] = scores_dep
    
    return crime_index

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())