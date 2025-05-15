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

from const import AGE_DATASET

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and processes age demographic data from Excel files for 2017 and 2022.
    
    Returns a dictionary structured as follows:
    {
        '2017': {'CHILDRENS': total_children, 'ADULTS': total_adults, 'SENIORS': total_seniors},
        '2022': {'CHILDRENS': total_children, 'ADULTS': total_adults, 'SENIORS': total_seniors}
    }
    """
    
    #----------------- DATA LOADING -----------------#
    HEADER_LINE = 4
    df_2017 = pd.read_excel(AGE_DATASET, sheet_name='2017', header=HEADER_LINE)
    df_2022 = pd.read_excel(AGE_DATASET, sheet_name='2022', header=HEADER_LINE)
    
    #----------------- AGE CATEGORIES DEFINITION -----------------#
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
    
    #----------------- DATA PROCESSING -----------------#
    def sum_categories(df):
        """
        Process a dataframe to sum population counts by age category.
        
        Returns a dictionary with totals for each age category:
        {
            'CHILDRENS': total_children_count,
            'ADULTS': total_adults_count,
            'SENIORS': total_seniors_count
        }
        """
        # Convert department column to numeric for filtering
        df["dept_numeric"] = pd.to_numeric(df["Unnamed: 0"], errors='coerce')
        
        # Filter to include only valid department codes (1-95)
        df = df[(df["dept_numeric"] >= 1) & (df["dept_numeric"] <= 95)]
        
        # Extract relevant columns for each age category
        children_cols = [col for col in df.columns if col in children_ages]
        adult_cols = [col for col in df.columns if col in adult_ages]
        senior_cols = [col for col in df.columns if col in senior_ages]
        
        # Calculate sums for each category
        return {
            "CHILDRENS": int(df[children_cols].sum().sum()),
            "ADULTS": int(df[adult_cols].sum().sum()),
            "SENIORS": int(df[senior_cols].sum().sum())
        }
    
    #----------------- RESULT CONSTRUCTION -----------------#
    return {
        "2017": sum_categories(df_2017),
        "2022": sum_categories(df_2022)
    }

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())