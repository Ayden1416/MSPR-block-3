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
    DENSITY_2012_DATASET,
    DENSITY_2017_DATASET,
    DENSITY_2022_DATASET
)

#===================================================#
#                   CONSTANTS                       #
#===================================================#

# Dictionnaire des superficies des départements en km²
DEPARTMENT_AREAS = {
    "01": 5762.4, "02": 7361.7, "03": 7340.1, "04": 6925.2, "05": 5549.3,
    "06": 4298.6, "07": 5528.6, "08": 5229.1, "09": 4889.9, "10": 6004.0,
    "11": 6139.2, "12": 8735.1, "13": 5087.5, "14": 5548.3, "15": 5726.0,
    "16": 5956.0, "17": 6864.3, "18": 7235.1, "19": 5857.1, "21": 8763.2,
    "22": 6822.7, "23": 5565.4, "24": 9060.2, "25": 5232.6, "26": 6530.2,
    "27": 6039.6, "28": 5880.4, "29": 6733.2, "2A": 4014.2, "2B": 4665.6,
    "30": 5853.0, "31": 6309.3, "32": 6257.4, "33": 9975.6, "34": 6101.2,
    "35": 6775.0, "36": 6791.1, "37": 6126.7, "38": 7431.5, "39": 4999.2,
    "40": 9243.0, "41": 6343.4, "42": 4781.0, "43": 4977.2, "44": 6815.0,
    "45": 6775.2, "46": 5216.7, "47": 5361.1, "48": 5167.2, "49": 7166.1,
    "50": 5938.4, "51": 8162.2, "52": 6211.4, "53": 5175.2, "54": 5245.9,
    "55": 6211.4, "56": 6822.9, "57": 6216.3, "58": 6816.7, "59": 5742.5,
    "60": 5860.2, "61": 6103.4, "62": 6671.4, "63": 7970.0, "64": 7645.0,
    "65": 4464.1, "66": 4116.0, "67": 4755.0, "68": 3525.1, "69": 3249.1,
    "70": 5360.0, "71": 8575.0, "72": 6206.1, "73": 6188.9, "74": 4388.0,
    "75": 105.4, "76": 6278.0, "77": 5915.3, "78": 2284.4, "79": 5999.4,
    "80": 6170.1, "81": 5758.0, "82": 3718.3, "83": 5973.0, "84": 3567.4,
    "85": 6719.6, "86": 6990.0, "87": 5520.1, "88": 5874.4, "89": 7427.3,
    "90": 609.4, "91": 1804.4, "92": 175.6, "93": 236.0, "94": 245.0, "95": 1246.0
}

#===================================================#
#              HELPER FUNCTIONS                     #
#===================================================#

def is_metropolitan(code):
    """
    Checks if department code belongs to Metropolitan France.
    
    Args:
        code: The department code to check
        
    Returns:
        bool: True if the department is in Metropolitan France, False otherwise
    """
    code_str = str(code).strip()
    if code_str in ['2A', '2B']:
        return True
    try:
        return int(code_str) < 970
    except ValueError:
        return False

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and cleans population density data from CSV files for years 2012, 2017, and 2022.
    
    Returns a dictionary structured as follows:
    {
        '2012': {department_code: density, ...},
        '2017': {department_code: density, ...},
        '2022': {department_code: density, ...}
    }
    """
    
    #----------------- DATA PREPARATION -----------------#
    datasets = {
        2012: DENSITY_2012_DATASET,
        2017: DENSITY_2017_DATASET,
        2022: DENSITY_2022_DATASET
    }
    
    results = {
        "2012": {},
        "2017": {},
        "2022": {}
    }
    
    #----------------- DATASET PROCESSING -----------------#
    for year, file_path in datasets.items():
        try:
            #----------------- FILE LOADING -----------------#
            df = pd.read_csv(file_path)
            
            #----------------- COLUMN STANDARDIZATION -----------------#
            rename_map = {
                "Code département": "code_departement",
                "Nom du département": "nom_departement",
                "Population municipale": "population_municipale"
            }
            df = df.rename(columns=rename_map)
            
            #----------------- HANDLING MISSING COLUMNS -----------------#
            for original, standardized in rename_map.items():
                if standardized not in df.columns and original in df.columns:
                    df[standardized] = df[original]
            
            #----------------- DATA CLEANING -----------------#
            df["population_municipale"] = df["population_municipale"].astype(str).str.replace(r'\D', '', regex=True)
            df["population_municipale"] = pd.to_numeric(df["population_municipale"], errors='coerce')
            
            #----------------- METROPOLITAN FILTERING -----------------#
            df = df[df['code_departement'].apply(is_metropolitan)]
            
            #----------------- DENSITY CALCULATION -----------------#
            density_dict = {}
            for _, row in df.iterrows():
                dep_code = str(row['code_departement']).strip()
                # Fill with zeros to ensure 2-digit format
                if dep_code.isdigit() and len(dep_code) == 1:
                    dep_code = dep_code.zfill(2)
                    
                # Calculate density if area data is available
                if dep_code in DEPARTMENT_AREAS and not pd.isna(row['population_municipale']):
                    area = DEPARTMENT_AREAS[dep_code]
                    density = row['population_municipale'] / area
                    density_dict[dep_code] = round(density, 2)  # Round to 2 decimal places
                else:
                    # If no area data, store None
                    density_dict[dep_code] = None
            
            results[str(year)] = density_dict
            
        except Exception as e:
            print(f"Error processing data for year {year}: {e}")
            print(f"File path: {file_path}")
            results[str(year)] = {}
    
    return results

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())