# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import sqlite3
from pathlib import Path

#===================================================#
#               PATH CONFIGURATION                  #
#===================================================#
BASE_PATH_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_PATH_SCRIPT, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.age.get_cleaned_data import get_cleaned_data as get_age_data
from data.criminality.get_cleaned_data import get_cleaned_data as get_criminality_data
from data.elections.get_cleaned_data import get_cleaned_data as get_elections_data
from data.unemployment.get_cleaned_data import get_cleaned_data as get_unemployment_data
from data.wealth_per_capita.get_cleaned_data import get_cleaned_data as get_wealth_data
from data.immigration.get_cleaned_data import get_cleaned_data as get_immigration_data
from data.real_estate.get_cleaned_data import get_cleaned_data as get_real_estate_data
from data.average_salary.get_cleaned_data import get_cleaned_data as get_average_salary_data
from data.natality.get_cleaned_data import get_cleaned_data as get_natality_data
from data.density.get_cleaned_data import get_cleaned_data as get_density_data
from data.median_standard_living.get_cleaned_data import get_cleaned_data as get_median_data

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def create_department_stats_database(db_file="data/dataset.sqlite", table_name="department_stats"):
    """
    Collects and merges data from multiple datasets into a SQLite database.
    
    Returns:
        dict: A summary of the operation with the following structure:
        {
            "status": "success" or "error",
            "rows_count": number of rows created,
            "years_processed": list of years processed,
            "db_file": path to the created database file
        }
    """
    
    #----------------- CONFIGURATION -----------------#
    YEARS = ["2017", "2022"]
    METROPOLITAN_DEPTS = [f"{i:02d}" for i in range(1, 96)]
    METROPOLITAN_DEPTS.remove("20")  # Remove Corsica's old code
    METROPOLITAN_DEPTS.extend(["2A", "2B"])  # Add Corsica's new codes
    
    #----------------- DATA COLLECTION -----------------#
    age_data = get_age_data()
    criminality_data = get_criminality_data()
    elections_data = get_elections_data()
    unemployment_data = get_unemployment_data()
    wealth_data = get_wealth_data()
    immigration_data = get_immigration_data()
    real_estate_data = get_real_estate_data()
    average_salary_data = get_average_salary_data()
    natality_data = get_natality_data()
    density_data = get_density_data()
    median_income_data = get_median_data()
    
    print(median_income_data)
    
    #----------------- DATA MERGING -----------------#
    rows = []
    years_processed = []
    
    for year in YEARS:
        age_stats = age_data.get(year)
        if age_stats is None:
            continue
        
        years_processed.append(year)
        
        for dept_code in METROPOLITAN_DEPTS:
            # Skip departments with no criminality data
            if dept_code not in criminality_data.get(year, {}):
                continue
                
            crime_rate = criminality_data[year][dept_code]
            election_dept = elections_data.get(year, {}).get(dept_code, {})
            
            # Create row with base statistics
            row = {
                "department_code": dept_code,
                "year": int(year),
                "criminality_indice": crime_rate,
                "childs": age_stats.get("CHILDRENS"),
                "adults": age_stats.get("ADULTS"),
                "seniors": age_stats.get("SENIORS"),
                "average_price_per_m2": real_estate_data.get(year, {}).get(dept_code),
                "average_salary": average_salary_data.get(year, {}).get(dept_code),
                "unemployment_rate": unemployment_data.get(year, {}).get(dept_code),
                "wealth_per_capita": wealth_data.get(year, {}).get(dept_code),
                "immigration_rate": immigration_data.get(year, {}).get(dept_code),
                "abstentions_pct": election_dept.get("abstentions_pct"),
                "natality_rate": natality_data.get(year, {}).get(dept_code),
                "population_density": density_data.get(year, {}).get(dept_code),
                "median_income": median_income_data.get(year, {}).get(dept_code)
            }
            
            # Add election orientation percentages
            for orientation, pct in election_dept.get("resultats_orientation_pct", {}).items():
                row[f"vote_orientation_pct_{orientation}"] = pct
                
            # Add political party vote percentages
            for party, pct in election_dept.get("resultats_partis_pct", {}).items():
                row[f"vote_pct_{party}"] = pct
                
            rows.append(row)
    
    #----------------- DATABASE EXPORT -----------------#
    df = pd.DataFrame(rows)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    # Remove existing file if it exists
    Path(db_file).unlink(missing_ok=True)
    
    # Create SQLite database
    with sqlite3.connect(db_file) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    return {
        "status": "success",
        "rows_count": len(rows),
        "years_processed": years_processed,
        "db_file": db_file
    }

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    result = create_department_stats_database()
    print(f"Base de données créée avec succès: {result['db_file']}")
    print(f"Nombre de lignes: {result['rows_count']}")
    print(f"Années traitées: {', '.join(result['years_processed'])}")
    print("Terminé ✔️")
