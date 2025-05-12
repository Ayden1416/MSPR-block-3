from data.age.get_cleaned_data import get_cleaned_data as get_age_data
from data.criminality.get_cleaned_data import get_cleaned_data as get_criminality_data
from data.elections.get_cleaned_data import get_cleaned_data as get_elections_data
from data.unemployment.get_cleaned_data import get_cleaned_data as get_unemployment_data
from data.wealth_per_capita.get_cleaned_data import get_cleaned_data as get_wealth_data
from data.immigration.get_cleaned_data import get_cleaned_data as get_immigration_data
from data.real_estate.get_cleaned_data import get_cleaned_data as get_real_estate_data
from data.average_salary.get_cleaned_data import get_cleaned_data as get_average_salary_data

import pandas as pd
import sqlite3
from pathlib import Path

# Configuration
DB_FILE = "data/dataset.sqlite"
TABLE_NAME = "department_stats"
YEARS = ["2017", "2022"]
METROPOLITAN_DEPTS = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"]

# Récupération des datasets
age_data = get_age_data()
criminality_data = get_criminality_data()
elections_data = get_elections_data()
unemployment_data = get_unemployment_data()
wealth_data = get_wealth_data()
immigration_data = get_immigration_data()
real_estate_data = get_real_estate_data()
average_salary_data = get_average_salary_data()

# Fusion des données
rows = []
for year in YEARS:
    age_stats = age_data.get(year)
    if age_stats is None:
        continue

    for dept_code, crime_rate in criminality_data.get(year, {}).items():
        if dept_code not in METROPOLITAN_DEPTS:
            continue

        election_dept = elections_data.get(year, {}).get(dept_code, {})

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
        }

        for orientation, pct in election_dept.get("resultats_orientation_pct", {}).items():
            row[f"vote_orientation_pct_{orientation}"] = pct

        for party, pct in election_dept.get("resultats_partis_pct", {}).items():
            row[f"vote_pct_{party}"] = pct

        rows.append(row)

# Construction du DataFrame
df = pd.DataFrame(rows)

# Export vers SQLite
print(f"\nGénération du fichier SQLite : {DB_FILE}…")
Path(DB_FILE).unlink(missing_ok=True)

with sqlite3.connect(DB_FILE) as conn:
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

print("Terminé ✔️")
