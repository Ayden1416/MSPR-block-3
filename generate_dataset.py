from data.age.get_cleaned_data import get_cleaned_data as get_age_data
from data.criminality.get_cleaned_data import get_cleaned_data as get_criminality_data
import pandas as pd

if __name__ == "__main__":
  
    age_data = get_age_data()
    # Example of age_data structure:
    # {
    #     '2017': {
    #         'CHILDRENS': 15548288,
    #         'ADULTS': 32279939,
    #         'SENIORS': 16475968
    #     },
    #     '2022': {
    #         'CHILDRENS': 15309802,
    #         'ADULTS': 32137327,
    #         'SENIORS': 17850878
    #     }
    # }
    
    criminality_data = get_criminality_data()
    # Example of criminality_data structure:
    # {
    #     '2017': {
    #         '01': 3.611, # (float64) crime rate for department 01
    #         '02': 3.062,
    #         # ... other departments
    #     },
    #     '2022': {
    #         '01': 3.733,
    #         # ... other departments
    #     }
    # }
    
    # Liste des départements de France métropolitaine (96 départements)
    METROPOLITAN_DEPTS = (
        [f"{i:02d}" for i in range(1, 96)]  # Départements 01-95
        + ["2A", "2B"]  # Corse
    )
    
    # On ne traite que 2017 et 2022
    YEARS = ["2017", "2022"]
    
    rows = []
    for year in YEARS:
        # Récupérer les données d'âge pour l'année correspondante
        age_stats = age_data.get(year)
        if age_stats is None:
            continue
        
        departments = criminality_data.get(year, {})
        childs = age_stats.get("CHILDRENS")
        adults = age_stats.get("ADULTS")
        seniors = age_stats.get("SENIORS")
        
        # Pour chaque département métropolitain de l'année, on crée une ligne de données
        for dept_code, crime_rate in departments.items():
            if dept_code not in METROPOLITAN_DEPTS:
                continue
            
            rows.append({
                "department_code": dept_code,
                "criminality_indice": crime_rate,
                "childs": childs,
                "adults": adults,
                "seniors": seniors,
                "year": year
            })
    
    # Création d'un DataFrame pandas avec les données fusionnées
    df = pd.DataFrame(rows)
    
    # Exporter le DataFrame dans un fichier Excel sans index
    df.to_excel("dataset.xlsx", index=False)
    