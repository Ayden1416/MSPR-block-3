from data.age.get_cleaned_data import get_cleaned_data as get_age_data
from data.criminality.get_cleaned_data import get_cleaned_data as get_criminality_data
from data.elections.get_cleaned_data import get_cleaned_data as get_elections_data
from data.unemployment.get_cleaned_data import get_cleaned_data as get_unemployment_data
from data.wealth_per_capita.get_cleaned_data import get_cleaned_data as get_wealth_data
from data.immigration.get_cleaned_data import get_cleaned_data as get_immigration_data
from data.real_estate.get_cleaned_data import get_cleaned_data as get_real_estate_data
from data.average_salary.get_cleaned_data import get_cleaned_data as get_average_salary_data
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
    
    elections_data = get_elections_data()
    # Example of elections_data structure:
    # {
    #     '2017': {
    #         '76': {  # department code
    #             'resultats_partis': {  # raw vote counts
    #                 'DLF': 9910.8,
    #                 'LFI': 45718.2,
    #                 # ... other parties
    #             },
    #             'resultats_partis_pct': {  # vote percentages
    #                 'DLF': 1.60,
    #                 'LFI': 7.40,
    #                 # ... other parties
    #             },
    #             'resultats_orientation': {
    #                 'DROITE': 10000,
    #                 'GAUCHE': 20000,
    #                 # ... other orientations
    #             },
    #             'resultats_orientation_pct': {
    #                 'DROITE': 50.0,
    #                 'GAUCHE': 50.0,
    #                 # ... other orientations
    #             },
    #             'parti_gagnant': 'Renaissance',
    #             'abstentions': 201318.7,
    #             'abstentions_pct': 24.58
    #         },
    #         # ... other departments
    #     },
    #     '2022': {
    #         # similar structure
    #     }
    # }
    
    unemployment_data = get_unemployment_data()
    # Example of unemployment_data structure:
    # {
    #     '2017': {
    #         '01': 6.8,  # (float) unemployment rate for department 01
    #         '02': 13.2,
    #         # ... other departments
    #     },
    #     '2022': {
    #         '01': 5.4,
    #         # ... other departments
    #     }
    # }
    
    wealth_data = get_wealth_data()
    # Example of wealth_data structure:
    # {
    #     '2017': {
    #         '01': 33159.23,  # (float) GDP per capita for department 01
    #         '02': 27188.15,
    #         # ... other departments
    #     },
    #     '2022': {
    #         '01': 34521.45,
    #         # ... other departments
    #     }
    # }
    
    immigration_data = get_immigration_data()
    # Example of immigration_data structure:
    # {
    #     '2017': {
    #         '01': 5.401924,  # (float) immigration percentage for department 01
    #         '02': 1.765449,
    #         # ... other departments
    #     },
    #     '2021': {
    #         '01': 12.062011,
    #         # ... other departments
    #     }
    # }
    
    real_estate_data = get_real_estate_data()
    # {
    #     '2017': {
    #         '01': 3.611, // Prixm2Moyen par département en 2017
    #         '02': 3.062, // Prixm2Moyen par département en 2022
    #         # ... other departments
    #     },
    #     '2022': {
    #         '01': 3.733, // Prixm2Moyen par département en 2022
    #         # ... other departments
    #     }
    # }
    
    average_salary_data = get_average_salary_data()
    # {
    #     '2012': {
    #         '01': 25000,  # (int) average salary for department 01 in 2012
    #         '02': 27000,
    #         # ... other departments
    #     },
    #     '2017': {
    #         '01': 28000,
    #         # ... other departments
    #     },
    #     '2022': {
    #         '01': 30000,
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
            
            # Récupérer les données électorales pour le département et l'année correspondante
            election_dept_data = elections_data.get(year, {}).get(dept_code, {})
            vote_pct = election_dept_data.get("resultats_partis_pct", {})
            vote_orientation_pct = election_dept_data.get("resultats_orientation_pct", {})
            
            abstention_pct = election_dept_data.get("abstentions_pct", None)
            unemployment_rate = unemployment_data.get(year, {}).get(dept_code, None)
            wealth_per_capita = wealth_data.get(year, {}).get(dept_code, None)
            immigration_rate = immigration_data.get(year, {}).get(dept_code, None)
            real_estate_price = real_estate_data.get(year, {}).get(dept_code, None)
            average_salary = average_salary_data.get(year, {}).get(dept_code, None)
            
            # Construire la ligne de données
            row = {
                "department_code": dept_code,
                "year": year,
                "criminality_indice": crime_rate,   
                "childs": childs,
                "adults": adults,
                "seniors": seniors,
                "average_price_per_m2": real_estate_price,
                "average_salary": average_salary,
                "unemployment_rate": unemployment_rate,
                "wealth_per_capita": wealth_per_capita,
                "immigration_rate": immigration_rate,
                "abstentions_pct": abstention_pct
            }
            
            # Ajouter les pourcentages d'orientation pour chaque parti
            for orientation, pct in vote_orientation_pct.items():
                row[f"vote_orientation_pct_{orientation}"] = pct
            
            # Ajouter les pourcentages de vote pour chaque parti
            for party, pct in vote_pct.items():
                row[f"vote_pct_{party}"] = pct
            
            rows.append(row)
    
    # Création d'un DataFrame pandas avec les données fusionnées
    df = pd.DataFrame(rows)
    
    # Exporter le DataFrame dans un fichier Excel avec ajustement automatique des colonnes
    with pd.ExcelWriter("dataset.xlsx", engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
        worksheet = writer.sheets['Sheet1']
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 8)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    