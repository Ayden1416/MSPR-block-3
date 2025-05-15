# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

#===================================================#
#               PATH CONFIGURATION                  #
#===================================================#

BASE_PATH_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_PATH_SCRIPT, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from const import (
    NATALITY_2012_DATASET,
    NATALITY_2017_DATASET,
    NATALITY_2022_DATASET
)

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

#----------------- CONSTANTS -----------------#
# Table de correspondance nom -> code département
DEPARTMENT_CODE_MAP = {
    'Ain': '01', 'Aisne': '02', 'Allier': '03', 'Alpes-de-Haute-Provence': '04', 'Hautes-Alpes': '05',
    'Alpes-Maritimes': '06', 'Ardèche': '07', 'Ardennes': '08', 'Ariège': '09', 'Aube': '10',
    'Aude': '11', 'Aveyron': '12', 'Bouches-du-Rhône': '13', 'Calvados': '14', 'Cantal': '15',
    'Charente': '16', 'Charente-Maritime': '17', 'Cher': '18', 'Corrèze': '19', 'Corse-du-Sud': '2A',
    'Haute-Corse': '2B', "Côte-d'Or": '21', "Côtes-d'Armor": '22', 'Creuse': '23', 'Dordogne': '24',
    'Doubs': '25', 'Drôme': '26', 'Eure': '27', 'Eure-et-Loir': '28', 'Finistère': '29',
    'Gard': '30', 'Haute-Garonne': '31', 'Gers': '32', 'Gironde': '33', 'Hérault': '34',
    'Ille-et-Vilaine': '35', 'Indre': '36', 'Indre-et-Loire': '37', 'Isère': '38', 'Jura': '39',
    'Landes': '40', 'Loir-et-Cher': '41', 'Loire': '42', 'Haute-Loire': '43', 'Loire-Atlantique': '44',
    'Loiret': '45', 'Lot': '46', 'Lot-et-Garonne': '47', 'Lozère': '48', 'Maine-et-Loire': '49',
    'Manche': '50', 'Marne': '51', 'Haute-Marne': '52', 'Mayenne': '53', 'Meurthe-et-Moselle': '54',
    'Meuse': '55', 'Morbihan': '56', 'Moselle': '57', 'Nièvre': '58', 'Nord': '59',
    'Oise': '60', 'Orne': '61', 'Pas-de-Calais': '62', 'Puy-de-Dôme': '63', 'Pyrénées-Atlantiques': '64',
    'Hautes-Pyrénées': '65', 'Pyrénées-Orientales': '66', 'Bas-Rhin': '67', 'Haut-Rhin': '68', 'Rhône': '69',
    'Haute-Saône': '70', 'Saône-et-Loire': '71', 'Sarthe': '72', 'Savoie': '73', 'Haute-Savoie': '74',
    'Paris': '75', 'Seine-Maritime': '76', 'Seine-et-Marne': '77', 'Yvelines': '78', 'Deux-Sèvres': '79',
    'Somme': '80', 'Tarn': '81', 'Tarn-et-Garonne': '82', 'Var': '83', 'Vaucluse': '84',
    'Vendée': '85', 'Vienne': '86', 'Haute-Vienne': '87', 'Vosges': '88', 'Yonne': '89',
    'Territoire de Belfort': '90', 'Essonne': '91', 'Hauts-de-Seine': '92', 'Seine-Saint-Denis': '93',
    'Val-de-Marne': '94', "Val-d'Oise": '95', 'Guadeloupe': '971', 'Martinique': '972', 'Guyane': '973', 
    'La Réunion': '974', 'Mayotte': '976'
}

# Créer la correspondance inverse code -> nom
DEPARTMENT_NAME_MAP = {code: name for name, code in DEPARTMENT_CODE_MAP.items()}

# Liste des régions à exclure
REGIONS_TO_EXCLUDE = [
    'Île-de-France', 'Centre-Val de Loire', 'Bourgogne-Franche-Comté', 
    'Normandie', 'Hauts-de-France', 'Grand Est', 'Pays de la Loire', 
    'Bretagne', 'Nouvelle-Aquitaine', 'Occitanie', 'Auvergne-Rhône-Alpes', 
    'Provence-Alpes-Côte d\'Azur', 'Corse', 'France métropolitaine'
]

def get_cleaned_data():
    """
    Loads and cleans natality data from Excel files for years 2012, 2017, and 2022.
    
    Returns a dictionary structured as follows:
    {
        '2012': {department_code: value, ...},
        '2017': {department_code: value, ...},
        '2022': {department_code: value, ...}
    }
    """
    
    #----------------- CONSTANTS -----------------#
    # Différentes configurations selon les années
    CONFIGS = {
        2012: {
            'header_row': 13,
            'col_indices': [0, 2],  # Département, Total annuel
            'col_names': ["departement", "nes_vivants_total"]
        },
        2017: {
            'header_row': 13,
            'col_indices': [0, 2],  # Département, Total annuel
            'col_names': ["departement", "nes_vivants_total"]
        },
        2022: {
            'header_row': 4,
            'col_indices': [0, 1],  # Département, Année (total)
            'col_names': ["departement", "nes_vivants_total"]
        }
    }
    
    #----------------- DATA SOURCES -----------------#
    datasets = {
        2012: NATALITY_2012_DATASET,
        2017: NATALITY_2017_DATASET,
        2022: NATALITY_2022_DATASET
    }
    
    results = {
        "2012": {},
        "2017": {},
        "2022": {}
    }
    
    #----------------- DATA PROCESSING -----------------#
    for year, file_path in datasets.items():
        try:
            #----------------- FILE LOADING -----------------#
            if not os.path.exists(file_path):
                print(f"Le fichier pour l'année {year} n'existe pas: {file_path}")
                continue
            
            config = CONFIGS[year]
            df = pd.read_excel(file_path, header=config['header_row'])
            
            #----------------- COLUMN SELECTION -----------------#
            if df.shape[1] > max(config['col_indices']):
                df_selected = pd.DataFrame()
                
                for i, col_name in enumerate(config['col_names']):
                    idx = config['col_indices'][i] if i < len(config['col_indices']) else None
                    if idx is not None and idx < df.shape[1]:
                        df_selected[col_name] = df.iloc[:, idx]
            else:
                print(f"Problème de colonnes pour l'année {year}, nombre de colonnes: {df.shape[1]}")
                continue
            
            #----------------- DATA CLEANING -----------------#
            # Nettoyage des noms de départements
            df_selected.loc[:, "departement"] = df_selected["departement"].astype(str).str.strip()
            
            # Filtrer les départements valides et exclure les régions
            valid_depts = set(DEPARTMENT_CODE_MAP.keys())
            df_selected = df_selected[df_selected["departement"].isin(valid_depts)]
            df_selected = df_selected[~df_selected["departement"].isin(REGIONS_TO_EXCLUDE)]
            
            # Conversion en numérique pour les naissances
            df_selected.loc[:, "nes_vivants_total"] = pd.to_numeric(df_selected["nes_vivants_total"], errors='coerce')
            
            # Suppression des valeurs manquantes ou nulles
            df_selected = df_selected.dropna(subset=["departement", "nes_vivants_total"])
            df_selected = df_selected[df_selected["nes_vivants_total"] > 0]
            
            #----------------- OUTPUT FORMATTING -----------------#
            # Créer dictionnaire avec codes départementaux
            births_dict = {}
            for _, row in df_selected.iterrows():
                dept_name = row['departement']
                dept_code = DEPARTMENT_CODE_MAP.get(dept_name)
                if dept_code:
                    births_dict[dept_code] = int(row['nes_vivants_total'])
            
            results[str(year)] = births_dict
            
        except Exception as e:
            print(f"Erreur lors du traitement de l'année {year}: {e}")
            results[str(year)] = {}
    
    return results

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    data = get_cleaned_data()
    print(data)