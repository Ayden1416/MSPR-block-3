import pandas as pd
from const import WEALTH_DATASET

HEADER_LINE = 3


REGIONS_DEPARTMENTS = {
    'Auvergne-Rhône-Alpes': ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74'],
    'Bourgogne-Franche-Comté': ['21', '25', '39', '58', '70', '71', '89', '90'],
    'Bretagne': ['22', '29', '35', '56'],
    'Centre-Val de Loire': ['18', '28', '36', '37', '41', '45'],
    'Corse': ['2A', '2B'],
    'Grand Est': ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88'],
    'Hauts-de-France': ['02', '59', '60', '62', '80'],
    'Île-de-France': ['75', '77', '78', '91', '92', '93', '94', '95'],
    'Normandie': ['14', '27', '50', '61', '76'],
    'Nouvelle-Aquitaine': ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87'],
    'Occitanie': ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82'],
    'Pays de la Loire': ['44', '49', '53', '72', '85'],
    'Provence-Alpes-Côte d\'Azur': ['04', '05', '06', '13', '83', '84']
}

def get_cleaned_data():
    # Lecture du fichier Excel
    df = pd.read_excel(WEALTH_DATASET, sheet_name="PIB par hab 1990-2022", header=HEADER_LINE)
    
    regions = df.iloc[:, 0]  # Première colonne contenant les régions
    
    # Récupération des PIB pour 2017 et 2022
    pib_2017 = df[2017]
    pib_2022 = df[2022]
    
    # Initialisation du résultat final avec une structure par année
    result = {
        '2017': {},
        '2022': {}
    }
    
    # Pour chaque région, associer le PIB de la région à tous ses départements
    for region, pib17, pib22 in zip(regions, pib_2017, pib_2022):
        # Récupérer la liste des départements pour cette région
        departments = REGIONS_DEPARTMENTS.get(region, [])
        for dept in departments:
            result['2017'][dept] = round(pib17, 0)
            result['2022'][dept] = pib22
    
    # Trier les dictionnaires par clé (numéro de département)
    result['2017'] = dict(sorted(result['2017'].items()))
    result['2022'] = dict(sorted(result['2022'].items()))
            
    return result
    