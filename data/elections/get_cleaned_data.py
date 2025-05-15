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
    ELECTIONS_T1_2017_DATASET,
    ELECTIONS_T2_2017_DATASET,
    ELECTIONS_T1_2022_DATASET,
    ELECTIONS_T2_2022_DATASET
)

#===================================================#
#                   CONSTANTS                       #
#===================================================#

HEADER_LINE = 3

#----------------- POLITICAL MAPPING -----------------#
CANDIDATS_PARTIS = {
    ('ARTHAUD', 'Nathalie'): 'LO',  # Lutte Ouvrière
    ('ASSELINEAU', 'François'): 'UPR',  # Union Populaire Républicaine
    ('CHEMINADE', 'Jacques'): 'SP',  # Solidarité et Progrès
    ('DUPONT-AIGNAN', 'Nicolas'): 'DLF',  # Debout la France
    ('FILLON', 'François'): 'LR',  # Les Républicains
    ('HAMON', 'Benoît'): 'Génération.s',  # Nouveau mouvement après le PS
    ('HIDALGO', 'Anne'): 'PS',  # Parti Socialiste
    ('JADOT', 'Yannick'): 'EELV',  # Europe Écologie Les Verts
    ('LASSALLE', 'Jean'): 'Résistons',  # Résistons
    ('LE PEN', 'Marine'): 'RN',  # Rassemblement National (ex-FN)
    ('MACRON', 'Emmanuel'): 'Renaissance',  # Ex-LREM, renommé en Renaissance
    ('MÉLENCHON', 'Jean-Luc'): 'LFI',  # La France Insoumise
    ('POUTOU', 'Philippe'): 'NPA',  # Nouveau Parti Anticapitaliste
    ('PÉCRESSE', 'Valérie'): 'LR',  # Les Républicains
    ('ROUSSEL', 'Fabien'): 'PCF',  # Parti Communiste Français
    ('ZEMMOUR', 'Éric'): 'Reconquête',  # Reconquête
}

PARTIS_ORIENTATION = {
    'LO': 'Gauche',
    'NPA': 'Gauche',
    'LFI': 'Gauche',
    'PCF': 'Gauche',
    'PS': 'Gauche',
    'EELV': 'Gauche',
    'Génération.s': 'Gauche',
    'Renaissance': 'Centre',
    'UPR': 'Droite',
    'SP': 'Droite',
    'Résistons': 'Centre',
    'LR': 'Droite',
    'DLF': 'Droite',
    'RN': 'Droite',
    'Reconquête': 'Droite'
}

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and processes election data from Excel files for 2017 and 2022.
    
    Returns a dictionary structured as follows:
    {
        '2017': {
            department_code: {
                'resultats_partis': {party: votes, ...},
                'resultats_partis_pct': {party: percentage, ...},
                'resultats_orientation': {orientation: votes, ...},
                'resultats_orientation_pct': {orientation: percentage, ...},
                'parti_gagnant': winning_party,
                'abstentions': abstention_count,
                'abstentions_pct': abstention_percentage
            },
            ...
        },
        '2022': {
            department_code: {...}
        }
    }
    """
    #----------------- DATA LOADING -----------------#
    df_t1_2017 = pd.read_excel(ELECTIONS_T1_2017_DATASET, sheet_name="Départements Tour 1", header=HEADER_LINE)
    df_t2_2017 = pd.read_excel(ELECTIONS_T2_2017_DATASET, sheet_name="Départements Tour 2", header=HEADER_LINE)
    
    df_t1_2022 = pd.read_excel(ELECTIONS_T1_2022_DATASET)
    df_t2_2022 = pd.read_excel(ELECTIONS_T2_2022_DATASET)
    
    #----------------- DATA PROCESSING -----------------#
    resultats_par_dept_t1_2017 = get_partis_result(df_t1_2017)
    resultats_par_dept_t2_2017 = get_partis_result(df_t2_2017)
    resultats_par_dept_t1_2022 = get_partis_result(df_t1_2022)
    resultats_par_dept_t2_2022 = get_partis_result(df_t2_2022)
    
    #----------------- WEIGHTED MERGING -----------------#
    resultats_par_dept_2017 = merge_results_weighted(resultats_par_dept_t1_2017, resultats_par_dept_t2_2017)
    resultats_par_dept_2022 = merge_results_weighted(resultats_par_dept_t1_2022, resultats_par_dept_t2_2022)
    
    #----------------- RESULT STRUCTURING -----------------#
    return {
        '2017': resultats_par_dept_2017,
        '2022': resultats_par_dept_2022
    }

def get_partis_result(dataset):
    """
    Processes a dataset to extract electoral results by department.
    
    Args:
        dataset: DataFrame containing election data
        
    Returns:
        Dictionary with department codes as keys and nested dictionaries 
        of results as values
    """
    #----------------- INITIALIZATION -----------------#
    resultats_par_dept = {}
    
    #----------------- DATA EXTRACTION -----------------#
    for idx, row in dataset.iterrows():
        dept_code = str(row['Code du département']).zfill(2)
        resultats_dept = {}
        resultats_partis = {}
        
        abstentions = row['Abstentions']
        
        #----------------- CANDIDATE PROCESSING -----------------#
        for i in range(len(CANDIDATS_PARTIS)):
            suffixe = f'.{i}' if i > 0 else ''
            nom_col = f'Nom{suffixe}'
            prenom_col = f'Prénom{suffixe}'
            voix_col = f'Voix{suffixe}'
            
            if nom_col in dataset.columns:
                nom = row[nom_col]
                prenom = row[prenom_col]
                voix = row[voix_col]
                if pd.notna(nom) and pd.notna(prenom):
                    candidat = (nom, prenom)
                    resultats_dept[candidat] = voix
                    
                    # Add votes to corresponding party
                    if candidat in CANDIDATS_PARTIS:
                        parti = CANDIDATS_PARTIS[candidat]
                        resultats_partis[parti] = resultats_partis.get(parti, 0) + voix
        
        #----------------- CALCULATION -----------------#
        total_votes = sum(resultats_partis.values())
        
        # Calculate party percentage results
        resultats_partis_pct = {
            parti: (voix / total_votes * 100) if total_votes > 0 else 0
            for parti, voix in resultats_partis.items()
        }
        
        # Calculate orientation results
        resultats_orientation = {}
        for parti, voix in resultats_partis.items():
            orientation = PARTIS_ORIENTATION[parti]
            resultats_orientation[orientation] = resultats_orientation.get(orientation, 0) + voix
        
        # Calculate orientation percentage results
        resultats_orientation_pct = {
            orientation: (voix / total_votes * 100) if total_votes > 0 else 0
            for orientation, voix in resultats_orientation.items()
        }
        
        #----------------- RESULT STRUCTURING -----------------#
        parti_gagnant = max(resultats_partis.items(), key=lambda x: x[1])[0] if resultats_partis else None
        resultats_par_dept[dept_code] = {
            'resultats_detailles': resultats_dept,
            'resultats_partis': resultats_partis,
            'resultats_partis_pct': resultats_partis_pct,
            'resultats_orientation': resultats_orientation,
            'resultats_orientation_pct': resultats_orientation_pct,
            'parti_gagnant': parti_gagnant,
            'abstentions': abstentions,
            'abstentions_pct': (abstentions / (abstentions + total_votes) * 100) if (abstentions + total_votes) > 0 else 0
        }
    
    return resultats_par_dept

def merge_results_weighted(results_t1, results_t2, poids_t1=0.3, poids_t2=0.7):
    """
    Merges first and second round election results with weighted importance.
    
    Args:
        results_t1: Dictionary of first round results
        results_t2: Dictionary of second round results
        poids_t1: Weight for first round (default: 0.3)
        poids_t2: Weight for second round (default: 0.7)
        
    Returns:
        Dictionary with merged results by department
    """
    #----------------- INITIALIZATION -----------------#
    merged = {}
    all_depts = set(results_t1.keys()) | set(results_t2.keys())
    
    #----------------- MERGING PROCESS -----------------#
    for dept in all_depts:
        if dept in results_t1 and dept in results_t2:
            #----------------- PARTY RESULTS MERGING -----------------#
            partis = set(results_t1[dept]['resultats_partis'].keys()) | set(results_t2[dept]['resultats_partis'].keys())
            merged_partis = {}
            
            for parti in partis:
                votes_t1 = results_t1[dept]['resultats_partis'].get(parti, 0)
                votes_t2 = results_t2[dept]['resultats_partis'].get(parti, 0)
                merged_partis[parti] = votes_t1 * poids_t1 + votes_t2 * poids_t2
            
            #----------------- CALCULATIONS -----------------#
            total_votes = sum(merged_partis.values())
            merged_pct = {
                parti: (votes / total_votes * 100 if total_votes > 0 else 0)
                for parti, votes in merged_partis.items()
            }
            
            parti_gagnant = max(merged_partis.items(), key=lambda x: x[1])[0] if merged_partis else None
            
            abstentions = results_t1[dept]['abstentions'] * poids_t1 + results_t2[dept]['abstentions'] * poids_t2
            total_global = total_votes + abstentions
            abstentions_pct = abstentions / total_global * 100 if total_global > 0 else 0
            
            #----------------- ORIENTATION MERGING -----------------#
            orientations = set()
            for parti in partis:
                if parti in PARTIS_ORIENTATION:
                    orientations.add(PARTIS_ORIENTATION[parti])
            
            merged_orientations = {}
            for orientation in orientations:
                votes_orientation_t1 = sum(
                    results_t1[dept]['resultats_partis'].get(parti, 0)
                    for parti in results_t1[dept]['resultats_partis']
                    if parti in PARTIS_ORIENTATION and PARTIS_ORIENTATION[parti] == orientation
                )
                votes_orientation_t2 = sum(
                    results_t2[dept]['resultats_partis'].get(parti, 0)
                    for parti in results_t2[dept]['resultats_partis']
                    if parti in PARTIS_ORIENTATION and PARTIS_ORIENTATION[parti] == orientation
                )
                merged_orientations[orientation] = votes_orientation_t1 * poids_t1 + votes_orientation_t2 * poids_t2
            
            merged_orientation_pct = {
                orientation: (votes / total_votes * 100 if total_votes > 0 else 0)
                for orientation, votes in merged_orientations.items()
            }
            
            #----------------- RESULT STRUCTURING -----------------#
            merged[dept] = {
                'resultats_partis': merged_partis,
                'resultats_partis_pct': merged_pct,
                'resultats_orientation': merged_orientations,
                'resultats_orientation_pct': merged_orientation_pct,
                'parti_gagnant': parti_gagnant,
                'abstentions': abstentions,
                'abstentions_pct': abstentions_pct
            }
        elif dept in results_t1:
            merged[dept] = results_t1[dept]
        else:
            merged[dept] = results_t2[dept]
            
    return merged

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())