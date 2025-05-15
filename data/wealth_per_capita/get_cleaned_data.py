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

from const import WEALTH_DATASET

#===================================================#
#               GLOBAL CONSTANTS                    #
#===================================================#

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

#===================================================#
#              PROCESSING FUNCTIONS                 #
#===================================================#

def get_cleaned_data():
    """
    Loads and cleans GDP per capita data from an Excel file.

    Returns a dictionary structured as follows:
    {
        '2017': {department_code: GDP_2017, ...},
        '2022': {department_code: GDP_2022, ...}
    }
    """

    #----------------- DATA LOADING -----------------#
    df = pd.read_excel(
        WEALTH_DATASET,
        sheet_name="PIB par hab 1990-2022",
        header=HEADER_LINE
    )

    #----------------- DATA EXTRACTION -----------------#
    region_col = df.columns[0]
    regions = df[region_col]

    pib_2017 = df[2017]
    pib_2022 = df[2022]

    #----------------- DATA PROCESSING -----------------#
    result = {
        '2017': {},
        '2022': {}
    }

    for region_name, pib17, pib22 in zip(regions, pib_2017, pib_2022):
        if pd.isna(region_name):
            continue

        departments = REGIONS_DEPARTMENTS.get(region_name.strip(), [])

        for dept_code in departments:
            if pd.notna(pib17):
                result['2017'][dept_code] = round(float(pib17), 0)
            if pd.notna(pib22):
                result['2022'][dept_code] = round(float(pib22), 0)

    #----------------- FINALIZATION -----------------#
    result['2017'] = dict(sorted(result['2017'].items()))
    result['2022'] = dict(sorted(result['2022'].items()))

    return result

#===================================================#
#                 MAIN EXECUTION                    #
#===================================================#

if __name__ == "__main__":
    print(get_cleaned_data())