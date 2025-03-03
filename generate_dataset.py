from data.age.get_cleaned_data import get_cleaned_data as get_age_data

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
    