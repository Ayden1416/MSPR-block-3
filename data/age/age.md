# Documentation du fichier

Ce document présente la liste des colonnes et leur type de données respectif pour le fichier suivant :  
**Fichier** : `population-departements-france-2024.csv`

---

## Tableau des colonnes et types estimés

| Nom de la colonne        | Type de données (pandas) | Exemple / Commentaire                                                                                  |
|--------------------------|--------------------------|----------------------------------------------------------------------------------------------------------|
| **Départements**         | `object` / `string`      | « 01 » (chaîne de caractères, permet de conserver les zéros initiaux)                                    |
| **Nom_département**      | `object` / `string`      | « Ain », « Aisne », etc.                                                                                 |
| **Ensemble_0_4_ans**     | `int64`                  | 37,584                                                                                                   |
| **Ensemble_5_9_ans**     | `int64`                  | 44,592                                                                                                   |
| **Ensemble_10_14_ans**   | `int64`                  | 46,872                                                                                                   |
| **Ensemble_15_19_ans**   | `int64`                  | 41,596                                                                                                   |
| **Ensemble_20_24_ans**   | `int64`                  | 32,455                                                                                                   |
| **Ensemble_25_29_ans**   | `int64`                  | 32,252                                                                                                   |
| **Ensemble_30_34_ans**   | `int64`                  | 40,494                                                                                                   |
| **Ensemble_35_39_ans**   | `int64`                  | 44,180                                                                                                   |
| **Ensemble_40_44_ans**   | `int64`                  | 44,933                                                                                                   |
| **Ensemble_45_49_ans**   | `int64`                  | 45,637                                                                                                   |
| **Ensemble_50_54_ans**   | `int64`                  | 46,709                                                                                                   |
| **Ensemble_55_59_ans**   | `int64`                  | 45,043                                                                                                   |
| **Ensemble_60_64_ans**   | `int64`                  | 39,845                                                                                                   |
| **Ensemble_65_69_ans**   | `int64`                  | 35,517                                                                                                   |
| **Ensemble_70_74_ans**   | `int64`                  | 32,976                                                                                                   |
| **Ensemble_75_79_ans**   | `int64`                  | 22,932                                                                                                   |
| **Ensemble_80_84_ans**   | `int64`                  | 15,074                                                                                                   |
| **Ensemble_85_89_ans**   | `int64`                  | 10,760                                                                                                   |
| **Ensemble_90_94_ans**   | `int64`                  | 5,579                                                                                                    |
| **Ensemble_95_ans_plus** | `int64`                  | 1,941                                                                                                    |
| **Total_ensemble**       | `int64`                  | 666,971                                                                                                  |
| **Hommes_0_4_ans**       | `int64`                  | 19,489                                                                                                   |
| **Hommes_5_9_ans**       | `int64`                  | 23,044                                                                                                   |
| **Hommes_10_14_ans**     | `int64`                  | 23,911                                                                                                   |
| **Hommes_15_19_ans**     | `int64`                  | 21,765                                                                                                   |
| **Hommes_20_24_ans**     | `int64`                  | 17,040                                                                                                   |
| **Hommes_25_29_ans**     | `int64`                  | 16,233                                                                                                   |
| **Hommes_30_34_ans**     | `int64`                  | 20,125                                                                                                   |
| **Hommes_35_39_ans**     | `int64`                  | 21,741                                                                                                   |
| **Hommes_40_44_ans**     | `int64`                  | 21,958                                                                                                   |
| **Hommes_45_49_ans**     | `int64`                  | 22,804                                                                                                   |
| **Hommes_50_54_ans**     | `int64`                  | 23,384                                                                                                   |
| **Hommes_55_59_ans**     | `int64`                  | 22,588                                                                                                   |
| **Hommes_60_64_ans**     | `int64`                  | 19,386                                                                                                   |
| **Hommes_65_69_ans**     | `int64`                  | 16,974                                                                                                   |
| **Hommes_70_74_ans**     | `int64`                  | 15,594                                                                                                   |
| **Hommes_75_79_ans**     | `int64`                  | 10,545                                                                                                   |
| **Hommes_80_84_ans**     | `int64`                  | 6,580                                                                                                    |
| **Hommes_85_89_ans**     | `int64`                  | 3,939                                                                                                    |
| **Hommes_90_94_ans**     | `int64`                  | 1,624                                                                                                    |
| **Hommes_95_ans_plus**   | `int64`                  | 358                                                                                                      |
| **Total_hommes**         | `int64`                  | 329,082                                                                                                  |
| **Femmes_0_4_ans**       | `int64`                  | 18,095                                                                                                   |
| **Femmes_5_9_ans**       | `int64`                  | 21,548                                                                                                   |
| **Femmes_10_14_ans**     | `int64`                  | 22,961                                                                                                   |
| **Femmes_15_19_ans**     | `int64`                  | 19,831                                                                                                   |
| **Femmes_20_24_ans**     | `int64`                  | 15,415                                                                                                   |
| **Femmes_25_29_ans**     | `int64`                  | 16,019                                                                                                   |
| **Femmes_30_34_ans**     | `int64`                  | 20,369                                                                                                   |
| **Femmes_35_39_ans**     | `int64`                  | 22,439                                                                                                   |
| **Femmes_40_44_ans**     | `int64`                  | 22,975                                                                                                   |
| **Femmes_45_49_ans**     | `int64`                  | 22,833                                                                                                   |
| **Femmes_50_54_ans**     | `int64`                  | 23,325                                                                                                   |
| **Femmes_55_59_ans**     | `int64`                  | 22,455                                                                                                   |
| **Femmes_60_64_ans**     | `int64`                  | 20,459                                                                                                   |
| **Femmes_65_69_ans**     | `int64`                  | 18,543                                                                                                   |
| **Femmes_70_74_ans**     | `int64`                  | 17,382                                                                                                   |
| **Femmes_75_79_ans**     | `int64`                  | 12,387                                                                                                   |
| **Femmes_80_84_ans**     | `int64`                  | 8,494                                                                                                    |
| **Femmes_85_89_ans**     | `int64`                  | 6,821                                                                                                    |
| **Femmes_90_94_ans**     | `int64`                  | 3,955                                                                                                    |
| **Femmes_95_ans_plus**   | `int64`                  | 1,583                                                                                                    |
| **Total_femmes**         | `int64`                  | 337,889                                                                                                  |

---
