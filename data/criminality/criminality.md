Voici le fichier README au format Markdown :

# Documentation du fichier

Ce document présente la liste des colonnes et leur type de données respectif pour le fichier suivant :  
**Fichier** : `donnee-dep-data.gouv-2024-geographie2024-produit-le2025-01-26.csv`

---

## Tableau des colonnes et types estimés

| Nom de la colonne        | Type de données (pandas) | Exemple / Commentaire                                                                                  |
|--------------------------|--------------------------|----------------------------------------------------------------------------------------------------------|
| **Code_departement**     | `object` / `string`      | « 01 » (chaîne de caractères, permet de conserver les zéros initiaux)                                    |
| **Code_region**          | `int64`                  | 84                                                                                                       |
| **annee**                | `int64`                  | 2016                                                                                                     |
| **indicateur**           | `object` / `string`      | « Homicides », « Usage de stupéfiants », etc.                                                            |
| **unite_de_compte**      | `object` / `string`      | « Victime », « Mis en cause », etc.                                                                      |
| **nombre**               | `int64`                  | 5                                                                                                        |
| **taux_pour_mille**      | `float64`                | 0.0078318 (valeur initiale avec virgule dans le fichier, convertie en nombre à virgule flottante)        |
| **insee_pop**            | `int64`                  | 638425                                                                                                   |
| **insee_pop_millesime**  | `int64`                  | 2016                                                                                                     |
| **insee_log**            | `int64`                  | 308491                                                                                                   |
| **insee_log_millesime**  | `int64`                  | 2016                                                                                                     |

---
