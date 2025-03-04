
## 📄 Documentation des fichiers

Ce document présente le format et la structure des données des fichiers suivants :  
- **Fichier** : `EM_DEPARTEMENTS_2021.csv`
- **Fichier** : `BTX_TD_IMG1A_2017.csv`

---

## 📝 Description du format des fichiers

### 📂 1. `EM_DEPARTEMENTS_2021.csv`
Ce fichier contient des données sur la part de la population immigrée par département en 2021.  

#### **Caractéristiques du fichier :**
- **Champ** : France entière
- **Source** : Insee, recensement de la population 2021 (exploitation principale)
- **Spécificité** : Données réajustées. Recensement de la population 2017 utilisé pour Mayotte.

#### **Colonnes :**
1. **Libellé** (`string`) : Nom du département.
2. **Code département** (`string`) : Code administratif du département.
3. **Part de la population immigrée (%)** (`float64`) : Pourcentage d'immigrés dans la population totale du département.

#### **Exemple d'enregistrement :**
| Département       | Code | Part de la population immigrée (%) |
|------------------|------|-----------------------------------|
| Cantal          | 15   | 2.67                              |
| Charente        | 16   | 6.16                              |
| Charente-Maritime | 17 | 3.81                              |
| Côte-d'Or       | 21   | 7.61                              |

---

### 📂 2. `BTX_TD_IMG1A_2017.csv`
Ce fichier présente des tableaux détaillés sur la population selon le sexe, l'âge et la situation quant à l'immigration.  

#### **Caractéristiques du fichier :**
- **Champ** : France hors Mayotte
- **Niveau géographique** : Communes
- **Mise en ligne** : 29/06/2020
- **Géographie au** : 01/01/2020
- **Source** : Insee, RP2017 exploitation principale.

#### **Colonnes :**
1. **CODGEO** (`string`) : Code géographique de la commune.
2. **LIBGEO** (`string`) : Nom de la commune.
3. **AGE4** (`int`) : Groupe d'âge (00, 15, 25, 55).
4. **IMMI** (`int`) : Statut d'immigration (1 = Non immigré, 2 = Immigré).
5. **SEXE** (`int`) : Sexe (1 = Homme, 2 = Femme).
6. **AGE400_IMMI1_SEXE1** (`int`) : Nombre d'hommes non immigrés de moins de 4 ans.
7. **AGE400_IMMI1_SEXE2** (`int`) : Nombre de femmes non immigrées de moins de 4 ans.
8. **AGE400_IMMI2_SEXE1** (`int`) : Nombre d'hommes immigrés de moins de 4 ans.
9. **AGE400_IMMI2_SEXE2** (`int`) : Nombre de femmes immigrées de moins de 4 ans.
10. **AGE415_IMMI1_SEXE1** (`int`) : Nombre d'hommes non immigrés entre 4 et 15 ans.
11. **AGE415_IMMI1_SEXE2** (`int`) : Nombre de femmes non immigrées entre 4 et 15 ans.
12. *(Les autres colonnes suivent la même logique pour les âges 25 et 55 ans.)*

#### **Exemple d'enregistrement :**
| CODGEO | LIBGEO                 | AGE400_IMMI1_SEXE1 | AGE400_IMMI1_SEXE2 | AGE400_IMMI2_SEXE1 | AGE400_IMMI2_SEXE2 | AGE415_IMMI1_SEXE1 | AGE415_IMMI1_SEXE2 | AGE415_IMMI2_SEXE1 | AGE415_IMMI2_SEXE2 |
|--------|-------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 01001  | L'Abergement-Clémenciat  | 0                  | 0                  | 84                 | 77                 | 1                  | 0                  | 39                 | 29                 |
| 01002  | L'Abergement-de-Varey    | 0                  | 0                  | 27                 | 30                 | 0                  | 0                  | 5                  | 10                 |
| 01004  | Ambérieu-en-Bugey        | 59                 | 43                 | 1330               | 1293               | 87                 | 86                 | 880                | 826                |

---