## 📄 Documentation des fichiers

Ce document présente le format et la structure des données du fichier suivant :  
- **Fichier** : `Real Estate Indicators 2017.csv`

---

## 📝 Description du format du fichier

### 📂 1. `Real Estate Indicators 2017.csv`
Ce fichier contient des indicateurs immobiliers pour l'année 2017, regroupés par code INSEE des communes.

#### **Caractéristiques du fichier :**
- **Champ** : France entière
- **Année de référence** : 2017
- **Source** : Données immobilières issues des transactions recensées
- **Spécificité** : Données agrégées par commune, incluant le nombre de mutations, les types de biens concernés et les prix moyens.

#### **Colonnes :**
1. **INSEE_COM** (`int`) : Code INSEE de la commune.
2. **Annee** (`int`) : Année des transactions immobilières.
3. **Nb_mutations** (`int`) : Nombre total de mutations (transactions immobilières) enregistrées.
4. **NbMaisons** (`int`) : Nombre de transactions portant sur des maisons.
5. **NbApparts** (`int`) : Nombre de transactions portant sur des appartements.
6. **propmaison** (`float64`) : Proportion des maisons dans les transactions (%).
7. **propappart** (`float64`) : Proportion des appartements dans les transactions (%).
8. **PrixMoyen** (`float64`) : Prix moyen des biens immobiliers vendus (€).
9. **Prixm2Moyen** (`float64`) : Prix moyen au mètre carré des biens vendus (€).
10. **SurfaceMoy** (`float64`) : Surface moyenne des biens vendus (m²).

#### **Exemple d'enregistrement :**
| INSEE_COM | Annee | Nb_mutations | NbMaisons | NbApparts | propmaison | propappart | PrixMoyen  | Prixm2Moyen | SurfaceMoy |
|-----------|--------|-------------|-----------|-----------|------------|------------|------------|-------------|------------|
| 1001      | 2017   | 8           | 8         | 0         | 100.0      | 0.0        | 179955.0   | 1902.0      | 93.0       |
| 1002      | 2017   | 2           | 2         | 0         | 100.0      | 0.0        | 77750.0    | 932.5       | 80.0       |
| 1004      | 2017   | 99          | 61        | 38        | 61.62      | 38.38      | 167367.52  | 1995.91     | 85.27      |
| 1005      | 2017   | 13          | 6         | 7         | 46.15      | 53.85      | 199373.85  | 1967.31     | 108.38     |
| 1007      | 2017   | 16          | 16        | 0         | 100.0      | 0.0        | 200008.25  | 1693.31     | 125.63     |
| 1008      | 2017   | 7           | 7         | 0         | 100.0      | 0.0        | 155068.57  | 1807.43     | 85.43      |
| 1009      | 2017   | 2           | 1         | 1         | 50.0       | 50.0       | 83000.0    | 1176.5      | 69.5       |

---