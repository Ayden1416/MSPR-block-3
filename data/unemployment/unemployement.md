# Documentation du fichier

Ce document présente le format et la structure des données du fichier suivant :  
**Fichier** : `donnee-banque-france-2024-trimestre.csv`

---

## Description du format du fichier

Le fichier contient des données économiques organisées sous la forme d'un tableau où chaque ligne représente une observation spécifique (exemple : taux de chômage localisé par région). Les colonnes du fichier peuvent être regroupées en deux catégories principales :

### 1. Métadonnées
Ces colonnes décrivent les informations contextuelles des observations :
- **Libellé** (`string`) : Intitulé de l'indicateur économique.
- **idBank** (`string`) : Identifiant unique de la donnée.
- **Dernière mise à jour** (`datetime`) : Date de la dernière mise à jour de l'indicateur.
- **Période** (`string`, facultatif) : Peut contenir des informations sur la période de référence de la donnée.

### 2. Données trimestrielles
- Les autres colonnes correspondent aux périodes temporelles sous la forme `AAAA-TX`, où `AAAA` représente l'année et `TX` indique le trimestre (`T1`, `T2`, `T3`, `T4`).
- Chaque valeur numérique représente l'indicateur économique mesuré pour le trimestre correspondant.
- Certaines valeurs sont suivies de `(A)` ou `(SD)`, qui indiquent le statut ou la qualité de la donnée.

Exemple de colonnes temporelles :
- **1982-T1** (`float64`) : Valeur de l'indicateur pour le premier trimestre de 1982.
- **2024-T1** (`float64`) : Valeur de l'indicateur pour le premier trimestre de 2024.

---

## Exemple d'enregistrement
| Libellé                                      | idBank    | Dernière mise à jour | 2023-T4 | 2024-T1 |
|----------------------------------------------|----------|----------------------|---------|---------|
| Taux de chômage localisé par région - Île-de-France | 001515843 | 19/12/2024 12:00    | 7.1 (A) | 7.2 (A) |

---

Ce format permet d'analyser l'évolution d'un indicateur économique au fil du temps, en exploitant les valeurs trimestrielles des différentes périodes enregistrées.