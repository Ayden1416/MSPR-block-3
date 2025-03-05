
## 📄 Documentation des fichiers

Ce document présente le format et la structure des données des fichiers suivants :
- **Fichier** : `Niveau_de_vie_mediant_2017.xlsx`
- **Fichier** : `Niveau_de_vie_mediant_2022.xlsx`

---

## 📝 Description du format des fichiers

### 📂 1. `Niveau_de_vie_mediant_2017.xlsx`
Ce fichier contient des données sur le niveau de vie médiant des ménages par département en 2017.

### 📂 2. `Niveau_de_vie_mediant_2022.xlsx`
Ce fichier contient des données sur le niveau de vie médiant des ménages par département en 2022.

#### **Caractéristiques du fichier :**
- **Champ** : France entière par département
- **Source** : Insee, recensement de la population 2021 (exploitation principale)
- **Spécificité** : Données réajustées. Recensement de la population 2017 utilisé pour Mayotte.

#### **Colonnes :**
1. **Libellé** (`string`) : Nom du département.
2. **Niveau de vie mediant** (`string`) : revenu mediant des habitants

#### **Exemple d'enregistrement :**
| Département       | Revenu mediant des habitants           |
|-------------------|----------------------------------------|
| Ain               | 22640                                  |
| Savoie            | 22460                                  |
| Isère             | 22260                                  |