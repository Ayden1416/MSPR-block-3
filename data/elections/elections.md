# Documentation des fichiers

Ce document présente la liste des colonnes et leur type de données respectif pour les deux fichiers suivants :  
1. **resultats-par-niveau-cirlg-t2-france-entiere**  
2. **Presidentielle_2017_Resultats_Tour_2**

---

## 1. Fichier : resultats-par-niveau-cirlg-t2-france-entiere

**Tableau des colonnes et types estimés :**

| Nom de la colonne                  | Type de données (pandas) | Exemple / Commentaire                                  |
|------------------------------------|--------------------------|--------------------------------------------------------|
| **Code du département**            | `object` / `string`      | « 01 » (chaîne de caractères, car contient des zéros)  |
| **Libellé du département**         | `object` / `string`      | « Ain »                                               |
| **Code de la circonscription**     | `object` / `string`      | « 01 » ou « 2 »                                       |
| **Libellé de la circonscription**  | `object` / `string`      | « 1ère circonscription »                              |
| **Etat saisie**                    | `object` / `string`      | « Complet »                                           |
| **Inscrits**                       | `int64`                  | 85742                                                 |
| **Abstentions**                    | `int64`                  | 20298                                                 |
| **% Abs/Ins**                      | `float64`                | 23.67                                                 |
| **Votants**                        | `int64`                  | 65444                                                 |
| **% Vot/Ins**                      | `float64`                | 76.33                                                 |
| **Blancs**                         | `int64`                  | 4020                                                  |
| **% Blancs/Ins**                   | `float64`                | 4.69                                                  |
| **% Blancs/Vot**                   | `float64`                | 6.14                                                  |
| **Nuls**                           | `int64`                  | 1229                                                  |
| **% Nuls/Ins**                     | `float64`                | 1.43                                                  |
| **% Nuls/Vot**                     | `float64`                | 1.88                                                  |
| **Exprimés**                       | `int64`                  | 60195                                                 |
| **% Exp/Ins**                      | `float64`                | 70.2                                                  |
| **% Exp/Vot**                      | `float64`                | 91.98                                                 |
| **N°Panneau**                      | `int64`                  | 1                                                     |
| **Sexe**                           | `object` / `string`      | « M » ou « F »                                        |
| **Nom**                            | `object` / `string`      | « MACRON »                                           |
| **Prénom**                         | `object` / `string`      | « Emmanuel »                                         |
| **Voix**                           | `int64`                  | 32671                                                |
| **% Voix/Ins**                     | `float64`                | 38.1                                                 |
| **% Voix/Exp**                     | `float64`                | 54.28                                                |
| **N°Panneau(2)**                   | `int64`                  | 2                                                     |
| **Sexe(2)**                        | `object` / `string`      | « F »                                                |
| **Nom(2)**                         | `object` / `string`      | « LE PEN »                                           |
| **Prénom(2)**                      | `object` / `string`      | « Marine »                                           |
| **Voix(2)**                        | `int64`                  | 27524                                                |
| **% Voix/Ins(2)**                  | `float64`                | 32.1                                                 |
| **% Voix/Exp(2)**                  | `float64`                | 45.72                                                |

---

## 2. Fichier : Presidentielle_2017_Resultats_Tour_2
…

**Tableau des colonnes et types estimés :**

| Nom de la colonne          | Type de données (pandas) | Exemple / Commentaire                             |
|----------------------------|--------------------------|---------------------------------------------------|
| **Code du département**    | `int64`                  | 1                                                 |
| **Libellé du département** | `object` / `string`      | « Ain »                                           |
| **Inscrits**               | `int64`                  | 415952                                            |
| **Abstentions**            | `int64`                  | 93135                                             |
| **% Abs/Ins**              | `float64`                | 22.39                                             |
| **Votants**                | `int64`                  | 322817                                            |
| **% Vot/Ins**              | `float64`                | 77.61                                             |
| **Blancs**                 | `int64`                  | 28467                                             |
| **% Blancs/Ins**           | `float64`                | 6.84                                              |
| **% Blancs/Vot**           | `float64`                | 8.82                                              |
| **Nuls**                   | `int64`                  | 9122                                              |
| **% Nuls/Ins**             | `float64`                | 2.19                                              |
| **% Nuls/Vot**             | `float64`                | 2.83                                              |
| **Exprimés**               | `int64`                  | 285228                                            |
| **% Exp/Ins**              | `float64`                | 68.57                                             |
| **% Exp/Vot**              | `float64`                | 88.36                                             |
| **Sexe(1)**                | `object` / `string`      | « M »                                             |
| **Nom(1)**                 | `object` / `string`      | « MACRON »                                        |
| **Prénom(1)**              | `object` / `string`      | « Emmanuel »                                      |
| **Voix(1)**                | `int64`                  | 173832                                            |
| **% Voix/Ins(1)**          | `float64`                | 41.79                                             |
| **% Voix/Exp(1)**          | `float64`                | 60.94                                             |
| **Sexe(2)**                | `object` / `string`      | « F »                                             |
| **Nom(2)**                 | `object` / `string`      | « LE PEN »                                        |
| **Prénom(2)**              | `object` / `string`      | « Marine »                                        |
| **Voix(2)**                | `int64`                  | 111396                                            |
| **% Voix/Ins(2)**          | `float64`                | 26.78                                             |
| **% Voix/Exp(2)**          | `float64`                | 39.06                                             |

---