# Correspondance entre critères et colonnes des fichiers

Cette section indique, pour chaque critère, s'il est présent et dans quel(s) fichier(s) ainsi que les colonnes associées.

---

1. **Taux de chômage**  
   - **Fichier** : `donnee-banque-france-2024-trimestre.csv`  
     - *Colonnes pertinentes* :  
       - `Libellé` (*ex. : Taux de chômage localisé par région*), `idBank`, `Dernière mise à jour`, `Période` (facultatif),  
       - Colonnes trimestrielles sous la forme `AAAA-TX` (*ex. : 2023-T4, 2024-T1*), où chaque valeur représente l'indicateur économique mesuré pour le trimestre correspondant.

2. **Criminalité**  
   - **Fichier** : `donnee-dep-data.gouv-2024-geographie2024-produit-le2025-01-26.csv`  
     - *Colonnes pertinentes* :  
       - `Code_departement`, `Code_region`, `annee`, `indicateur` (*ex. : Homicides, Usage de stupéfiants*), `unite_de_compte` (*ex. : Victime, Mis en cause*), `nombre`, `taux_pour_mille`, `insee_pop`, `insee_pop_millesime`, `insee_log`, `insee_log_millesime`.

3. **Revenu moyen par habitant**  
   - **Fichier** : `PIB_regionaux_1990-2022.xlsx`  
      - *Colonnes pertinentes* :  
         - `Région`  
         - `Année`  
         - `PIB` (*Produit Intérieur Brut régional en millions d'euros*)  
         - `PIB par habitant` (*PIB ajusté en fonction de la population*)  
         - `Taux de croissance` (*variation annuelle du PIB*)  

4. **Résultats des précédentes élections**  
   - **Fichier** : `resultats-par-niveau-cirlg-t2-france-entiere`  
     - *Colonnes pertinentes* :  
       - `Inscrits`, `Abstentions`, `% Abs/Ins`, `Votants`, `% Vot/Ins`, `Blancs`, `Nuls`, `Exprimés`, `% Exp/Ins`, `% Exp/Vot`, `Voix`, `% Voix/Ins`, `% Voix/Exp` (et les colonnes concernant les deux candidats).  
   - **Fichier** : `Presidentielle_2017_Resultats_Tour_2`  
     - *Colonnes pertinentes* :  
       - `Inscrits`, `Abstentions`, `% Abs/Ins`, `Votants`, `% Vot/Ins`, `Blancs`, `Nuls`, `Exprimés`, `% Exp/Ins`, `% Exp/Vot`, `Voix(1)/(2)`, `% Voix/Ins(1)/(2)`, `% Voix/Exp(1)/(2)`.

5. **Prix de l'immobilier au m²**  
   - **Fichier** : `Real Estate Indicators 2017.csv`  
     - *Colonnes pertinentes* :  
       - `INSEE_COM` (Code INSEE de la commune)
       - `Annee` (Année des transactions)
       - `Prixm2Moyen` (Prix moyen au mètre carré des biens vendus en €)
       - `PrixMoyen` (Prix moyen des biens immobiliers vendus en €)
       - `NbMaisons`, `NbApparts` (Nombre de transactions par type de bien)
       - `propmaison`, `propappart` (Proportion des types de biens en %)
       - `SurfaceMoy` (Surface moyenne des biens vendus en m²)

6. **Taux Immigration**  
   - **Fichier** : `EM_DEPARTEMENTS_2021.csv`
     - *Colonnes pertinentes* :
       - `Libellé` (nom du département)
       - `Code département`
       - `Part de la population immigrée (%)` 
   - **Fichier** : `BTX_TD_IMG1A_2017.csv`
     - *Colonnes pertinentes* :
       - `CODGEO`, `LIBGEO` (code et nom de la commune)
       - Colonnes détaillées par âge, sexe et statut d'immigration :
         - `AGE400_IMMI[1/2]_SEXE[1/2]` (population de moins de 4 ans)
         - `AGE415_IMMI[1/2]_SEXE[1/2]` (population de 4-15 ans)
         - *(et autres tranches d'âge similaires)*

7. **Âge moyen**  
   - **Fichier** : `population-departements-france-2024.csv`  
     - *Colonnes pertinentes* :  
       - `Départements`, `Nom_département`, `Total_ensemble`, `Ensemble_0_4_ans`, `Ensemble_5_9_ans`, `Ensemble_10_14_ans`, `Ensemble_15_19_ans`, `Ensemble_20_24_ans`, `Ensemble_25_29_ans`, `Ensemble_30_34_ans`, `Ensemble_35_39_ans`, `Ensemble_40_44_ans`, `Ensemble_45_49_ans`, `Ensemble_50_54_ans`, `Ensemble_55_59_ans`, `Ensemble_60_64_ans`, `Ensemble_65_69_ans`, `Ensemble_70_74_ans`, `Ensemble_75_79_ans`, `Ensemble_80_84_ans`, `Ensemble_85_89_ans`, `Ensemble_90_94_ans`, `Ensemble_95_ans_plus`.

8. **Densité de population**
   - **Fichier** : `DENSITE-2012.csv`, `DENSITE-2017.csv`, `DENSITE-2022.csv`

9. **Revenus moyens**
   - **Fichier** : `revenus_2012.csv`, `revenus_2017.csv`, `revenus_2022.csv`

10. **Taux de natalité**
   - **Fichier** : `mspr_natalite_2012.xlsx`, `mspr_natalite_2017.xlsx`, `mspr_natalite_2022.xlsx`
