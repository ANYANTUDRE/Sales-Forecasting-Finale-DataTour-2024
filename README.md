# Challenge Finale DataTour 2024

## Prédiction des ventes futures pour une entreprise basée sur les ventes passées et des facteurs contextuels

### Contexte

L’Afrique connaît une dynamique économique croissante avec une diversité d'entreprises opérant dans des secteurs variés (commerce de détail, e-commerce, agroalimentaire, etc.). Cependant, l'un des défis majeurs pour les entreprises africaines reste la gestion des stocks et la prédiction des ventes futures.

Ce projet vise à **prédire les ventes futures d'une entreprise** en se basant sur les ventes passées et plusieurs facteurs contextuels influençant la demande, notamment les promotions, les jours fériés, les conditions météorologiques et la disponibilité des stocks. Une prédiction précise peut aider à mieux gérer l’inventaire, optimiser les promotions et ajuster les stratégies de prix pour maximiser les profits.

---

### Objectif

Prédire le nombre d'unités qui seront vendues entre le **2024-11-01 et le 2024-11-30**, à partir des données historiques de ventes d'une entreprise couvrant la période du **2022-01-01 au 2024-10-31**.

---

### Structure des Données

#### Fichier d’entraînement (`train.csv`):

- **Taille** : 83 047 lignes.
- **Colonnes** : Inclut toutes les colonnes, y compris la cible `quantite_vendue`.
- **Source** : [train.csv](https://raw.githubusercontent.com/dataafriquehub/donnee_vente/refs/heads/main/train.csv)

#### Fichier de soumission (`submission.csv`):

- **Taille** : 2 576 lignes.
- **Colonnes** : Contient toutes les caractéristiques sauf la colonne cible `quantite_vendue`.
- **Source** : [submission.csv](https://raw.githubusercontent.com/dataafriquehub/donnee_vente/refs/heads/main/submission.csv)

---

### Description des Colonnes

| **Variable**                               | **Description**                                                                               |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| **ID Produit (id\_produit)**               | Identifiant unique pour chaque produit vendu.                                                 |
| **Date (date)**                            | La date de la vente, allant du 1er janvier 2022 au 31 octobre 2024.                           |
| **Catégorie (categorie)**                  | La catégorie du produit (ex. Électronique, Habillement, Alimentaire).                         |
| **Marque (marque)**                        | La marque du produit (ex. Samsung, Nike, Nestlé, etc.).                                       |
| **Prix Unitaire (prix\_unitaire)**         | Le prix de vente par unité du produit.                                                        |
| **Promotion (promotion)**                  | Indicateur binaire (0 ou 1) indiquant si le produit était en promotion au moment de la vente. |
| **Jour Férié (jour\_ferie)**               | Indicateur binaire (0 ou 1) indiquant si la vente a eu lieu un jour férié.                    |
| **Week-end (weekend)**                     | Indicateur binaire (0 ou 1) indiquant si la vente a eu lieu un week-end.                      |
| **Stock Disponible (stock\_disponible)**   | Nombre d’unités disponibles pour la vente ce jour-là.                                         |
| **Condition Météo (condition\_meteo)**     | Type de condition météo le jour de la vente (ex. Ensoleillé, Pluie, Orageux, Neigeux).        |
| **Région (region)**                        | Région géographique de la vente (ex. Urbain, Périurbain, Rural).                              |
| **Moment de la Journée (moment\_journee)** | Le moment de la journée où la vente a eu lieu (ex. Matinée, Après-midi, Soirée).              |
| **Quantité Vendue (quantite\_vendue)**     | Le nombre d'unités réellement vendues de chaque produit (pour l'entraînement seulement).      |

*Note :* La **Quantité Vendue** est la variable cible à prédire pour novembre 2024.

---

### Critères d'évaluation

Les prédictions seront évaluées à l'aide de la **Mean Absolute Percentage Error (MAPE)** :

- Les équipes avec les MAPE les plus faibles (indiquant une meilleure précision) seront classées en tête.
- Trois soumissions par jour au maximum sont autorisées par équipe.

---

### Format de la soumission

Un fichier CSV contenant les colonnes suivantes :

- **`id`** : Identifiant de chaque ligne dans `submission.csv`.
- **`quantite_vendue`** : Prédiction de la quantite\_vendue pour chaque ligne.

Exemple de format attendu :

```csv
id,quantite_vendue
1,320
2,7500
3,1000
...
```

---