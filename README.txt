ML-Balance-Sheet-Prediction
Automate and improve the accuracy of a company’s balance sheet structure forecasts by developing a machine learning model capable of simultaneously leveraging: • The financial history specific to the target company (balance sheet, income statement, cash flows, etc.), • The common dynamics observed within a large sample of companies.

Project Structure : 
balnce-sheet-ml/
│
├── data/
│   ├── raw/               # raw JSON filings (local only, .gitignore) import it from the zip shared by Yousra
│   ├── interim/           # intermediate cleaned files
│   └── processed/         # final parquet/csv ready for ML
│
├── notebooks/             # Jupyter notebooks for EDA, prototyping
│
├── src/                   # reusable Python modules
│   ├── ingestion.py       # load JSON → DataFrame
│   ├── cleaning.py        # drop duplicates, fix types
│   └── normalisation.py   # scale, standardize, reconcile
│   └── merge.py           # merge after parsing
│   └── pivoting.py        
│
│
├── docs/                   # livrables finales
│
├── requirements.txt       # dependencies
├── README.md              # project overview
└── .gitignore             # ignore venv, raw data, cache

🚀 Étapes rapides
Cloner le repo

bash
git clone https://github.com/Fareshd049/balance-sheet-ml.git
cd balance-sheet-ml
Installer les dépendances

bash
pip install -r requirements.txt
Explorer la structure

src/ → scripts modulaires (ingestion, cleaning, normalisation, pivoting, merge).

data/processed/ → fichiers prêts pour ML (wide_norm.parquet).

notebooks/ → notebooks EDA et prototypage.

docs/ → livrables finaux (rapports, figures).

Collaborer

Créez une branche pour vos ajouts.

Faites un pull request pour review.

Utilisez Issues pour suivre les tâches.

Structure détaillée
data/
raw/

Usage : contient les fichiers JSON bruts des filings SEC. (partagé par Yousra)

Ordre : première étape du pipeline (source).

Output : données brutes, non versionnées sur GitHub (protégées par .gitignore).

interim/

Usage : fichiers intermédiaires après parsing et cleaning (par ex. CSV ou parquet partiels).

Ordre : après ingestion et cleaning, avant normalisation.

Output : tables semi‑propres, utilisées pour debug ou étapes intermédiaires.

processed/

Usage : données finales prêtes pour ML (parquet/csv).

Ordre : dernière étape du pipeline (après normalisation et pivoting).

Output : wide_norm.parquet, features.csv, etc. → base pour EDA et modélisation.

notebooks/
Usage : Jupyter notebooks pour EDA, prototypage, visualisations.

Ordre : après que les données soient dans processed/.

Output : graphiques, analyses exploratoires, rapports interactifs.

src/
Modules Python réutilisables, chacun correspond à une étape du pipeline :

ingestion.py

Usage : parse les JSON bruts → DataFrame.

Ordre : première étape (raw → interim).

Output : DataFrame brut, sauvegardé en parquet dans interim/.

cleaning.py

Usage : nettoyage (drop duplicates, correction de types, harmonisation des colonnes).

Ordre : après ingestion.

Output : fichiers propres dans interim/.

merge.py

Usage : fusionner plusieurs tables ou sources (par ex. filings multiples par entreprise).

Ordre : après cleaning, avant pivoting.

Output : table consolidée dans interim/.

pivoting.py

Usage : transformer format long → format wide (concepts en colonnes, périodes en lignes).

Ordre : après merge.

Output : parquet wide directement dans processed/.

normalisation.py

Usage : imputation, règles comptables, MICE, log transform.

Ordre : dernière étape de préparation.

Output : wide_norm.parquet dans processed/, prêt pour ML et EDA.

docs/
Usage : livrables finaux (rapports, figures, documentation technique).

Ordre : après EDA et modélisation.

Output : PDF, slides, notes pour l’équipe ou stakeholders.

Fichiers racine
requirements.txt

Usage : liste des dépendances Python (pandas, numpy, scikit-learn, pyarrow, statsmodels, seaborn, etc.).

Ordre : installation avant d’exécuter le pipeline.

Output : environnement reproductible.

README.md

Usage : documentation du projet (objectif, étapes, structure).

Ordre : consulté en premier par tes coéquipiers.

Output : guide d’utilisation.

.gitignore

Usage : exclure data/raw/, fichiers volumineux, venv, caches.

Ordre : toujours actif.

Output : repo propre, sans données sensibles ou lourdes.

🔄 Ordre d’exécution du pipeline
Ingestion (src/ingestion.py) → parse JSON → data/interim/ingested.parquet.

Cleaning (src/cleaning.py) → corrige types, supprime doublons → data/interim/cleaned.parquet.

Merge (src/merge.py) → fusionne tables → data/interim/merged.parquet.

Pivoting (src/pivoting.py) → long → wide → data/interim/wide.parquet.

Normalisation (src/normalisation.py) → imputation + log transform → data/processed/wide_norm.parquet.

EDA (notebooks/EDA.ipynb) → analyse exploratoire → figures + insights → docs/eda_report.pdf.








