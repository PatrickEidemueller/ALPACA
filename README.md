ALPACA
==============================

"**A**ctive-**L**earning **P**ipeline **AC**ronym: **A**LPACA"

Setup:

1. `conda env create -f environment.yml`
2. `conda activate alpaca-venv`
3. `pip install -r requirements.txt`

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries (mol2vec)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── sessions           <- Results of experiments conducted with alpaca are written to this folder
    |                         do not edit  manually
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so alpaca and src can be imported
    ├── alpaca             <- Source code of the alpaca pipeline framework and experiments
    |  |
    |  ├── system          <- Base framwork
    |  ├── experiments     <- Implementations of selectors and models for the pipeline
    |  ├── visualization   <- Functions for plotting performance and selected points
    |  ├── vis2d.py        <- CLI for visualizing data point selection
    |  └── runner.py       <- CLI for running experiments and plotting performance
    └── src                <- Source code of alpaca framework.
        │
        ├── smiles_to_feature_vector.py <- Code to convert SMILES to feature vectors using mol2vec
        ├── data_generator.py           <- Code to create synthetic datasets  
        └── models         <- Contains initial implementations of algorithms before they were added to the pipeline
                              Also contains some experimental code that was not incorporated in the pipeline


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
