# gmtames

A **g**rouped **m**ulti**t**ask deep learning approach for the prediction of **Ames** mutagenicity.

This Git repository contains data, source code, and results for the research article *'Mechanistic task groupings enhance multitask deep learning of strain-specific Ames mutagenicity'* (currently under review).


# Project file tree
```bash
gmtames/
├── execute/
│   ├── dataset_analysis.bat
│   ├── dataset_preparation.bat
│   └── mtg_experiment.bat
├── gmtames/
│   ├── data/
│   │   ├── ad_datasets/
│   │   │   ├── ad_calc/
│   │   │   │   ├── { ... Applicability Domain toolbox for MATLAB files ... }
│   │   │   │   └── gmtamesAD_script.m
│   │   │   ├── ad_results/
│   │   │   │   ├── { ... *_test.csv ... }
│   │   │   │   └── { ... *_val.csv ... }
│   │   │   ├── { ... gmtamesAD_*_test.csv ... }
│   │   │   ├── { ... gmtamesAD_*_train.csv ... }
│   │   │   ├── { ... gmtamesAD_*_trainval.csv ... }
│   │   │   └── { ... gmtamesAD_*_val.csv ... }
│   │   ├── base_datasets/
│   │   │   ├── { ... gmtamesQSAR_*_test.csv ... }
│   │   │   ├── { ... gmtamesQSAR_*_train.csv ... }
│   │   │   └── { ... gmtamesQSAR_*_val.csv ... }
│   │   └── master_datasets/
│   │       ├── gmtamesQSAR_endpoints.csv
│   │       └── gmtamesQSAR_fingerprints.csv
│   ├── __main__.py
│   ├── data.py
│   ├── logging.py
│   ├── mtg.py
│   ├── nn.py
│   └── results.py
├── output/
│   ├── dataset_analysis/
│   │   ├── gmtamesQSAR_descriptive_stats.csv
│   │   └── gmtamesQSAR_heatmaps.svg
│   └── mtg_experiment/
│       ├── final_models/
│       │   ├── { ... *_hyperparam_dict.json ... }
│       │   └── { ... *_state_dict.pt ... }
│       ├── logs/
│       │   └── { ... gmtames_*.log ... }
│       ├── results/
│       │   ├── gmtames_curated_average_results.csv
│       │   ├── gmtames_curated_frameshift_results.csv
│       │   ├── gmtames_curated_substitution_results.csv
│       │   └── gmtames_full_experimental_results.csv
│       └── test_predictions/
│           └── { ... *_test_predictions.pkl ... }
├── .gitignore
├── gmtames.yml
└── README.md
```

# To do
- [ ] Add master dataset curation KNIME workflow and files to `gmtames/gmtames/data/master_datasets/`
- [ ] Add protocol (code summary) section to `README.md`