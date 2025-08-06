# Estructure of the folders

pneumonia-dx/
│
├── .venv/                        # Virtual environment
│
├── 📁 data/                      # Datasets
│   ├── raw/                     # Raw data (original chest X-rays)
│   └── processed/               # Preprocessed and augmented data
│
├── 📁 notebooks/                 # Jupyter notebooks for analysis
│   └── eda_model_exploration.ipynb
│
├── 📁 models/                    # Trained models (.h5, .pt, etc.)
│   └── best_model.h5
│
├── 📁 reports/                   # Evaluation results, plots, confusion matrices
│   └── confusion_matrix.png
│
├── 📁 src/                       # Source code
│   ├── data_loader/             # Data loading and preprocessing
│   │   └── dataset.py
│   ├── training/                # Training loop, metrics, validation
│   │   └── train.py
│   ├── utils/                   # Utility functions (visualization, helpers)
│   │   └── helpers.py
│   └── interface/               # Interface code (e.g., Streamlit, Flask)
│       └── app.py
│
├── 📁 scripts/                   # Automation scripts
│   └── download_dataset.py      # Script to download Kaggle dataset
│
├── 📄 requirements.txt          # Project dependencies
├── 📄 README.md                 # Documentation and project overview
└── 📄 LICENSE                   # License file



# Traduction of leguage 🌐

| Español            | Inglés técnico sugerido             |
| ------------------ | ----------------------------------- |
| Interfaz gráfica   | `interface/` o `ui/`                |
| Script             | `main.py`, `run.py`, `predictor.py` |
| Modelo entrenado   | `models/`                           |
| Datos procesados   | `data/processed/`                   |
| Datos sin procesar | `data/raw/`                         |
| Cuaderno Jupyter   | `notebooks/`                        |
| Funciones comunes  | `utils/`                            |
| Pruebas            | `tests/`                            |




chest_xray
    _MACOSX
    test 
    train
    val