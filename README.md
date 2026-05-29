# PredictDementia

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn%20%7C%20PyTorch-brightgreen)
![Status](https://img.shields.io/badge/Project-Portfolio-informational)

> Predicting dementia risk from patient health data using scikit-learn, PyTorch, and PyTorch Lightning.

## Why this project matters

Early risk prediction can help support faster clinical follow-up and better patient monitoring. In this project, I explore dementia prediction as a **binary classification** task using patient health and lifestyle data, and compare multiple machine learning workflows ranging from a classical baseline to deep learning.

This repository was built as a hands-on machine learning project to strengthen practical skills in:

- exploratory data analysis
- preprocessing structured health data
- feature engineering
- classical machine learning
- neural network training
- experiment organization with PyTorch Lightning

## Project summary

The project includes four main notebooks:

- **`understand_data.ipynb`** — explores the dataset using visualizations
- **`predict_dementia_scikit.ipynb`** — builds a scikit-learn logistic regression baseline
- **`predict_dementia_pytoch.ipynb`** — trains a PyTorch neural network model
- **`predict_dementia_pl.ipynb`** — refactors the PyTorch workflow using PyTorch Lightning

The goal is not just to make predictions, but to compare modeling approaches and understand how data preparation and training workflows affect results.

## Tech stack

- Python
- Jupyter Notebook
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- PyTorch
- torchmetrics
- Lightning

## Repository structure

- `understand_data.ipynb`  
  Exploratory data analysis of feature distributions and their relationship to dementia risk.

- `predict_dementia_scikit.ipynb`  
  Baseline classification pipeline using logistic regression.

- `predict_dementia_pytoch.ipynb`  
  Deep learning workflow implemented directly in PyTorch.

- `predict_dementia_pl.ipynb`  
  Cleaner and more maintainable PyTorch Lightning training workflow.

- `dementia_patients_health_data 2.csv`  
  Dataset used throughout the project.

- `image.png`  
  Setup screenshot used in earlier project documentation.

- `requirements.txt`  
  Python dependencies required to run the notebooks.

## Dataset

This project uses the **Dementia Patient Health Dataset** from Kaggle.

Source:
`https://www.kaggle.com/datasets/timothyadeyemi/dementia-patient-health-dataset/data`

The dataset contains patient-level information such as:

- demographics
- family history
- smoking status
- physical activity
- sleep quality
- nutrition
- prescription and medication history
- cognitive and health indicators
- APOE-ε4 status

These features are used to predict the target label: **whether a patient is classified as having dementia risk**.

## Modeling approaches

### 1. Exploratory Data Analysis
The EDA notebook investigates feature distributions and class relationships using seaborn and matplotlib visualizations.

### 2. Scikit-learn baseline
The scikit-learn notebook:

- loads the dataset into pandas
- preprocesses categorical columns
- handles missing values
- performs train/validation splitting
- applies feature scaling
- trains a logistic regression classifier
- reports accuracy, precision, recall, and F1-score

### 3. PyTorch neural network
The PyTorch notebook:

- applies one-hot encoding and scaling
- converts data to tensors
- uses `DataLoader` for batching
- defines a feedforward neural network
- trains over multiple epochs
- tracks training and validation performance

### 4. PyTorch Lightning workflow
The Lightning notebook reorganizes the PyTorch solution into a cleaner training framework with:

- modular training code
- built-in logging
- easier validation tracking
- better maintainability for future experiments

## Results and observations

The recorded notebook outputs show very strong performance, especially in the scikit-learn and Lightning workflows.

This is promising, but it also suggests a useful next step: validating whether performance remains strong under stricter testing conditions such as:

- cross-validation
- ROC-AUC analysis
- confusion matrix review
- leakage checks
- stronger train/test separation

## How to run the project

### 1. Clone the repository

```bash
git clone https://github.com/swchak/PredictDementia.git
cd PredictDementia
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Recommended notebook order:

1. `understand_data.ipynb`
2. `predict_dementia_scikit.ipynb`
3. `predict_dementia_pytoch.ipynb`
4. `predict_dementia_pl.ipynb`

## Key takeaways

This project demonstrates:

- end-to-end ML workflow development on tabular healthcare data
- comparison of classical ML vs neural networks
- practical preprocessing for mixed-type structured datasets
- training and evaluation in both raw PyTorch and Lightning
- iterative improvement of code quality and experimentation workflow

## Future improvements

Potential next steps include:

- adding ROC-AUC and confusion matrix evaluation
- introducing cross-validation
- testing additional models such as random forest or gradient boosting
- moving reusable preprocessing code into Python modules
- adding experiment tracking and reproducibility controls
- documenting feature definitions in more detail

## Author

Created by [@swchak](https://github.com/swchak)

## Disclaimer

This project is for educational and portfolio purposes only. It is **not** intended for clinical or diagnostic use.
