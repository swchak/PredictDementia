# PredictDementia

A machine learning project for predicting dementia risk from patient health data using multiple modeling approaches in Python and Jupyter notebooks.

## Overview

This repository explores dementia prediction as a binary classification problem using a patient health dataset.  
The project includes:

- exploratory data analysis (EDA)
- a baseline model built with scikit-learn
- a neural network implementation in PyTorch
- a refactored training workflow using PyTorch Lightning

The goal of the project is to compare different approaches for the same prediction task while learning and practicing core machine learning workflows such as preprocessing, feature engineering, model training, and evaluation.

## Repository Structure

- `understand_data.ipynb`  
  Exploratory data analysis of the dataset using pandas, seaborn, and matplotlib.

- `predict_dementia_scikit.ipynb`  
  Baseline binary classification workflow using scikit-learn logistic regression.

- `predict_dementia_pytoch.ipynb`  
  Neural network implementation in PyTorch for dementia prediction.

- `predict_dementia_pl.ipynb`  
  PyTorch Lightning version of the PyTorch workflow with a cleaner training loop and metric logging.

- `dementia_patients_health_data 2.csv`  
  Dataset used for training and evaluation.

- `image.png`  
  Screenshot used in the project setup walkthrough.

## Dataset

This project uses the **Dementia Patient Health Dataset** from Kaggle.

Dataset source:  
`https://www.kaggle.com/datasets/timothyadeyemi/dementia-patient-health-dataset/data`

The dataset contains patient health and lifestyle information that can be used to predict whether a patient is at risk of dementia.

Example feature groups used in the notebooks include:

- demographic information
- family history
- smoking status
- physical activity
- nutrition and sleep quality
- prescription and medication history
- cognitive and health-related indicators
- APOE-ε4 status

## Approaches Implemented

### 1. Exploratory Data Analysis
The `understand_data.ipynb` notebook loads the dataset and visualizes relationships between features and the target variable using seaborn and matplotlib.

### 2. Scikit-learn Baseline
The `predict_dementia_scikit.ipynb` notebook:

- loads the dataset into a pandas DataFrame
- preprocesses categorical variables
- handles missing values
- scales features
- splits the data into train and validation sets
- trains a logistic regression classifier
- reports classification metrics such as accuracy, precision, recall, and F1-score

### 3. PyTorch Model
The `predict_dementia_pytoch.ipynb` notebook:

- preprocesses the data with one-hot encoding and scaling
- converts the dataset into PyTorch tensors
- creates `DataLoader` objects
- defines a feedforward neural network
- trains the model across epochs
- tracks training and validation loss
- evaluates predictive performance

### 4. PyTorch Lightning Version
The `predict_dementia_pl.ipynb` notebook refactors the PyTorch workflow using PyTorch Lightning to provide:

- cleaner training and validation loops
- better experiment organization
- built-in metric logging
- easier scaling and maintenance

## Results

The notebooks include evaluation metrics for each modeling approach.

Examples of reported metrics include:

- accuracy
- precision
- recall
- F1-score
- validation loss

The scikit-learn notebook currently shows very high validation performance in its recorded output. This may be worth investigating further to confirm whether the preprocessing pipeline, feature set, or dataset characteristics make the task unusually easy.

## Requirements

To run the notebooks locally, you should have:

- Python 3.11+
- Jupyter Notebook or JupyterLab
- pip
- a virtual environment tool such as `venv`

Suggested Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- torch
- torchmetrics
- lightning
- ipykernel

## Setup

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
pip install pandas numpy matplotlib seaborn scikit-learn torch torchmetrics lightning jupyter ipykernel
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Then open any of the notebooks from the repository root.

## How to Run

A good order for exploring the project is:

1. `understand_data.ipynb`
2. `predict_dementia_scikit.ipynb`
3. `predict_dementia_pytoch.ipynb`
4. `predict_dementia_pl.ipynb`

This order helps you move from data understanding to baseline modeling and then to deep learning workflows.

## Notes

- The PyTorch notebook filename currently contains a typo: `predict_dementia_pytoch.ipynb` instead of `predict_dementia_pytorch.ipynb`.
- The repository appears to be notebook-first, so reproducibility could be improved further by adding a `requirements.txt` file.
- Since this is a health-related prediction project, the model outputs should be treated as educational/experimental rather than medical advice.

## Future Improvements

Some useful next steps for the project could include:

- adding a `requirements.txt` or `environment.yml`
- splitting preprocessing into reusable utility functions
- adding ROC-AUC and confusion matrix evaluation
- performing cross-validation
- testing additional models such as random forest or XGBoost
- improving experiment tracking
- renaming notebook files for consistency
- documenting feature definitions and target labels more clearly

## Author

Created by [@swchak](https://github.com/swchak)

## Acknowledgments

- Kaggle dataset contributors
- scikit-learn
- PyTorch
- PyTorch Lightning
- pandas, seaborn, and matplotlib
