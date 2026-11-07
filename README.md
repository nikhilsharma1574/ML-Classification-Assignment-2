# Bank Marketing Classification Dashboard

This project contains a Streamlit dashboard for evaluating and comparing multiple trained machine learning models for the Bank Marketing classification task.

## Project Overview

The app loads saved model files from the `model/` directory and compares them on an uploaded CSV dataset. It includes:

- Model comparison table
- Accuracy, AUC, Precision, Recall, F1 Score, and MCC metrics
- Individual model evaluation
- Confusion matrix
- Classification report
- Model comparison chart
- CSV export of model results

## Included Models

- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Support Vector Machine

## Files in the Repository

- `app.py` – Streamlit dashboard application
- `requirements.txt` – Python dependencies
- `feature_columns.json` – Ordered feature list used for model input
- `model/` – Saved trained models in joblib format
- `test_data.csv` – Sample test dataset
- `model_results.csv` – Generated comparison results
- `model_training.ipynb` – Training notebook

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt