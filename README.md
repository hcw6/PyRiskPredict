# PyRiskPredict: Simple Patient Risk Predictor
## Overview
PyRiskPredict is a single-member-team Python project focused on building and evaluating simple machine learning models to predict patient risk based on tabular clinical-like datasets. This project serves as a learning exercise and the basis for the "Final Project" assignment.

The primary goal is to implement a pipeline that includes data loading, preprocessing, model training, evaluation, and prediction using standard Python data science libraries.

## General Software Architecture
The project will follow a modular structure to separate concerns:

<ol>
<li> Data Loading & Preprocessing (data/ & src/data_processing.py):

  * Scripts/functions to load datasets.
  * Functions for data cleaning, feature engineering, feature encoding, and feature scaling.
  * Responsible for preparing the data into a suitable format for model training.
<li> Model Training (src/training.py):

* Functions to split the processed data into training and testing sets.
* Implementation of model training using algorithms from scikit-learn.
* Logic for saving trained models.
<li> Model Evaluation (src/evaluation.py):

* Functions to evaluate the performance of trained models on the test set.
* Calculation of relevant metrics (e.g., accuracy, precision, recall, F1-score, AUC).
* Potentially includes functions for generating confusion matrices or ROC curves.
<li> Prediction (src/predict.py):

* Functionality to load a pre-trained model.
* Ability to take new, unseen data (appropriately preprocessed) and generate risk predictions.
<li> Main Script / Orchestration (main.py or run_pipeline.py):

* A main script to execute the end-to-end pipeline: Load -> Preprocess -> Train -> Evaluate.
* May utilize command-line arguments (using argparse) to control aspects like dataset path or model choice.
<li><s> Configuration (config.py or config.yaml):</s> Skipped due to time contraints

* <s>A place to store configuration variables like file paths, model parameters, random seeds, etc., to avoid hardcoding.
</s></ol>

## Tech Stack
* Language: Python 3.x
* Core Libraries:
  * pandas 1.5 or greater : For data loading and manipulation.
  * scikit-learn 1.2 or greater : For machine learning tasks (preprocessing, modeling, evaluation).
  * matplotlib 3.5 or greater + seaborn or greater 0.12: For data visualization.
  * joblib 1.2 or greater: For saving/loading trained models.
  * pytest 7.0 or greater: Code hardening etc.
  * ucimlrepo 0.0.3 or greater : API for UC Irvine machine learning repository
## Data Source
This project will initially use publicly available, tabular datasets suitable for risk prediction tasks, such as those found on the UCI Machine Learning Repository or Kaggle Datasets. The specific dataset will be chosen and documented shortly.

## Project Plan & Tasks
The main development tasks are outlined as GitHub Issues within this repository.
