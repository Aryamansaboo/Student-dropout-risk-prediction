# Student Course Dropout Risk Prediction

## Overview
This project implements an end-to-end machine learning system to predict student course dropout risk in a university setting. The focus of the project is on building a reproducible and maintainable ML pipeline rather than optimizing model performance.

## Problem Statement
Student dropout is a critical issue in universities. Early identification of at-risk students can help academic administrators intervene and improve student retention.

## Data
The dataset is generated using a reproducible Python script. Static datasets are not stored in the repository to ensure reproducibility.

## Models
Traditional machine learning models are used:
- Logistic Regression
- Random Forest
- Decision Tree

Models are evaluated using Accuracy, Precision, Recall, and F1-score.

## Prediction
The trained model is used to predict whether a student is at risk of dropping out based on academic and engagement-related inputs.

## Dashboard
An interactive dashboard is provided to visualize inputs and generate predictions using the trained model.

## Model Lifecycle
New student data can be generated periodically and added to the database. The model can be retrained at regular intervals, and the latest model is used for predictions.

## Repository Structure
- data/: Data generation scripts
- database/: SQL schema
- scripts/: Training and prediction scripts
- dashboard/: Prediction dashboard
