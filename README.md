# ADAS-EV Action Classification

A machine learning project exploring the classification of Advanced Driver Assistance System (ADAS) actions using vehicle dynamics and environmental data.
**Project status:** In progress

## Project Overview

This project develops a supervised machine learning pipeline to classify recorded ADAS actions into four categories:

- Accelerate
- Brake
- Lane Correct
- Maintain Speed

The goal is to explore how Python and machine learning can support automotive data analysis, ADAS scenario evaluation, and vehicle testing workflows.

## Dataset

The dataset contains 10,000 records and 15 columns, including vehicle speed, acceleration, braking intensity, battery level, obstacle distance, lane deviation, road type, and weather conditions.

**Target variable:** `ADAS_output`

Dataset source: [ADAS-EV Dataset on Kaggle](https://www.kaggle.com/datasets/ziya07/adas-ev-dataset)

Please check the dataset's license and terms before redistributing the original data.

## Technologies

- Python
- Pandas
- Scikit-learn
- OneHotEncoder
- Random Forest Classification
- Exploratory Data Analysis (EDA)
- Model evaluation and feature importance

## Methodology

1. Load and inspect the dataset using Pandas.
2. Separate input features from the target variable.
3. Split the data chronologically into training and testing sets (70% / 30%).
4. Encode categorical variables using OneHotEncoder.
5. Establish a baseline using DummyClassifier.
6. Train a Random Forest classifier with 100 trees.
7. Evaluate predictions using accuracy, precision, recall, F1-score, and error analysis.
8. Investigate feature importance and potential target leakage.

## Initial Results

The initial Random Forest evaluation produced the following results on a 3,000-record test set:

| Metric | Result |

| Test records | 3,000 |
| Accuracy | 99.97% |
| Correct predictions | 2,999 |
| Incorrect predictions | 1 |

One record with the actual label `Maintain Speed` was predicted as `Lane Correct`.

These results are preliminary. The unusually high accuracy requires further investigation for potential target leakage, synthetic label patterns, and generalization limitations.

## Potential Automotive Applications

- Exploratory analysis of ADAS scenario data
- Review of vehicle dynamics and environmental features
- Identification of potentially misclassified scenarios
- Support for data-driven automotive testing and validation workflows

This project is a learning prototype. It is not a vehicle controller and has not been validated for real-world ADAS safety applications.


