# Bank Customer Churn Analysis

## Overview
This project aims to predict bank customer churn using machine learning techniques. The goal is to identify and visualize factors contributing to customer churn, and to build a predictive model that classifies whether a customer is likely to churn. Additionally, the model will provide a probability of churn to help customer service teams target customers most at risk.

## Dataset
- **Filename**: `Churn_Modelling.csv`
- **Description**: The dataset contains 1,000 rows and 14 attributes. Each row represents a customer, and the attributes include various demographic and account-related features.
- **Example Features**:
  - `Geographic Loaction`
  - `Gender`
  - `CreditCard User`
  - `Balance`
## Project Objective
- **Identify and visualize** which factors contribute to customer churn.
- **Build a prediction model** that will:
  - Classify if a customer is going to churn or not.
  - Attach a probability to the churn, helping customer service target efforts effectively.

## Project Structure
- **Jupyter Notebook**: `bank-customer-churn-prediction.ipynb`
  - The notebook includes data exploration, preprocessing, model training, and evaluation.
  - Multiple machine learning models were explored, with the best-performing model selected for final predictions.

## Machine Learning Models
- The project explores different machine learning algorithms, including those that provide classification and probability outputs.
- Model performance was assessed using metrics such as accuracy, precision, recall, and AUC-ROC.

## Results
- The final model effectively predicts customer churn, providing actionable insights into the factors driving churn.
- The probability-based predictions allow targeted interventions by customer service teams.

## How to Run the Project
1. Clone the repository.
2. Place the `Churn_Modelling.csv` file in the same directory as the Jupyter notebook.
3. Open the `bank-customer-churn-prediction.ipynb` notebook.
4. Execute the cells to perform the analysis and see the results.

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
