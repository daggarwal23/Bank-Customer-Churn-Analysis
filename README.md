# Bank Customer Churn Analysis

## Overview
This project aims to predict bank customer churn using machine learning techniques. Churn prediction is based on various customer attributes such as geographic location, credit card usage, gender, and other relevant factors. By identifying customers likely to churn, banks can take proactive measures to retain them.

## Dataset
- **Filename**: `Churn_Modelling.csv`
- **Description**: The dataset contains 1,000 rows and 14 attributes. Each row represents a customer, and the attributes include demographic information, account details, and churn status.
- **Key Features**:
  - `Geography`: Customer's geographic location.
  - `Gender`: Gender of the customer.
  - `CreditCard`: Whether the customer holds a credit card.
  - **Additional Features**: Age, Balance, Number of Products, Estimated Salary, etc.
  
## Project Structure
- **Jupyter Notebook**: `bank-customer-churn-prediction.ipynb`
  - The notebook includes data exploration, preprocessing, model training, and evaluation.
  - Several machine learning models were explored, and the best-performing model was selected for prediction.

## Machine Learning Models
- Various machine learning algorithms were tested, including Logistic Regression, Random Forest, and Gradient Boosting.
- The model's performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Results
- The final model successfully predicts customer churn with a high degree of accuracy.
- Insights were derived from the model to understand the key factors contributing to customer churn.

## How to Run the Project
1. Clone the repository.
2. Ensure that the `Churn_Modelling.csv` file is in the same directory as the Jupyter notebook.
3. Open the `bank-customer-churn-prediction.ipynb` notebook.
4. Run the cells in the notebook to see the analysis and results.

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
