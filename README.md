#  Fraud Detection in Financial Transactions

##  Project Overview
This project detects fraudulent financial transactions using machine learning techniques. It helps identify suspicious activities in transaction data.

##  Objective
To build a system that can automatically detect fraud transactions and help prevent financial losses.

##  Dataset
- Credit Card Transaction Dataset
- Contains anonymized features (V1 to V28), Time, Amount, and Class
- Class = 0 (Normal), 1 (Fraud)

- The dataset is too large to upload on GitHub.

Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit (Dashboard)
- Matplotlib / Seaborn

##  Steps Involved
1. Data Loading
2. Data Exploration
3. Handling Imbalanced Data using SMOTE
4. Model Building using Isolation Forest
5. Prediction of Fraud Transactions
6. Model Evaluation
7. Dashboard Development using Streamlit

##  Model Used
- Isolation Forest (Anomaly Detection)

##  Results
- Successfully detected fraudulent transactions
- Improved fraud detection using SMOTE
- Visualized fraud alerts in dashboard

##  Output
(Add dashboard screenshot )

##  How to Run Project

```bash
pip install streamlit
streamlit run app.py
