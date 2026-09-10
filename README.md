# Credit Card Fraud Detection

## 📌 Project Overview

This project uses unsupervised machine learning to identify potentially suspicious credit card transactions by detecting unusual transaction patterns.

## 🎯 Objective

The main objective is to detect anomalous transactions without depending on manually labelled fraud data.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Isolation Forest
- StandardScaler

## 🔍 Methodology

1. Load transaction data from a CSV file
2. Select relevant numerical features
3. Handle missing values
4. Standardize the data
5. Apply the Isolation Forest algorithm
6. Identify potentially anomalous transactions
7. Display the suspicious transactions

## 📂 Project Structure

```text
credit-card-fraud-detection/
├── README.md
├── fraud_detection.py
├── requirements.txt
└── sample_transactions.csv
