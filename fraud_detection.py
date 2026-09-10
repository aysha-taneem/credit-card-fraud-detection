# Credit Card Fraud Detection
# Unsupervised Machine Learning using Isolation Forest

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_fraud(file_path):
    # Load transaction data
    data = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Dataset shape:", data.shape)

    # Select numerical columns
    numerical_data = data.select_dtypes(include=["number"]).copy()

    if numerical_data.empty:
        print("No numerical columns found in the dataset.")
        return

    # Handle missing values
    numerical_data = numerical_data.fillna(numerical_data.median())

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)

    # Create Isolation Forest model
    model = IsolationForest(
        contamination=0.02,
        random_state=42
    )

    # Predict anomalies
    predictions = model.fit_predict(scaled_data)

    # -1 = anomaly, 1 = normal
    data["fraud_prediction"] = predictions

    # Display suspicious transactions
    suspicious = data[data["fraud_prediction"] == -1]

    print("\nPotentially suspicious transactions:")
    print(suspicious.head())

    print("\nTotal transactions:", len(data))
    print("Potential anomalies detected:", len(suspicious))

    return data


if __name__ == "__main__":
    print("Credit Card Fraud Detection System")
    print("----------------------------------")
    print("Upload a CSV dataset and call detect_fraud('your_file.csv')")
