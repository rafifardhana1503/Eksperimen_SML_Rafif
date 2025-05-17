import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import mlflow

def preprocessing_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Konversi kolom object ke category
    object_columns = [
        "SeniorCitizen", "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "Churn"
    ]
    
    for col in object_columns:
        df[col] = df[col].astype('category')
    
    # Konversi TotalCharges ke numerikal (float)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Hapus baris TotalCharges yang missing
    df.dropna(subset=["TotalCharges"], inplace=True)
    
    # Reset index dataset
    df.reset_index(drop=True, inplace=True)

    # Mapping pilihan fitur layanan dan replace
    replace_map = {
        'MultipleLines': 'No phone service', 'OnlineSecurity': 'No internet service',
        'OnlineBackup': 'No internet service','DeviceProtection': 'No internet service',
        'TechSupport': 'No internet service', 'StreamingTV': 'No internet service',
        'StreamingMovies': 'No internet service'
    }

    for column, to_replace_value in replace_map.items():
        df[column] = df[column].replace(to_replace_value, 'No')

    # Drop kolom customerID
    df.drop("customerID", axis=1, inplace=True)

    # Inisialisasi enoder
    label_encoder = LabelEncoder()
    
    # Encoding kolom target Churn
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encoding untuk kolom binary kategori
    binary_columns = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup","DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling"
    ]

    for column in binary_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # One-Hot Encoding untuk non-binary kategori
    multi_columns = ["InternetService", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_columns, dtype=int)

    # Normalisasi fitur numerik
    scaler = StandardScaler()
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

if __name__ == "__main__":
    input_file = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), "dataset_raw/telco-customer-churn_raw.csv")
    output_dir = os.path.join(os.getenv("GITHUB_WORKSPACE", "./"), "preprocessing/output")

    print(f"Output directory: {output_dir}")
    print(f"Input file: {input_file}")

    mlruns_path = os.path.join(output_dir, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlruns_path}")

    mlflow.set_experiment("Preprocessing_Experiment")

    with mlflow.start_run(run_name="Preprocessing_Run"):
        df_clean = preprocessing_data(input_file)

        output_path = os.path.join(output_dir, "telco-customer-churn_preprocessing.csv")
        df_clean.to_csv(output_path, index=False)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_file", output_path)
        mlflow.log_metric("rows_clean", df_clean.shape[0])
        mlflow.log_artifact(output_path)