import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import mlflow

def preprocessing_data(filepath,output_dir):
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

    # Split fitur dan target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simpan hasil split
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    return {
        "rows_clean": df.shape[0],
        "files": [
            os.path.join(output_dir, "X_train.csv"),
            os.path.join(output_dir, "X_test.csv"),
            os.path.join(output_dir, "y_train.csv"),
            os.path.join(output_dir, "y_test.csv"),
        ]
    }

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
        result = preprocessing_data(input_file, output_dir)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_metric("rows_clean", result["rows_clean"])

        for f in result:
            mlflow.log_artifact(f)