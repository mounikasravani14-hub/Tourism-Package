import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# File paths
RAW_PATH = Path("/content/tourism_project/data/tourism.csv")
OUTPUT_DIR = Path("tourism_project/data/prepared")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
if not RAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Clean inconsistent categorical labels
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace(
        {"Unmarried": "Single"}
    )

# Remove extra spaces from text values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Validate target column
if "ProdTaken" not in df.columns:
    raise ValueError("Target column 'ProdTaken' is missing.")

df = df.dropna(subset=["ProdTaken"]).copy()

if not df["ProdTaken"].isin([0, 1]).all():
    raise ValueError("ProdTaken must contain only 0 and 1.")

df["ProdTaken"] = df["ProdTaken"].astype(int)

# Separate features and target
drop_cols = ["ProdTaken", "CustomerID", "Unnamed: 0"]
X = df.drop(columns=drop_cols, errors="ignore")
y = df["ProdTaken"]

# Split before learning imputation values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Fill missing values using training data only
numeric_cols = X_train.select_dtypes(include=["number"]).columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    median_value = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

for col in categorical_cols:
    mode_values = X_train[col].mode()
    fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
    X_train[col] = X_train[col].fillna(fill_value)
    X_test[col] = X_test[col].fillna(fill_value)

# Save train-test split files
output_dir = "tourism_project/data/prepared"
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

print("Training and testing datasets saved successfully.")

# Print summary
print("\nDATA PREPARATION SUMMARY")
print("-" * 35)
print("Cleaned dataset shape:", df.shape)
print("Number of input features:", X.shape[1])
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Missing values in training data:", X_train.isnull().sum().sum())
print("Missing values in testing data:", X_test.isnull().sum().sum())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTraining target proportions:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target proportions:")
print(y_test.value_counts(normalize=True).round(3))

print("\nSaved prepared data to:", OUTPUT_DIR)
print("Data preparation completed successfully.")
