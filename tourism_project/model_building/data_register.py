import pandas as pd
from pathlib import Path

RAW_PATH = Path("tourism_project/data/tourism.csv")

# Check whether the dataset exists
if not RAW_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# validate that the expected columns are present before registering it
EXPECTED_COLUMNS = [
    "Unnamed: 0", "CustomerID", "ProdTaken", "Age",
    "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

# Validate expected columns
missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]

if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Print dataset registration summary
print("DATASET REGISTRATION SUMMARY")
print("-" * 35)
print("Dataset shape:", df.shape)
print("Expected columns check: PASSED")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nTarget distribution:")
print(df["ProdTaken"].value_counts(dropna=False))

print("\nDataset registration and validation completed successfully.")
