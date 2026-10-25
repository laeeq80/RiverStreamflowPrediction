from pathlib import Path
import pandas as pd
#import seaborn as sns
#import numpy as np
#import matplotlib.pyplot as plt

# Get the project root (the parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the absolute path to the Excel file
DATA_PATH = PROJECT_ROOT / "data" / "flood_data.xlsx"

df = pd.read_excel(DATA_PATH)

print(f"Loaded data from: {DATA_PATH}")
#print(df.head())
#df.info()

# To detect the problem in data, we only need the numeric value in that column
mask = pd.to_numeric(df["Water_Discharge"], errors='coerce').isna() & df["Water_Discharge"].notna()
problematic = df.loc[mask, "Water_Discharge"]
print(problematic.unique())# see unique problem values

# Here we check the unneccessory values in the column of Minimum Temperature
mask = pd.to_numeric(df["Minimum_Temperature"], errors='coerce').isna() & df["Minimum_Temperature"].notna()
problematic = df.loc[mask, "Minimum_Temperature"]
# Show row numbers with values
print("The problematic values in the feature Minimum_Temperature are \n", problematic.to_string(),"\n")

# here also check the unneccessory values in the column of Maximum Temperature
mask = pd.to_numeric(df["Maximum_Temperature"], errors='coerce').isna() & df["Maximum_Temperature"].notna()
problematic = df.loc[mask, "Maximum_Temperature"]
print("The problematic values in the feature Maximum_Temperature are \n", problematic.to_string(),"\n")# see first 20 unique problem values

# here also check the unneccessory values in the column of Rainfall
mask = pd.to_numeric(df["Rainfall"], errors='coerce').isna() & df["Rainfall"].notna()
problematic = df.loc[mask, "Rainfall"]
print("The Problematic value in the feature Rainfall are \n",problematic.to_string(),"\n") # see first 20 unique problem values

# Replace non-numeric Rainfall values (found in previous cell) with 0.05
# Recreate the same mask
mask = pd.to_numeric(df["Rainfall"], errors='coerce').isna() & df["Rainfall"].notna()

# Replace problematic values with 0.05
df.loc[mask, "Rainfall"] = 0.05