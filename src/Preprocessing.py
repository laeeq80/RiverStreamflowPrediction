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
print("The problematic values in the feature Rainfall are \n",problematic.to_string(),"\n") # see first 20 unique problem values

# Replace non-numeric Rainfall values (found in previous cell) with 0.05
# Recreate the same mask
mask = pd.to_numeric(df["Rainfall"], errors='coerce').isna() & df["Rainfall"].notna()

# Replace problematic values with 0.05
df.loc[mask, "Rainfall"] = 0.05

# Convert the column to numeric type
df["Rainfall"] = pd.to_numeric(df["Rainfall"])

def clean_temperature_column(series):
    # Convert to string
    s = series.astype(str)

    # Replace non-breaking spaces (\xa0), normal spaces, and other invisible chars
    s = s.str.replace(u'\xa0', '', regex=False)
    s = s.str.replace(' ', '', regex=False)
    s = s.str.replace('\t', '', regex=False)

    # Remove any non-numeric characters except digits, dot, or minus
    s = s.str.replace(r'[^0-9\.\-]', '', regex=True)

    # Convert to float safely
    return pd.to_numeric(s, errors='coerce')

# Apply to your numeric columns
cols = ["Maximum_Temperature","Minimum_Temperature","Water_Discharge"]
for c in cols:
    df[c] = clean_temperature_column(df[c])

df=df.dropna(subset=['Minimum_Temperature']) #Removing rows with NaN in them, i.e. we converted text to NaN. We only want numeric values

df['Rainfall']=df['Rainfall'].fillna(0.05)# we fill the Null values with 0.05 replacing the string Trace which means a very low value

# Save the preprocessed dataset as a CSV file
OUTPUT_PATH = PROJECT_ROOT / "data" / "flood_clean_data.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"Preprocessed data saved to: {OUTPUT_PATH}")
