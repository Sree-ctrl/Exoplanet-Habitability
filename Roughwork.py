from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
import time
import numpy as np

start_time = time.time()

weights = {
    'pl_bmasse': 0.15,
    'pl_insol': 0.15,
    'pl_eqt': 0.15,
    'pl_orbper': 0.1,
    'pl_orbsmax': 0.1,
    'st_teff': 0.1,
    'pl_rade': 0.05,
    'pl_radj': 0.05,
    'pl_bmassj': 0.05,
    'pl_orbeccen': 0.05,
    'st_rad': 0.05,
    'st_mass': 0.05,
    'st_met': 0.05,
    'st_logg': 0.05,
    'sy_dist': 0.05,
    # Zero weight variables omitted for brevity
}

def calculate_habitability(row, weights):
    habitability = 0
    for feature, weight in weights.items():
        if feature in row and pd.notnull(row[feature]):  # Check if feature exists and is not NaN
            normalized_value = (row[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
            habitability += weight * normalized_value

    # Ensure the habitability score is between 0 and 1
    habitability = round(min(1, max(0, habitability)) * 100, 3)
    return habitability

# Load the dataset
df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets Trial.csv")

# Calculate habitability for each row
df['Habitability'] = df.apply(lambda row: calculate_habitability(row, weights), axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets_Updated.csv", index=False)

end_time = time.time()
print("Total Time : ", round(end_time - start_time, 2), " seconds")
