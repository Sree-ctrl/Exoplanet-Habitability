from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
import time
import numpy as np

start_time = time.time()

# Define Earth's values for each feature
earth_values = {
    'pl_bmasse': 5.97e24,  # Earth's mass in kg
    'pl_insol': 1.0,  # Earth's insolation in Earth units
    'pl_eqt': 255.0,  # Earth's equilibrium temperature in K
    'pl_orbper': 365.25,  # Earth's orbital period in days
    'pl_orbsmax': 1.0,  # Earth's semi-major axis in AU
    'st_teff': 5778.0,  # Sun's effective temperature in K
    'pl_rade': 6371.0,  # Earth's radius in km
    'pl_radj': 0.08921,  # Earth's radius in Jupiter radii
    'pl_bmassj': 0.00315,  # Earth's mass in Jupiter masses
    'pl_orbeccen': 0.0167,  # Earth's orbital eccentricity
    'st_rad': 695700.0,  # Sun's radius in km
    'st_mass': 1.989e30,  # Sun's mass in kg
    'st_met': 0.0,  # Sun's metallicity
    'st_logg': 4.438,  # Sun's surface gravity in cm/s^2
    'sy_dist': 0.0,  # Earth's distance from Sun in pc (not applicable)
}

# Define initial weights (these can be adjusted)
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
}

def calculate_normalized_value(feature, row, earth_values):
    normalization_functions = {
        'pl_bmasse': lambda: row[feature] / earth_values['pl_bmasse'],
        'pl_insol': lambda: row[feature] / earth_values['pl_insol'],
        'pl_eqt': lambda: 1 - abs(row[feature] - earth_values['pl_eqt']) / (2 * earth_values['pl_eqt']),
        'pl_orbper': lambda: 1 - abs(row[feature] - earth_values['pl_orbper']) / (2 * earth_values['pl_orbper']),
        'pl_orbsmax': lambda: 1 - abs(row[feature] - earth_values['pl_orbsmax']) / (2 * earth_values['pl_orbsmax']),
        'st_teff': lambda: 1 - abs(row[feature] - earth_values['st_teff']) / (2 * earth_values['st_teff']),
        'pl_rade': lambda: row[feature] / earth_values['pl_rade'],
        'pl_radj': lambda: row[feature] / earth_values['pl_radj'],
        'pl_bmassj': lambda: row[feature] / earth_values['pl_bmassj'],
        'pl_orbeccen': lambda: 1 - row[feature] / earth_values['pl_orbeccen'],
        'st_rad': lambda: row[feature] / earth_values['st_rad'],
        'st_mass': lambda: row[feature] / earth_values['st_mass'],
        'st_met': lambda: 1 if row[feature] == 0 else 0 if earth_values['st_met'] == 0 else 1 - abs(row[feature] - earth_values['st_met']) / (2 * earth_values['st_met']),
        'st_logg': lambda: 1 - abs(row[feature] - earth_values['st_logg']) / (2 * earth_values['st_logg']),
        'sy_dist': lambda: 1 - row[feature] / (1e-6 if earth_values['sy_dist'] == 0 else earth_values['sy_dist']),
    }

    return normalization_functions[feature]() if feature in normalization_functions else 0

def calculate_habitability(row, weights, earth_values):
    habitability = 0
    for feature, weight in weights.items():
        if feature in row and pd.notnull(row[feature]):  # Check if feature exists and is not NaN
            normalized_value = calculate_normalized_value(feature, row, earth_values)
            habitability += weight * normalized_value

    # Ensure the habitability score is between 0 and 1
    habitability = round(min(1, max(0, habitability)) * 100, 3)
    return habitability

# Load the dataset
df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\PS_2024.09.04_22.27.50.csv")

# Calculate habitability for each row
df['Habitability'] = df.apply(lambda row: calculate_habitability(row, weights, earth_values), axis=1)

# Use Earth's habitability score directly from the predefined values
earth_habitability = calculate_habitability(earth_values, weights, earth_values)

# Adjust weights to ensure Earth's habitability score is 100%
if earth_habitability != 100:
    for feature in weights.keys():
        weights[feature] *= 100 / earth_habitability

    # Recalculate habitability for each row with adjusted weights
    df['Habitability'] = df.apply(lambda row: calculate_habitability(row, weights, earth_values), axis=1)

# Save the updated DataFrame to a new CSV file
df.to_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets_Updated.csv", index=False)

end_time = time.time()
print("Total Time : ", round(end_time - start_time, 2), " seconds")

def balance_habitability_data(file_path):
    # Load the updated dataset
    df = pd.read_csv(file_path)

    # Count habitable and non-habitable samples
    habitable_count = df[df['Habitability'] >= 50].shape[0]  # Assuming 50% as the threshold for habitability
    non_habitable_count = df[df['Habitability'] < 50].shape[0]

    print(f"Habitability Count: {habitable_count}")
    print(f"Non-Habitability Count: {non_habitable_count}")

    # Balance the dataset
    if habitable_count > non_habitable_count:
        # Randomly sample habitable rows to match non-habitable count
        df_habitable = df[df['Habitability'] >= 50]
        df_non_habitable = df[df['Habitability'] < 50]
        df_habitable_balanced = df_habitable.sample(n=non_habitable_count, random_state=42)
        balanced_df = pd.concat([df_habitable_balanced, df_non_habitable])
    else:
        # Randomly sample non-habitable rows to match habitable count
        df_habitable = df[df['Habitability'] >= 50]
        df_non_habitable = df[df['Habitability'] < 50]
        df_non_habitable_balanced = df_non_habitable.sample(n=habitable_count, random_state=42)
        balanced_df = pd.concat([df_habitable, df_non_habitable_balanced])

    # Shuffle the balanced DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the balanced DataFrame to a new CSV file
    balanced_df.to_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Balanced_Exoplanets.csv", index=False)
    print("Balanced dataset saved to 'Balanced_Exoplanets.csv'.")

# Call the function with the path to the updated CSV file
balance_habitability_data(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets_Updated.csv")