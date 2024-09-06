import pandas as pd
import numpy as np

# Define Earth's values for each feature as a NumPy array for faster processing
earth_values = np.array([
    5.97e24,  # pl_bmasse
    1.0,      # pl_insol
    255.0,    # pl_eqt
    365.25,   # pl_orbper
    1.0,      # pl_orbsmax
    5778.0,   # st_teff
    6371.0,   # pl_rade
    0.08921,  # pl_radj
    0.00315,  # pl_bmassj
    0.0167,   # pl_orbeccen
    695700.0, # st_rad
    1.989e30, # st_mass
    0.0,      # st_met
    4.438,    # st_logg
    0.0       # sy_dist
])

# Define initial weights as a NumPy array
weights = np.array([
    0.15,
    0.15,
    0.15,
    0.1,
    0.1,
    0.1,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05
])

def calculate_habitability(df, weights, earth_values):
    # Extract relevant features into a NumPy array
    feature_columns = [
        'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper',
        'pl_orbsmax', 'st_teff', 'pl_rade', 'pl_radj',
        'pl_bmassj', 'pl_orbeccen', 'st_rad', 'st_mass',
        'st_met', 'st_logg', 'sy_dist'
    ]
    
    # Convert DataFrame to NumPy array for faster processing
    feature_values = df[feature_columns].to_numpy()

    # Replace NaN values with 0
    feature_values = np.nan_to_num(feature_values)

    # Calculate normalized values using vectorized operations
    normalized_values = np.zeros_like(feature_values, dtype=float)
    
    # Non-linear transformations
    normalized_values[:, 6] = (feature_values[:, 6] ** 2) / (earth_values[6] ** 2)  # Squared radius
    normalized_values[:, 1] = np.log(feature_values[:, 1] + 1) / np.log(earth_values[1] + 1)  # Log transformation for insolation
    
    # For other features, perform normalization
    for i in range(feature_values.shape[1]):
        if i != 6 and i != 1:  # Skip the transformed features
            normalized_values[:, i] = np.where(earth_values[i] != 0, feature_values[:, i] / earth_values[i], 0)

    # Calculate the habitability score using dot product
    habitability_scores = np.dot(normalized_values, weights)

    # Ensure the habitability score is between 0 and 1
    habitability_scores = np.clip(habitability_scores, 0, 1) * 100
    return np.round(habitability_scores, 3)

# Load the datasets
ps_df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\PS_2024.09.05_07.07.39.csv", low_memory=False)
psc_df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\PSCompPars_2024.09.05_07.07.36.csv", low_memory=False)
kknW76pv_df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\kknW76pv.csv", low_memory=False)
table_df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\table.csv", low_memory=False)
hwc_df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\hwc.csv", low_memory=False)
phl_df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\phl_exoplanet_catalog_2019.csv", low_memory=False)

# Merge the datasets
df = pd.concat([ps_df, psc_df, kknW76pv_df, table_df, hwc_df, phl_df], ignore_index=True)

# Calculate habitability for each row using vectorized processing
df['Habitability'] = calculate_habitability(df, weights, earth_values)

# Save the updated DataFrame to a new CSV file
df.to_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets_Updated.csv", index=False)

def balance_habitability_data(file_path):
    # Load the updated dataset
    df = pd.read_csv(file_path, low_memory=False)

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

    return balanced_df

# Call the function to balance the dataset
balanced_df = balance_habitability_data(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets_Updated.csv")

# Check the positive and negative samples after balancing
positive_count = balanced_df[balanced_df['Habitability'] >= 50].shape[0]
negative_count = balanced_df[balanced_df['Habitability'] < 50].shape[0]

print(f"Positive Samples (Habitable): {positive_count}")
print(f"Negative Samples (Non-Habitable): {negative_count}")
