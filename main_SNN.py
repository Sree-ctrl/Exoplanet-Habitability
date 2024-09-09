import pandas as pd
import numpy as np
from keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from time import time
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


start_time = time()

Dataset_Path = filedialog.askdirectory()

# Define Earth's values for each feature as a NumPy array for faster processing
earth_values = np.array([
    5.97e24,  # pl_bmasse
    1.0,  # pl_insol
    255.0,  # pl_eqt
    365.25,  # pl_orbper
    1.0,  # pl_orbsmax
    5778.0,  # st_teff
    6371.0,  # pl_rade
    0.08921,  # pl_radj
    0.00315,  # pl_bmassj
    0.0167,  # pl_orbeccen
    695700.0,  # st_rad
    1.989e30,  # st_mass
    0.0,  # st_met
    4.438,  # st_logg
    0.0  # sy_dist
])

# Define initial weights as a NumPy array
weights = np.array([
    0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    0.05, 0.05
])

def calculate_habitability(df, weights, earth_values):
    # Extract relevant features into a NumPy array
    feature_columns = [
        'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax',
        'st_teff', 'pl_rade', 'pl_radj', 'pl_bmassj', 'pl_orbeccen', 
        'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist'
    ]

    # Convert DataFrame to NumPy array for faster processing
    feature_values = df[feature_columns].to_numpy()

    # Replace NaN values with 0
    feature_values = np.nan_to_num(feature_values)

    # Calculate normalized values using vectorized operations
    normalized_values = np.zeros_like(feature_values, dtype=float)

    # Non-linear transformations
    normalized_values[:, 6] = (feature_values[:, 6]*2) / (earth_values[6]*2)  # Squared radius
    normalized_values[:, 1] = np.log(feature_values[:, 1] + 1) / np.log(earth_values[1] + 1)  # Log transformation for insolation

    # For other features, perform normalization
    for i in range(feature_values.shape[1]):
        if i != 6 and i != 1:  # Skip the transformed features
            normalized_values[:, i] = np.where(
                earth_values[i] != 0, feature_values[:, i] / (earth_values[i] + 1e-10), 0
            )

    # Calculate the habitability score using dot product
    habitability_scores = np.dot(normalized_values, weights)

    # Ensure the habitability score is between 0 and 1
    habitability_scores = np.clip(habitability_scores, 0, 1) * 100

    return np.round(habitability_scores, 3)

# Load the datasets
ps_df = pd.read_csv(Dataset_Path + "/PS_2024.09.05_07.07.39.csv", low_memory=False)
psc_df = pd.read_csv(Dataset_Path + "/PSCompPars_2024.09.05_07.07.36.csv", low_memory=False)
hwc_df = pd.read_csv(Dataset_Path + "/hwc.csv", low_memory=False)
table_df = pd.read_csv(Dataset_Path + "/table.csv", low_memory=False)
kknW76pv_df = pd.read_csv(Dataset_Path + "/kknW76pv.csv", low_memory=False)

def hwc_format(df):  
    # Define the mapping of old column names to new column names
    rename_mapping = {
        'P_MASS': 'pl_bmasse',
        'P_TEMP_EQUIL': 'pl_eqt',
        'P_PERIOD': 'pl_orbper',
        'P_SEMI_MAJOR_AXIS': 'pl_orbsmax',
        'S_TEMPERATURE': 'st_teff',
        'P_RADIUS': 'pl_rade',
        'P_ECCENTRICITY': 'pl_orbeccen',
        'S_RADIUS': 'st_rad',
        'S_MASS': 'st_mass',
        'S_METALLICITY': 'st_met',
        'S_LOG_G': 'st_logg',
        'S_DISTANCE': 'sy_dist'
    }

    # Rename columns using the mapping
    df_renamed = df.rename(columns=rename_mapping)

    # Define the columns to keep based on the new names
    columns_to_keep = list(rename_mapping.values())

    # Filter the DataFrame to keep only the specified columns
    df_filtered = df_renamed[columns_to_keep]

    return df_filtered

def table_format(df):  
    # Define the mapping of old column names to new column names
    rename_mapping = {
        'MASS': 'pl_bmasse',
        'RADIUS': 'pl_rade',
        'PERIOD': 'pl_orbper',
        'ECC': 'pl_orbeccen'
    }

    # Rename columns using the mapping
    df_renamed = df.rename(columns=rename_mapping)

    # Define the columns to keep based on the new names
    columns_to_keep = list(rename_mapping.values())

    # Filter the DataFrame to keep only the specified columns
    df_filtered = df_renamed[columns_to_keep]

    return df_filtered

def kkn_format(df):  
    # Define the mapping of old column names to new column names
    rename_mapping = {
        'mass': 'pl_bmasse',
        'radius': 'pl_rade',
        'orbital_period': 'pl_orbper',
        'eccentricity': 'pl_orbeccen',
        'omega': 'pl_orbsmax',
        'star_teff': 'st_teff',
        'star_mass': 'st_mass',
        'star_radius': 'st_rad'
    }

    # Rename columns using the mapping
    df_renamed = df.rename(columns=rename_mapping)

    # Define the columns to keep based on the new names
    columns_to_keep = list(rename_mapping.values())

    # Filter the DataFrame to keep only the specified columns
    df_filtered = df_renamed[columns_to_keep]

    return df_filtered

# Format the datasets
hwc_formatted_df = hwc_format(hwc_df)
table_formatted_df = table_format(table_df)
kkn_formatted_df = kkn_format(kknW76pv_df)

# Merge the datasets
balanced_df = pd.concat([ps_df, psc_df, hwc_formatted_df, table_formatted_df, kkn_formatted_df],
               ignore_index=True)

# Replace hash values with NaN
balanced_df.replace('#', np.nan, inplace=True)

# Ensure relevant columns are numeric
feature_columns = [
    'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax',
    'st_teff', 'pl_rade', 'pl_radj', 'pl_bmassj', 'pl_orbeccen', 
    'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist'
]

# Convert relevant columns to numeric
for column in feature_columns:
    balanced_df[column] = pd.to_numeric(balanced_df[column], errors='coerce')

# Optionally handle NaN values
balanced_df.fillna(0, inplace=True)

# Calculate habitability for each row using vectorized processing
balanced_df['Habitability'] = calculate_habitability(balanced_df, weights, earth_values)

# Keep only the required columns for the final output
final_columns = feature_columns + ['Habitability']
balanced_df = balanced_df[final_columns]

# Save the updated DataFrame to a new CSV file
balanced_df.to_csv(Dataset_Path + "/Exoplanets_Updated.csv", index=False)

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
    balanced_df.to_csv(Dataset_Path + "/Balanced_Exoplanets.csv", index=False)
    print("Balanced dataset saved to 'Balanced_Exoplanets.csv'.")

    return balanced_df

# Call the function to balance the dataset
balanced_df = balance_habitability_data(Dataset_Path + "/Exoplanets_Updated.csv")

# Check the positive and negative samples after balancing
positive_count = balanced_df[balanced_df['Habitability'] >= 50].shape[0]
negative_count = balanced_df[balanced_df['Habitability'] < 50].shape[0]

print(f"Positive Samples (Habitable): {positive_count}")
print(f"Negative Samples (Non-Habitable): {negative_count}")

balanced_df = pd.get_dummies(balanced_df, drop_first=True)

# Setting X and Y data
X = balanced_df.drop(columns=['Habitability'])
y = balanced_df['Habitability']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')
# Predict values for training and test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Plotting
plt.figure(figsize=(14, 6))

# Plot for Training Data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')

# Plot for Test Data
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')

plt.tight_layout()
plt.show()

end_time = time()

print("Runtime is : ",round((end_time-start_time),3),"ms")