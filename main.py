import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from time import time
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tkinter import filedialog
import seaborn as sns
from imblearn.over_sampling import SMOTE
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
    feature_columns = [
        'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax',
        'st_teff', 'pl_rade', 'pl_radj', 'pl_bmassj', 'pl_orbeccen', 
        'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist'
    ]

    feature_values = df[feature_columns].to_numpy()
    feature_values = np.nan_to_num(feature_values)
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

    # Calculate and clip habitability scores
    habitability_scores = np.dot(normalized_values, weights)
    habitability_scores = np.clip(habitability_scores, 0, 1) * 100

    return np.round(habitability_scores, 3)

# Load the datasets
ps_df = pd.read_csv(Dataset_Path + "/PS_2024.09.05_07.07.39.csv", low_memory=False)
psc_df = pd.read_csv(Dataset_Path + "/PSCompPars_2024.09.05_07.07.36.csv", low_memory=False)
hwc_df = pd.read_csv(Dataset_Path + "/hwc.csv", low_memory=False)
table_df = pd.read_csv(Dataset_Path + "/table.csv", low_memory=False)
kknW76pv_df = pd.read_csv(Dataset_Path + "/kknW76pv.csv", low_memory=False)

def hwc_format(df):  
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
    return df.rename(columns=rename_mapping)[list(rename_mapping.values())]

def table_format(df):  
    rename_mapping = {
        'MASS': 'pl_bmasse',
        'RADIUS': 'pl_rade',
        'PERIOD': 'pl_orbper',
        'ECC': 'pl_orbeccen'
    }
    return df.rename(columns=rename_mapping)[list(rename_mapping.values())]

def kkn_format(df):  
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
    return df.rename(columns=rename_mapping)[list(rename_mapping.values())]

# Format and merge datasets
hwc_formatted_df = hwc_format(hwc_df)
table_formatted_df = table_format(table_df)
kkn_formatted_df = kkn_format(kknW76pv_df)

balanced_df = pd.concat([ps_df, psc_df, hwc_formatted_df, table_formatted_df, kkn_formatted_df],
                       ignore_index=True)

# Clean and prepare data
balanced_df.replace('#', np.nan, inplace=True)

feature_columns = [
    'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax',
    'st_teff', 'pl_rade', 'pl_radj', 'pl_bmassj', 'pl_orbeccen', 
    'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist'
]

for column in feature_columns:
    balanced_df[column] = pd.to_numeric(balanced_df[column], errors='coerce')

balanced_df.fillna(0, inplace=True)

# Calculate habitability scores
balanced_df['Habitability'] = calculate_habitability(balanced_df, weights, earth_values)

# Keep only required columns and save
final_columns = feature_columns + ['Habitability']
balanced_df = balanced_df[final_columns]
balanced_df.to_csv(Dataset_Path + "/Exoplanets_Updated.csv", index=False)

def balance_habitability_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    bin_edges = np.linspace(0, 100, 11)
    df['hab_bin'] = pd.cut(df['Habitability'], bins=bin_edges, labels=False)
    min_samples = df['hab_bin'].value_counts().min()
    
    balanced_df = pd.DataFrame()
    for i in range(10):
        bin_df = df[df['hab_bin'] == i]
        if len(bin_df) > min_samples:
            balanced_df = pd.concat([balanced_df, bin_df.sample(n=min_samples, random_state=42)])
        else:
            balanced_df = pd.concat([balanced_df, bin_df])
    
    balanced_df = balanced_df.drop('hab_bin', axis=1)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df.to_csv(Dataset_Path + "/Balanced_Exoplanets.csv", index=False)
    return balanced_df

# Balance the dataset
balanced_df = balance_habitability_data(Dataset_Path + "/Exoplanets_Updated.csv")

# Print class distribution
positive_count = balanced_df[balanced_df['Habitability'] >= 50].shape[0]
negative_count = balanced_df[balanced_df['Habitability'] < 50].shape[0]
print(f"Positive Samples (Habitable): {positive_count}")
print(f"Negative Samples (Non-Habitable): {negative_count}")

# Prepare data for modeling
balanced_df = pd.get_dummies(balanced_df, drop_first=True)
X = balanced_df.drop(columns=['Habitability'])
y = balanced_df['Habitability']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the improved RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,            # Increased from 10 to 15 for better fitting
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R² Score: {r2:.4f}')

# Create improved visualizations
plt.figure(figsize=(15, 5))

# Training Data Plot
plt.subplot(1, 3, 1)
plt.hexbin(y_train, y_train_pred, gridsize=30, cmap='YlOrRd')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')
plt.legend()
plt.colorbar(label='Count')

# Test Data Plot
plt.subplot(1, 3, 2)
plt.hexbin(y_test, y_test_pred, gridsize=30, cmap='YlOrRd')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')
plt.legend()
plt.colorbar(label='Count')

# Distribution Plot
plt.subplot(1, 3, 3)
sns.kdeplot(data=y_train, label='Actual', alpha=0.6)
sns.kdeplot(data=y_train_pred, label='Predicted', alpha=0.6)
plt.title('Distribution of Habitability Scores')
plt.xlabel('Habitability Score')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

end_time = time()
print("Runtime is : ", round((end_time - start_time), 3), "seconds")
