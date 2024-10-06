import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from time import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import seaborn as sns

start_time = time()

# Ask for Dataset Path
Dataset_Path = filedialog.askdirectory()

# Define Earth's values for each feature as a NumPy array
earth_values = np.array([
    5.97e24, 1.0, 255.0, 365.25, 1.0, 5778.0, 6371.0, 0.08921, 0.00315, 0.0167,
    695700.0, 1.989e30, 0.0, 4.438, 0.0
])

# Define weights
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
    normalized_values[:, 6] = (feature_values[:, 6] * 2) / (earth_values[6] * 2)
    normalized_values[:, 1] = np.log(feature_values[:, 1] + 1) / np.log(earth_values[1] + 1)

    for i in range(feature_values.shape[1]):
        if i != 6 and i != 1:
            normalized_values[:, i] = np.where(earth_values[i] != 0, feature_values[:, i] / (earth_values[i] + 1e-10), 0)

    habitability_scores = np.dot(normalized_values, weights)
    habitability_scores = np.clip(habitability_scores, 0, 1) * 100

    return np.round(habitability_scores, 3)

# Load datasets
ps_df = pd.read_csv(Dataset_Path + "/PS_2024.09.05_07.07.39.csv", low_memory=False)
psc_df = pd.read_csv(Dataset_Path + "/PSCompPars_2024.09.05_07.07.36.csv", low_memory=False)
hwc_df = pd.read_csv(Dataset_Path + "/hwc.csv", low_memory=False)
table_df = pd.read_csv(Dataset_Path + "/table.csv", low_memory=False)
kknW76pv_df = pd.read_csv(Dataset_Path + "/kknW76pv.csv", low_memory=False)

def hwc_format(df):
    rename_mapping = {
        'P_MASS': 'pl_bmasse', 'P_TEMP_EQUIL': 'pl_eqt', 'P_PERIOD': 'pl_orbper',
        'P_SEMI_MAJOR_AXIS': 'pl_orbsmax', 'S_TEMPERATURE': 'st_teff', 'P_RADIUS': 'pl_rade',
        'P_ECCENTRICITY': 'pl_orbeccen', 'S_RADIUS': 'st_rad', 'S_MASS': 'st_mass',
        'S_METALLICITY': 'st_met', 'S_LOG_G': 'st_logg', 'S_DISTANCE': 'sy_dist'
    }
    df_renamed = df.rename(columns=rename_mapping)
    columns_to_keep = list(rename_mapping.values())
    df_filtered = df_renamed[columns_to_keep]
    return df_filtered

# Format datasets
hwc_formatted_df = hwc_format(hwc_df)

# Merge datasets
balanced_df = pd.concat([ps_df, psc_df, hwc_formatted_df], ignore_index=True)
balanced_df.replace('#', np.nan, inplace=True)

# Ensure columns are numeric
feature_columns = [
    'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax', 'st_teff', 'pl_rade',
    'pl_radj', 'pl_bmassj', 'pl_orbeccen', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist'
]

for column in feature_columns:
    balanced_df[column] = pd.to_numeric(balanced_df[column], errors='coerce')

balanced_df.fillna(0, inplace=True)

balanced_df['Habitability'] = calculate_habitability(balanced_df, weights, earth_values)

final_columns = feature_columns + ['Habitability']
balanced_df = balanced_df[final_columns]

# Drop 15000 records with Habitability equal to 100
balanced_df = balanced_df.drop(balanced_df[balanced_df['Habitability'] == 100].sample(n=15000).index)

# Balance the data (same as before)
balanced_df = pd.get_dummies(balanced_df, drop_first=True)
X = balanced_df.drop(columns=['Habitability'])
y = balanced_df['Habitability']

# Downsample records with habitability of 100
habitability_100 = balanced_df[balanced_df['Habitability'] == 100]
habitability_non_100 = balanced_df[balanced_df['Habitability'] != 100]

# Randomly sample a smaller portion of the 'Habitability == 100' records
habitability_100_downsampled = habitability_100.sample(frac=0.1, random_state=42)  # Keep only 10%

# Combine the downsampled records with the non-100 records
balanced_df = pd.concat([habitability_100_downsampled, habitability_non_100])

# Specify the file path and save the DataFrame to a CSV file
balanced_df.to_csv('Balanced_Exoplanets.csv', index=False)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer 1
model.add(Dense(16, activation='relu'))  # Hidden layer 2
model.add(Dense(1, activation='linear')) # Output layer 

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1,callbacks = [early_stopping])

# Predict and evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Regression metrics
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# Plotting
# Create a figure with a specific size
plt.figure(figsize=(15, 5))

# Plot for Training Data
plt.subplot(1, 4, 1)  # Change this to 1 row and 3 columns
plt.scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')

# Plot for Test Data
plt.subplot(1, 4, 2)  # Adjust to the second subplot
plt.scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')

# Plot for Model Loss
plt.subplot(1, 4, 3)  # Adjust to the third subplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,4,4)
sns.histplot(balanced_df['Habitability'], bins=20, kde=True)
plt.title('Habitability Score Distribution')
plt.xlabel('Habitability')
plt.ylabel('Frequency')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# Save the model
model.save('NN_model.h5')
joblib.dump(scaler,'scaler.pk1')

end_time = time()
print("Runtime is:", round((end_time - start_time), 3), "secs")
