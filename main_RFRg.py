import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.regularizers import l1_l2
import matplotlib as plt

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
    0.0   # sy_dist
])

# Define initial weights as a NumPy array
weights = np.array([
    0.15, 0.15, 0.15, 0.1, 0.1,
    0.1, 0.05, 0.05, 0.05,
    0.05, 0.05, 0.05,
    0.05, 0.05,
    0.05
])

def calculate_habitability(df):
    feature_columns = [
        'pl_bmasse', 'pl_insol', 'pl_eqt', 'pl_orbper', 'pl_orbsmax',
        'st_teff', 'pl_rade', 'pl_radj', 'pl_bmassj', 'pl_orbeccen', 
        'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist'
    ]

    feature_values = df[feature_columns].to_numpy()
    feature_values = np.nan_to_num(feature_values)
    
    normalized_values = np.zeros_like(feature_values)

    normalized_values[:, 6] = (feature_values[:, 6]*2) / (earth_values[6]*2)  
    normalized_values[:, 1] = np.log(feature_values[:, 1] + 1) / np.log(earth_values[1] + 1)  

    for i in range(feature_values.shape[1]):
        if i != 6 and i != 1:  
            normalized_values[:, i] = np.where(
                earth_values[i] != 0,
                feature_values[:, i] / (earth_values[i] + 1e-10),
                0
            )

    habitability_scores = np.dot(normalized_values, weights)
    
    # Binning the habitability scores into ranges (e.g., [0-20], [20-40], etc.)
    bins = np.linspace(0,100,num=6)
    binned_scores = np.digitize(habitability_scores, bins) - 1

    # Adding small random noise to prevent exact predictions when converting back to continuous values.
    noise = np.random.uniform(-5,5,len(habitability_scores))
    
    smoothed_scores = bins[binned_scores] + noise
    
    return np.round(np.clip(smoothed_scores, a_min=0, a_max=100),3)

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

    df_renamed = df.rename(columns=rename_mapping)
    
    columns_to_keep = list(rename_mapping.values())
    
    df_filtered = df_renamed[columns_to_keep]
    
    return df_filtered

def table_format(df):  
    rename_mapping = {
        'MASS': 'pl_bmasse',
        'RADIUS': 'pl_rade',
        'PERIOD': 'pl_orbper',
        'ECC': 'pl_orbeccen'
    }

    df_renamed = df.rename(columns=rename_mapping)
    
    columns_to_keep = list(rename_mapping.values())
    
    df_filtered = df_renamed[columns_to_keep]
    
    return df_filtered

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

    df_renamed = df.rename(columns=rename_mapping)
    
    columns_to_keep = list(rename_mapping.values())
    
    df_filtered = df_renamed[columns_to_keep]
    
    return df_filtered

# Format the datasets 
hwc_formatted_df = hwc_format(hwc_df)
table_formatted_df = table_format(table_df)
kkn_formatted_df = kkn_format(kknW76pv_df)

# Merge the datasets 
balanced_df = pd.concat([ps_df, psc_df, hwc_formatted_df,
                          table_formatted_df,
                          kkn_formatted_df],
                         ignore_index=True)

balanced_df.replace('#', np.nan, inplace=True)

# Ensure relevant columns are numeric
feature_columns = [
   "pl_bmasse", "pl_insol", "pl_eqt", "pl_orbper", "pl_orbsmax",
   "st_teff", "pl_rade", "pl_radj", "pl_bmassj", "pl_orbeccen",
   "st_rad", "st_mass", "st_met", "st_logg", "sy_dist"
]

for column in feature_columns:
   balanced_df[column] = pd.to_numeric(balanced_df[column], errors='coerce')

balanced_df.fillna(0, inplace=True)

# Calculate habitability for each row using vectorized processing with smoothing and binning.
balanced_df['Habitability'] = calculate_habitability(balanced_df)

final_columns = feature_columns + ['Habitability']
balanced_df = balanced_df[final_columns]

balanced_df.to_csv(Dataset_Path + "/Exoplanets_Updated.csv", index=False)

def balance_habitability_data(file_path):
   df = pd.read_csv(file_path, low_memory=False)

   habitable_count = df[df['Habitability'] >= 50].shape[0]
   non_habitable_count = df[df['Habitability'] < 50].shape[0]

   print(f"Habitability Count: {habitable_count}")
   print(f"Non-Habitability Count: {non_habitable_count}")

balanced_df_dummies = pd.get_dummies(balanced_df.drop(columns=['Habitability']), drop_first=True)

X = balanced_df_dummies
y = balanced_df['Habitability']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Custom loss function to penalize extreme predictions.
def custom_loss(y_true,y_pred):
   return K.mean(K.square(y_pred - y_true) * K.cast(K.abs(y_pred - y_true) < K.abs(y_true - K.mean(y_true)), K.floatx()))

# Build a neural network model with batch normalization and dropout.
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1)) 

model.compile(optimizer='adam', loss=custom_loss)

# Early stopping callback to prevent overfitting.
early_stopping_monitor = EarlyStopping(
   monitor='val_loss', 
   patience=10,
   restore_best_weights=True,
   verbose=1
)

# Dynamic learning rate adjustment callback.
reduce_lr_monitor = ReduceLROnPlateau(
   monitor='val_loss', 
   factor=0.5,
   patience=5,
   min_lr=1e-6,
   verbose=1
)

history = model.fit(X_train,
                    y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping_monitor, reduce_lr_monitor],
                    verbose=1)

y_pred_train = model.predict(X_train).flatten()
y_pred_test = model.predict(X_test).flatten()

mse_train = mean_squared_error(y_train,y_pred_train)
mae_train = mean_absolute_error(y_train,y_pred_train)
r2_train = r2_score(y_train,y_pred_train)

mse_test = mean_squared_error(y_test,y_pred_test)
mae_test = mean_absolute_error(y_test,y_pred_test)
r2_test = r2_score(y_test,y_pred_test)

print(f'Train Mean Squared Error: {mse_train}')
print(f'Train Mean Absolute Error: {mae_train}')
print(f'Train R² Score: {r2_train}')

print(f'Test Mean Squared Error: {mse_test}')
print(f'Test Mean Absolute Error: {mae_test}')
print(f'Test R² Score: {r2_test}')

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.scatter(y_train,y_pred_train,color='blue',alpha=0.5,label='Predicted')
plt.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()],color='red',linestyle='--')
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')

plt.subplot(1,2,2)
plt.scatter(y_test,y_pred_test,color='green',alpha=0.5,label='Predicted')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red',linestyle='--')
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Actual Habitability')
plt.ylabel('Predicted Habitability')

plt.tight_layout()
plt.show()

end_time=time()
print("Runtime is : ", round((end_time-start_time),3), "ms")
