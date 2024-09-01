from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd
import time
import numpy as np


start_time = time.time()

weights = {
    'pl_bmasse': 0.15,              # Planet mass (Earth masses)
    'pl_insol': 0.15,               # Insolation flux received by the planet
    'pl_eqt': 0.15,                 # Equilibrium temperature of the planet
    'pl_orbper': 0.1,               # Orbital period of the planet
    'pl_orbsmax': 0.1,              # Semi-major axis of the orbit
    'st_teff': 0.1,                 # Effective temperature of the host star
    'pl_rade': 0.05,                # Planet radius (Earth radii)
    'pl_radj': 0.05,                # Planet radius (Jupiter radii)
    'pl_bmassj': 0.05,              # Planet mass (Jupiter masses)
    'pl_orbeccen': 0.05,            # Orbital eccentricity of the planet
    'st_rad': 0.05,                 # Radius of the host star
    'st_mass': 0.05,                # Mass of the host star
    'st_met': 0.05,                 # Metallicity of the host star
    'st_logg': 0.05,                # Surface gravity of the host star
    'sy_dist': 0.05,                 # Distance to the system from Earth

    # Zero weight variables
    'pl_orbpererr1': 0.0,           # Orbital period error (1st)
    'pl_orbpererr2': 0.0,           # Orbital period error (2nd)
    'pl_orbperlim': 0.0,            # Orbital period limit flag
    'pl_orbsmaxerr1': 0.0,          # Semi-major axis error (1st)
    'pl_orbsmaxerr2': 0.0,          # Semi-major axis error (2nd)
    'pl_orbsmaxlim': 0.0,           # Semi-major axis limit flag
    'pl_radeerr1': 0.0,             # Planet radius error (Earth radii, 1st)
    'pl_radeerr2': 0.0,             # Planet radius error (Earth radii, 2nd)
    'pl_radelim': 0.0,              # Planet radius limit flag (Earth radii)
    'pl_radjerr1': 0.0,             # Planet radius error (Jupiter radii, 1st)
    'pl_radjerr2': 0.0,             # Planet radius error (Jupiter radii, 2nd)
    'pl_radjlim': 0.0,              # Planet radius limit flag (Jupiter radii)
    'pl_bmasseerr1': 0.0,           # Planet mass error (Earth masses, 1st)
    'pl_bmasseerr2': 0.0,           # Planet mass error (Earth masses, 2nd)
    'pl_bmasselim': 0.0,            # Planet mass limit flag (Earth masses)
    'pl_bmassjerr1': 0.0,           # Planet mass error (Jupiter masses, 1st)
    'pl_bmassjerr2': 0.0,           # Planet mass error (Jupiter masses, 2nd)
    'pl_bmassjlim': 0.0,            # Planet mass limit flag (Jupiter masses)
    'pl_bmassprov': 0.0,            # Planet mass measurement method
    'pl_orbeccenerr1': 0.0,         # Orbital eccentricity error (1st)
    'pl_orbeccenerr2': 0.0,         # Orbital eccentricity error (2nd)
    'pl_orbeccenlim': 0.0,          # Orbital eccentricity limit flag
    'pl_insolerr1': 0.0,            # Insolation flux error (1st)
    'pl_insolerr2': 0.0,            # Insolation flux error (2nd)
    'pl_insollim': 0.0,             # Insolation flux limit flag
    'pl_eqterr1': 0.0,              # Equilibrium temperature error (1st)
    'pl_eqterr2': 0.0,              # Equilibrium temperature error (2nd)
    'pl_eqtlim': 0.0,               # Equilibrium temperature limit flag
    'ttv_flag': 0.0,                # Transit timing variation flag
    'st_spectype': 0.0,             # Spectral type of the host star
    'st_tefferr1': 0.0,             # Effective temperature error (1st)
    'st_tefferr2': 0.0,             # Effective temperature error (2nd)
    'st_tefflim': 0.0,              # Effective temperature limit flag
    'st_raderr1': 0.0,              # Radius error of the host star (1st)
    'st_raderr2': 0.0,              # Radius error of the host star (2nd)
    'st_radlim': 0.0,               # Radius limit flag of the host star
    'st_masserr1': 0.0,             # Mass error of the host star (1st)
    'st_masserr2': 0.0,             # Mass error of the host star (2nd)
    'st_masslim': 0.0,              # Mass limit flag of the host star
    'st_meterr1': 0.0,              # Metallicity error of the host star (1st)
    'st_meterr2': 0.0,              # Metallicity error of the host star (2nd)
    'st_metlim': 0.0,               # Metallicity limit flag of the host star
    'st_metratio': 0.0,             # Metallicity ratio of the host star
    'st_loggerr1': 0.0,             # Surface gravity error (1st)
    'st_loggerr2': 0.0,             # Surface gravity error (2nd)
    'st_logglim': 0.0,              # Surface gravity limit flag
    'rastr': 0.0,                   # Right Ascension string
    'ra': 0.0,                      # Right Ascension in degrees
    'decstr': 0.0,                  # Declination string
    'dec': 0.0,                     # Declination in degrees
    'sy_disterr1': 0.0,             # Distance error (1st)
    'sy_disterr2': 0.0,             # Distance error (2nd)
    'sy_vmag': 0.0,                 # Visual magnitude
    'sy_vmagerr1': 0.0,             # Visual magnitude error (1st)
    'sy_vmagerr2': 0.0,             # Visual magnitude error (2nd)
    'sy_kmag': 0.0,                 # System K magnitude
    'sy_kmagerr1': 0.0,             # System K magnitude error (1st)
    'sy_kmagerr2': 0.0,             # System K magnitude error (2nd)
    'sy_gaiamag': 0.0,              # System Gaia magnitude
    'sy_gaiamagerr1': 0.0,          # System Gaia magnitude error (1st)
    'sy_gaiamagerr2': 0.0,          # System Gaia magnitude error (2nd)
    'rowupdate': 0.0,               # Row update date
    'pl_pubdate': 0.0,              # Publication date
    'releasedate': 0.0,             # Data release date

    # Metadata and flags, not relevant to habitability
    'pl_name': 0.0,                 # Planet name
    'hostname': 0.0,                # Host star name
    'default_flag': 0.0,            # Default flag
    'sy_snum': 0.0,                 # Number of stars in system
    'sy_pnum': 0.0,                 # Number of planets in system
    'discoverymethod': 0.0,         # Discovery method
    'disc_year': 0.0,               # Discovery year
    'disc_facility': 0.0,           # Discovery facility
    'soltype': 0.0,                 # Solution type
    'pl_controv_flag': 0.0          # Controversial flag
}


def calculate_habitability(row, weights):
    # Assign weights to each feature based on its importance in determining habitability
    habitability = 0
    # Normalize each feature and multiply by its weight
    if feature.isnumeric():
        for feature, weight in weights.items():
            normalized_value = (row[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
            habitability += weight * normalized_value

    # Ensure the habitability score is between 0 and 1
    habitability = round((min(1, max(0, habitability))) * 100,3)
    return habitability

df = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets Trial.csv")

data_cleaned = df.head()
data_cleaned.to_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Exoplanet_habitability_with_sreedharshan\Exoplanets Trial.csv",index=False)
try:
    for col in data_cleaned:
        if col in weights:
            data_cleaned['Habitability'] = data_cleaned.apply(calculate_habitability(col,weights=weights))
        else:
            print(f'missing feature: {col}')
except Exception as e:
    print(f"Error Occured on column {col} !!! ",e)

end_time = time.time()
print("Total Time : ",round(end_time-start_time, 2)," ms")
