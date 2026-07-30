''' 
Script to convert the data of the old practicals (<= 2023) to the one used 
for the new practicals (>= 2024). 

The scripts reads in the data from Shanon and Zdenko (for more info contact: gabrielle.delannoy@kuleuven.be)
and converts it to the the following scripts:
- Data/processed/Afvoer_Demerbekken.csv
- Data/processed/Meteo_Demerbekken.csv

created on Tuesday 16/07/26 by Lucas Boeykens - lucas.boeykens@kuleuven.be lucas.boeykens@ugent.be
'''

# --- modules ---
import pandas as pd 
import os, sys, rootutils
from pathlib import Path

ROOT_PATH = rootutils.find_root(search_from=Path.cwd(), indicator=".git")
sys.path.append(str(ROOT_PATH))
ROOTDIR_DATA = os.path.join(ROOT_PATH, "Data")
sys.path.append(ROOTDIR_DATA)

# --- hard coded constants ---
start_date_data=pd.Timestamp("2000-12-31 23:59:59")


# ---read in data ---
excel_old_practicals = os.path.join(ROOTDIR_DATA, 
                        "data_download", 
                        "catchment_template_bare_ZdenkoShanon.xlsx"
                        )

dfQdemer = pd.read_excel(excel_old_practicals, 
                         sheet_name="Demer_afvoer", 
                         header=4
                        )

dfMeteodemer = pd.read_excel(excel_old_practicals, 
                         sheet_name="Demer_meteo", 
                         header=4
                        )

# --- select data for the time period after the start date ---
dfQdemer_subset=dfQdemer.copy()
dfQdemer_subset["Datum_tijd"] = pd.to_datetime(dfQdemer_subset["Datum_tijd"], dayfirst=True)
dfQdemer_subset=dfQdemer_subset.loc[dfQdemer_subset["Datum_tijd"]>start_date_data].reset_index(drop=True)

dfMeteodemer_subset=dfMeteodemer.copy()
dfMeteodemer_subset=dfMeteodemer_subset.loc[dfMeteodemer_subset["Datum"]>start_date_data].reset_index(drop=True)

# --- reset the hours to proper values -> only for the discharge data ---
datum_tijd = dfQdemer_subset[["Datum_tijd"]].copy()

datum_tijd["Datum_tijd"] = pd.to_datetime(datum_tijd["Datum_tijd"], dayfirst=True)
datum_tijd = datum_tijd.sort_values("Datum_tijd").reset_index(drop=True)
datum_tijd["Datum"] = datum_tijd["Datum_tijd"].dt.normalize()
datum_tijd["Tijd"] = datum_tijd.groupby("Datum").cumcount().map(lambda hour: f"{hour:02d}:00")
datum_tijd["Datum_tijd"] = datum_tijd["Datum"] + pd.to_timedelta(datum_tijd.groupby("Datum").cumcount(), unit="h")

dfQdemer_subset["Datum_tijd"] = datum_tijd["Datum_tijd"]

# --- save the processed data to a new CSV file ---
dfQdemer_subset.to_csv(os.path.join(ROOTDIR_DATA, 
                                    "processed", 
                                    "Afvoer_Demerbekken.csv"
                                    ), 
                        index=False
                        )

dfMeteodemer_subset.to_csv(os.path.join(ROOTDIR_DATA, 
                                    "processed", 
                                    "Meteo_Demerbekken.csv"
                                    ), 
                        index=False
                        )