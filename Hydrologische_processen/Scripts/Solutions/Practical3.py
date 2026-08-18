#%%
# --- modules ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, rootutils
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- hard coded paths ---
ROOT_PATH = rootutils.find_root(search_from=Path.cwd(), indicator=["environment.yml", "Data", "Scripts"])
sys.path.append(str(ROOT_PATH))
ROOTDIR_DATA = os.path.join(ROOT_PATH, "Data", "processed")
sys.path.append(ROOTDIR_DATA)

# --- other imports ---
from Scripts.scripts_hydrological_model.model import (
    ForcingsTimeSeries,
    Parameters,
    States,
    RainfallRunoffModel,
    Simulation
)
from Scripts.scripts_hydrological_model.calibration import (
    NelderMeadCalibration
)
from Scripts.constants import (
    CATCHMENT_AREA_ZWALM
)
from Scripts.scripts_hydrological_model.ensembles import (
    Perturbations
)

#%% Question 1: Data-analysis of the meteorological forcing data
# --- load the data ---
df_forcings = pd.read_csv(
    os.path.join(ROOTDIR_DATA, "data_zwalm", "forcings_discharge.csv"),
    index_col=0,
    parse_dates=True
)

# --- make function that plots the hydrograph and hyetograph ---
def PlotHydroHyetographTimePeriod(
    df_forcings:pd.DataFrame,
    date_ini:pd.Timestamp = pd.Timestamp("2020-01-01"),
    date_end:pd.Timestamp = pd.Timestamp("2020-12-31"),
    discharge_column:str = "river_discharge",
    precipitation_column:str = "precipitation",
    ) -> plt.plot:
    """Function to plot a hydrograph and hyetograph for a given time period."""

    #---error handling---
    if not isinstance(df_forcings.index, pd.DatetimeIndex):
        raise ValueError("The index of df_forcings must be a DatetimeIndex.")

    #---select data for the time period---
    df_tp=df_forcings.loc[(df_forcings.index >= date_ini) & 
                        (df_forcings.index <= date_end)]

    #---make the plot---
    fig, ax = plt.subplots(figsize=(10, 6))

    #discharge at the bottom
    ax.plot(
        df_tp.index,
        df_tp[discharge_column],
        color="tab:blue",
        lw=2,
    )
    ax.set_ylabel("Discharge (m³ s$^{-1}$)", color="tab:blue", fontsize=18)
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.spines["top"].set_visible(False)

    #precipitation  at the top
    ax2 = ax.twinx()
    ax2.bar(
        df_tp.index,
        df_tp[precipitation_column],
        width=0.9,
        color="tab:orange",
        alpha=0.6,
    )
    ax2.invert_yaxis()
    ax2.set_ylim(df_tp[precipitation_column].max() * 1.1, 0) #0 at the top

    # Move ticks and label to the top
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax2.set_ylabel("Precipitation (mm)", color="tab:orange", fontsize=18)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Cosmetics
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Date [-]", fontsize=18)
    fig.tight_layout()

    ax.tick_params(axis="both", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)

    plt.show()

def PlotMonthlyAggregates(
    df_forcings:pd.DataFrame,
    date_ini:pd.Timestamp = pd.Timestamp("2020-01-01"),
    date_end:pd.Timestamp = pd.Timestamp("2020-12-31"),
    discharge_column:str = "river_discharge",
    evapotranspiration_column:str = "potential_evapotranspiration"
    ) -> plt.plot:
    """Function to plot monthly aggregates for a given time period."""
    #---error handling---
    if not isinstance(df_forcings.index, pd.DatetimeIndex):
        raise ValueError("The index of df_forcings must be a DatetimeIndex.")
    
    #---select data for the time period---
    df_tp=df_forcings.loc[(df_forcings.index >= date_ini) & 
                        (df_forcings.index <= date_end)]

    #---calculate monthly aggregates---
    monthlyAggregates=df_tp[[discharge_column, 
                             evapotranspiration_column]].\
                                groupby([df_tp.index.year,
                                         df_tp.index.month]).mean()
    #NOTE: you can use pandas.groupby("month").mean() to calculate the monthly mean

    #---make the plot---
    fig,ax=plt.subplots(figsize=(10, 6))

    monthlyAggregates[discharge_column].plot(
        kind="bar",
        color="tab:blue",
        ax=ax,
        width=0.8
    )
    
    ax2=ax.twinx()
    monthlyAggregates[evapotranspiration_column].plot(
        color="tab:orange",
        ax=ax2,
        lw=2
    )

    ax.tick_params(axis="both", labelsize=15)
    ax2.tick_params(axis="both", labelsize=15)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_ylabel(""" Set a proper label for the primary axis""", color="tab:blue", fontsize=18)
    ax2.set_ylabel(""" Set a proper label for the secondary axis""", color="tab:orange", fontsize=18)
    ax.set_xlabel("")

    fig.tight_layout()

    plt.show()


# --- plot the hydrograph and hyetograph for the time period 2018-01-01 to 2019-12-31 ---
PlotHydroHyetographTimePeriod(
    df_forcings=df_forcings,
    date_ini=pd.Timestamp("2018-01-01"),
    date_end=pd.Timestamp("2019-12-31")
)

# --- plot the hydrograph and hyetograph for the time period 2018-05-01 to 2018-06-30 ---
PlotHydroHyetographTimePeriod(
    df_forcings=df_forcings,
    date_ini=pd.Timestamp("2018-05-01"),
    date_end=pd.Timestamp("2018-06-30")
)

# --- plot the monthly aggregates for the time period 2018-01-01 to 2019-12-31 ---
PlotMonthlyAggregates(
    df_forcings=df_forcings,
    date_ini = pd.Timestamp("2018-01-01"),
    date_end = pd.Timestamp("2019-12-31")
    )

#%% Question 2: Run the model for the time period of 2010-01-01 to 2019-12-31. Use a warm-up period of 50 days.
#TODO: add the individual parts of the function as an example on how to run the model for a given time period.

# --- function to run the model ---
from Scripts.scripts_hydrological_model.helper_functions import (
    runModelTimePeriod
)

# --- run the model ---
from Scripts.scripts_hydrological_model.helper_functions import (
    runModelTimePeriod
)

def PlotQobsSim(
        df_sim: pd.DataFrame,
        obs_column: str = "Qobs",
        sim_column: str = "Qsim",
        date_ini: pd.Timestamp | None = None,
        date_end: pd.Timestamp | None = None
    ) -> plt.plot:
    '''Function to plot observed and simulated discharge.'''
    if date_ini is not None and date_end is not None:
        df_sim = df_sim.loc[(df_sim.index >= date_ini) & 
                            (df_sim.index <= date_end)]

    # plot the simulated and observed discharge
    fig,ax=plt.subplots(figsize=(12, 5))
    ax.plot(
        df_sim.index,
        df_sim[sim_column],
        color="#6BD117",
        lw=2,
        alpha=1,
        label="Simulated discharge"
    )

    ax.plot(
        df_sim.index,
        df_sim[obs_column],
        color="#5495BB",
        lw=2,
        alpha=0.9,
        label="Observed discharge"
    )

    ax.legend(
        loc="upper left",
        fontsize=15,
        frameon=False
    )
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Discharge (m³ s$^{-1}$)", fontsize=18)
    ax.set_xlabel("Date [-]", fontsize=18)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=15)
    fig.tight_layout()

    plt.show()


# --- run the model for a given time period ---
parameters=Parameters()
states=States()
date_ini=pd.Timestamp("2012-01-01")
date_end=pd.Timestamp("2019-12-31")

df_sim=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=parameters, 
    states=states, 
    date_ini=date_ini, 
    date_end=date_end
    )

# --- plot the observed and simulated discharge ---
PlotQobsSim(
    df_sim=df_sim,
    )

# --- rerun and plot difference between no warm up and warm up period ---
df_sim_nwump=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=parameters, 
    states=states, 
    date_ini=date_ini, 
    date_end=date_end,
    n_warmup_days=0
    )



fig,ax=plt.subplots(figsize=(12, 5))
ax.plot(
    df_sim_nwump.index,
    df_sim_nwump["Qsim"],
    color="#6BD117",
    lw=2,
    alpha=1,
    label="Simulated discharge (no warm-up)"
)
ax.plot(
    df_sim.index,
    df_sim["Qsim"],
    color="#3C81AF",
    lw=2,
    alpha=1,
    label="Simulated discharge (warm-up)"
)
ax.set_xlim([date_ini, pd.Timestamp("2012-07-01")])
ax.set_ylim([0, 20])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Discharge (m³ s$^{-1}$)", fontsize=18)
ax.set_xlabel("Date [-]", fontsize=18)
ax.tick_params(axis="both", labelsize=15)
#%% Question 3: manual calibration of some model parameters
# --- adjust alpha and rerun ---
parameters=Parameters(alpha=0.2)
states=States()
date_ini=pd.Timestamp("2012-01-01")
date_end=pd.Timestamp("2019-12-31")

df_sim=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=parameters, 
    states=states, 
    date_ini=date_ini, 
    date_end=date_end
    )

# --- plot the observed and simulated discharge ---
PlotQobsSim(
    df_sim=df_sim,
    )

# --- adjust Q_p_max and rerun ---
parameters=Parameters(alpha=0.2, Q_p_max=5)
states=States()
date_ini=pd.Timestamp("2012-01-01")
date_end=pd.Timestamp("2019-12-31")

df_sim=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=parameters, 
    states=states, 
    date_ini=date_ini, 
    date_end=date_end
    )

# --- plot the observed and simulated discharge ---
PlotQobsSim(
    df_sim=df_sim,
    )

#%% Question 4: Calibrate the model using Nelder-mead for the time period 2010-01-01 to 2012-12-31. Use a warm-up period of 50 days.
# --- calibrate using NSE ---
def NSE(Qsim: np.ndarray, 
        Qobs: np.ndarray,
        greater_is_better: bool = True
    ) -> float:
    '''Implementation of the Nash-Sutcliffe Efficiency (NSE) as a loss function to evaluate during the calibration of the model parameters.'''

    # implement the NSE
    mask=np.isfinite(Qsim) & np.isfinite(Qobs)
    Qsim = Qsim[mask]
    Qobs = Qobs[mask]

    ss_res = np.sum((Qsim - Qobs) ** 2)
    ss_tot = np.sum((Qobs - np.mean(Qobs)) ** 2)
    loss = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    if Qobs.size == 0:
        return np.nan, greater_is_better

    return loss, greater_is_better

params_calib_NSE, history_calib_NSE = NelderMeadCalibration(
    df_forcings=df_forcings,
    loss_function=NSE,
    start_parameters=Parameters(),
    states=States(),
    date_ini=pd.Timestamp("2010-01-01"),
    date_end=pd.Timestamp("2013-12-31"),
    n_warmup_days=50,
    max_iterations=1000,
    x_tolerance=1e-6,
    objective_tolerance=1e-4,
    verbose=False
)

# --- simulate with the calibrated parameters ---
df_sim_NSE=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=params_calib_NSE,
    states=States(),
    date_ini=pd.Timestamp("2014-01-01"),
    date_end=df_forcings.index.max(),
)

# --- compute the bias between the observed and simulated discharge ---
mask=~np.isnan(df_sim_NSE["Qobs"]) & ~np.isnan(df_sim_NSE["Qsim"])
print(f"The mean error after NSE calibration is: {np.mean(df_sim_NSE[mask]['Qsim']-df_sim_NSE[mask]['Qobs'])}")

# --- calibrate using KGE ---
def KGE(Qsim: np.ndarray, 
        Qobs: np.ndarray,
        greater_is_better: bool = True
    ) -> float:
    '''Implementation of the Kling-Gupta Efficiency (KGE) as a loss function to evaluate during the calibration of the model parameters.'''

    # implement the KGE
    mask=np.isfinite(Qsim) & np.isfinite(Qobs)
    Qsim = Qsim[mask]
    Qobs = Qobs[mask]

    if len(Qsim) < 2:
        return np.nan, greater_is_better

    r = np.corrcoef(Qsim, Qobs)[0, 1]
    alpha = np.std(Qsim) / np.std(Qobs)
    beta = np.mean(Qsim) / np.mean(Qobs)

    loss=1.0 - np.sqrt(
        (r - 1.0) ** 2
        + (alpha - 1.0) ** 2
        + (beta - 1.0) ** 2
    )

    return loss, greater_is_better

params_calib_KGE, history_calib_KGE = NelderMeadCalibration(
    df_forcings=df_forcings,
    loss_function=KGE,
    start_parameters=Parameters(),
    states=States(),
    date_ini=pd.Timestamp("2010-01-01"),
    date_end=pd.Timestamp("2013-12-31"),
    n_warmup_days=50,
    max_iterations=1000,
    x_tolerance=1e-6,
    objective_tolerance=1e-4,
    verbose=False
)

# --- simulate with the KGE and compare with the NSE calibration ---
df_sim_KGE=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=params_calib_KGE,
    states=States(),
    date_ini=pd.Timestamp("2014-01-01"),
    date_end=df_forcings.index.max(),
)

fig,ax=plt.subplots(figsize=(12, 5))
ax.plot(
    df_sim_KGE.index,
    df_sim_KGE["Qsim"],
    color="#6BD117",
    lw=2,
    alpha=1,
    label="Simulated discharge (KGE)"
)
ax.plot(
    df_sim_NSE.index,
    df_sim_NSE["Qobs"],
    color="#3C81AF",
    lw=2,
    ls="-.",
    alpha=1,
    label="Simulated discharge (NSE)"
)
ax.plot(
    df_sim_NSE.index,
    df_sim_NSE["Qsim"],
    color="#FF6347",
    lw=2,
    ls="--",
    alpha=0.7,
    label="Simulated discharge (Original)"
)

ax.set_xlabel("Date [-]", fontsize=18)
ax.set_ylabel("Discharge (m³/s)", fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=15)
ax.legend(frameon=False, fontsize=15, loc="upper left")
ax.set_xlim([pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31")])
ax.set_ylim([0, 12])
plt.show()

# --- metrics ---
# Correlation 
mask=~np.isnan(df_sim_KGE["Qobs"])
r_KGE=df_sim_KGE.loc[mask]["Qobs"].corr(df_sim_KGE.loc[mask]["Qsim"])
r_NSE=df_sim_NSE.loc[mask]["Qobs"].corr(df_sim_NSE.loc[mask]["Qsim"])

print(f"Correlation coefficient (KGE): {r_KGE:.3f}")
print(f"Correlation coefficient (NSE): {r_NSE:.3f}")

# Bias
bias_KGE=np.mean(df_sim_KGE.loc[mask]["Qsim"]-df_sim_KGE.loc[mask]["Qobs"])
bias_NSE=np.mean(df_sim_NSE.loc[mask]["Qsim"]-df_sim_NSE.loc[mask]["Qobs"])
print(f"Bias (KGE): {bias_KGE:.3f}")
print(f"Bias (NSE): {bias_NSE:.3f}")


#%% Question 5: Compute for the NSE calibration whether a hydropower plant that should have 2 m3/s for at least 20% of the time.
# --- function to compute the cumulative distribution function (CDF) of the simulated discharge ---
def ComputeCDF(
        df_sim: pd.DataFrame
    ) -> pd.DataFrame:
    '''Function to compute the cumulative distribution function (CDF) of the simulated discharge.'''
    df_cdf=df_sim.sort_values(by="Qsim")
    unique=df_cdf["Qsim"].unique()

    cdf=[(df_cdf["Qsim"]<=unique[i]).sum()/len(df_cdf) for i in range(len(unique))]

    cdf=pd.DataFrame({"Qsim":unique, "CDF":cdf})

    return cdf

# ---- compute the CDF of the simulated discharge for the NSE calibration ---
min_discharge=6

# compute the CDF of the simulated discharge
CDF = ComputeCDF(df_sim_NSE)

# make a plot of the CDF of the simulated discharge
fig,ax = plt.subplots(figsize=(10, 6))
ax.plot(CDF["Qsim"], 1-CDF["CDF"], color="tab:blue", lw=2, label="Probability of exceedance")
ax.axvline(x=min_discharge, color="tab:orange", lw=2, ls="--", label=f"Discharge threshold: {min_discharge} m³/s")
ax.tick_params(axis="both", labelsize=15)
ax.set_xlabel("Discharge (m³ s$^{-1}$)", fontsize=18)
ax.set_ylabel("Probability of exceedance [-]", fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=15, loc="upper right", frameon=False)
fig.tight_layout()
plt.show()

# compute the percentage of time that the discharge is above 2 m3/s
pctge = (1-CDF.iloc[np.argmin(np.abs(CDF["Qsim"]-min_discharge))]["CDF"]).item()*100
print(f"The percentage of time that the discharge is above {min_discharge} m³/s is {pctge:.2f}%")