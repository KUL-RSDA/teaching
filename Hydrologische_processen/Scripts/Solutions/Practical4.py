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
from Scripts.scripts_hydrological_model.helper_functions import (
    runModelTimePeriod
)
from Scripts.scripts_hydrological_model.ensembles import (
    Perturbations
)
#%% Question 1: Perturb the forcings
# --- open the forcings --- 
df_forcings = pd.read_csv(
    os.path.join(ROOTDIR_DATA, "data_zwalm", "forcings_discharge.csv"),
    index_col=0,
    parse_dates=True
)

# --- calibrate the model using the Nelder-Mead algorithm with the NSE---
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

params_calib_NSE, _ = NelderMeadCalibration(
    df_forcings=df_forcings,
    loss_function=NSE,
    start_parameters=Parameters(),
    states=States(),
    date_ini=pd.Timestamp("2010-01-01"),
    date_end=pd.Timestamp("2014-12-31"),
    n_warmup_days=50,
    max_iterations=1000,
    x_tolerance=1e-6,
    objective_tolerance=1e-4,
    verbose=False
)

# --- make plot of the simulations and compute certain metrics ---
date_ini=pd.Timestamp("2015-01-01")
date_end=df_forcings.index.max()

# Simulate the model using the calibrated parameters
df_sim=runModelTimePeriod(
    df_forcings=df_forcings,
    parameters=params_calib_NSE, 
    states=States(), 
    date_ini=date_ini, 
    date_end=date_end
    )

# plot the simulated and observed discharge
fig,ax=plt.subplots(figsize=(12, 5))
ax.plot(
    df_sim.index,
    df_sim["Qsim"],
    color="#6BD117",
    lw=2,
    alpha=1,
    label="Simulated discharge"
)

ax.plot(
    df_sim.index,
    df_sim["Qobs"],
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

# Compute certain metrics
mask=np.isfinite(df_sim["Qsim"]) & np.isfinite(df_sim["Qobs"])
df_sim = df_sim[mask]
corr=df_sim["Qsim"].corr(df_sim["Qobs"])
bias=np.mean(df_sim["Qsim"]-df_sim["Qobs"])
print(f"Correlation: {corr:.3f}")
print(f"Bias: {bias:.3f} m³/s")

# --- create a Perturbations object to generate perturbed forcings ---
model=RainfallRunoffModel(
    catchment_area=CATCHMENT_AREA_ZWALM,
    states=States(),
    parameters=params_calib_NSE
)

perturbations=Perturbations(
    df_forcings=df_forcings,
    sigma_P=0.5,
    sigma_ET=0.25,
    model=model,
    n_perturbations=30,
    date_ini=date_ini,
    date_end=date_end,
    n_warmup_days=100
)

# --- perturb and get the variance of the 10th ensemble for P ---
perturbed_forcings=perturbations.GeneratePerturbedForcings()

var=perturbed_forcings[[c for c in perturbed_forcings.columns 
                        if "precip" in c and "10" in c]].std().item()**2

print(f"Variance of the 10th ensemble for precipitation: {var:.3f} mm²")

# --- plot the perturbed forcings ---
perturbations.PlotPerturbedForcings()

#%% Question 2: run the open loop simulation
# --- let the model do the open loop ---
perturbations.open_loop()
results = perturbations.open_loop_results

# --- plot the mean discharge and plot the min and maximum of the ensemble for the time period 2019-01-01 2021-12-31 ---
date_ini = pd.Timestamp("2019-01-01")
date_end = pd.Timestamp("2020-12-31")

df_plot = results.loc[(results.index >= date_ini) & (results.index <= date_end)]
fig,ax=plt.subplots(figsize=(12,6))
# plot the mean discharge
ax.plot(df_plot.index, df_plot["Qmean"], color='tab:blue', label='Mean Discharge', lw=2)
# plot the min and max of the ensemble
ax.fill_between(df_plot.index, df_plot["Qmin"], df_plot["Qmax"], color='tab:blue', alpha=0.2, label='Min-Max Range')
# plot the observed discharge
ax.plot(df_plot.index, df_plot["Qobs"], color='tab:orange', label='Observed Discharge', lw=2, alpha=0.6)
ax.legend(loc="upper left", fontsize=15, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylabel("Discharge [m$^3$s$^{-1}$]", fontsize=15)
ax.set_xlabel("Date [-]", fontsize=15)

# --- plot the soil moisture reservoir and associated uncertainty (+- 1 std dev) ---
fig,ax=plt.subplots(figsize=(12,6))
# plot the mean discharge
ax.plot(df_plot.index, df_plot["Smean"], color='tab:blue', label='Mean Discharge', lw=2)
# plot the min and max of the ensemble
ax.fill_between(df_plot.index, df_plot["Smean"] - df_plot["Sstd"], df_plot["Smean"] + df_plot["Sstd"], color='tab:blue', alpha=0.2, label='+- 1 std dev')
ax.legend(loc="upper left", fontsize=15, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylabel("S [mm]", fontsize=15)
ax.set_xlabel("Date [-]", fontsize=15)

# --- optional: plot the forcings for that time period ---
perturbations.PlotPerturbedForcings(date_ini=date_ini, date_end=date_end)