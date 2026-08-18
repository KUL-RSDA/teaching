# --- modules ---
from attrs import fields
import pandas as pd
import os, sys, rootutils
from pathlib import Path

ROOT_PATH = rootutils.find_root(search_from=Path.cwd(), indicator=["environment.yml", "Data", "Scripts"])
sys.path.append(str(ROOT_PATH))

from Scripts.constants import (
    CATCHMENT_AREA_ZWALM
)
from .model import (
    RainfallRunoffModel,
    ForcingsTimeSeries,
    Simulation,
    Parameters,
    States
)

# --- functions ---
def get_parameter_bounds(parameters):
    return {
        f.name: tuple(f.metadata["bounds"])
        for f in fields(type(parameters))
        if "bounds" in f.metadata
    }

def runModelTimePeriod(
        df_forcings: pd.DataFrame,
        parameters: Parameters,
        states: States,
        date_ini:pd.Timestamp = pd.Timestamp("2010-01-01"),
        date_end:pd.Timestamp = pd.Timestamp("2021-12-31"),
        saved_variables: list = ["S", "S_1", "S_2", "Q_m3s", "Q_1", "Q_2"],
        n_warmup_days:int = 50,
        QobsCol: str = "river_discharge",
        catchment_area: float = CATCHMENT_AREA_ZWALM
    ) -> pd.DataFrame:
    '''Run the rainfall-runoff model and return the simulated and observed discharge.'''
    # --- Instantiate the model with parameters and states ---
    model = RainfallRunoffModel(
        catchment_area=catchment_area,
        states=states,
        parameters=parameters,
    )

    # --- use the model to run a simulation with the forcings ---
    #generate the forcing time series object
    forcing_timeseries = ForcingsTimeSeries(
        precipitation=df_forcings["precipitation"].values,
        potential_evaporation=df_forcings["potential_evapotranspiration"].values,
        time=df_forcings.index.values,
    )

    # change the start date to account for the warmup period
    date_ini_wrmup = date_ini-pd.Timedelta(days=n_warmup_days)
    date_ini_wrmup = df_forcings.index[0] if date_ini_wrmup < df_forcings.index[0] else date_ini_wrmup

    # generate the simulation object and run the simulation
    sim = Simulation(
        model=model,
        forcing_time_series=forcing_timeseries,
        output_vars=saved_variables,
        start_time=date_ini_wrmup,
        end_time=date_end
    )
    sim.run()

    # --- return the simulated discharge together with the measured discharge ---
    # simulated discharge
    output=sim.output
    output=output.loc[(output.index >= date_ini) & (output.index <= date_end)]

    # observed discharge
    Qobs=df_forcings.loc[(df_forcings.index >= date_ini) & 
                        (df_forcings.index <= date_end)][QobsCol]

    # merge the simulated and observed discharge into a single dataframe
    df_results=pd.merge(output, Qobs, left_index=True, right_index=True, how="inner").\
        rename(columns={"Q_m3s":"Qsim", QobsCol:"Qobs"})
    
    del sim, model, forcing_timeseries, output, Qobs

    return df_results