# --- modules ---
import numpy as np
import pandas as pd
from attrs import evolve
from itertools import count
from scipy.optimize import minimize
import pandas as pd 
import numpy as np
import sys, rootutils, warnings
from pathlib import Path
from collections.abc import Callable

ROOT_PATH = rootutils.find_root(search_from=Path.cwd(), indicator=["environment.yml", "Data", "Scripts"])
sys.path.append(str(ROOT_PATH))

from Scripts.scripts_hydrological_model.model import (
    ForcingsTimeSeries,
    Parameters,
    RainfallRunoffModel,
    Simulation,
    States
)

from Scripts.constants import (
    CATCHMENT_AREA_ZWALM,
)

from Scripts.scripts_hydrological_model.helper_functions import (
    get_parameter_bounds
)

# --- functions ---
def runModel(
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
    Qsim=output["Q_m3s"]

    # observed discharge
    Qobs=df_forcings.loc[(df_forcings.index >= date_ini) & 
                        (df_forcings.index <= date_end)][QobsCol]

    # merge the simulated and observed discharge into a single dataframe
    df_discharge=pd.merge(Qsim, Qobs, left_index=True, right_index=True, how="inner").\
        rename(columns={"Q_m3s":"Qsim", QobsCol:"Qobs"})
    
    del sim, model, forcing_timeseries, output, Qsim, Qobs

    return df_discharge


def NelderMeadCalibration(
    df_forcings: pd.DataFrame,
    loss_function: Callable[[np.ndarray, np.ndarray, bool], tuple[float, bool]],
    start_parameters: Parameters = None,
    states: States|None = None,
    date_ini: pd.Timestamp = pd.Timestamp("2010-01-01"),
    date_end: pd.Timestamp = pd.Timestamp("2021-12-31"),
    n_warmup_days: int = 50,
    max_iterations: int = 1000,
    x_tolerance: float = 1e-6,
    objective_tolerance: float = 1e-6,
    verbose: bool = True
    ) -> tuple[Parameters, pd.DataFrame]:
    """
    Calibrate the rainfall-runoff model by maximizing KGE with Nelder-Mead.
    """

    # -- set defaults ---
    if start_parameters is None:
        start_parameters = Parameters()
    if states is None:
        states = States(S=0.0, S_1=0.0, S_2=0.0)

    # --- get boundaries of the parameters to calibrate ---
    bounds_dict = get_parameter_bounds(start_parameters) #retrieves the boundaries of the parameters to calibrate as a dictionary
    parameter_names = list(bounds_dict) # retrieve the names of the parameters to calibrate

    bounds = [bounds_dict[name] for name in parameter_names] # make list of tuples with the boundaries of the parameters to calibrate

    # --- convert the starting parameters to a vector of values in physical space ---
    x0 = np.array(
        [getattr(start_parameters, name) for name in parameter_names],
        dtype=float,
    ) #initial parameter values in physical space


    # --- define the objective function to minimize ---
    history = []
    counter = count(1)
    last_score, last_parameters = None, None

    def vector_to_parameters(x) -> Parameters:
        '''Convert a vector of parameter values to a Parameters object.'''
        
        return evolve(
            start_parameters,
            **dict(zip(parameter_names, x)),
        )

    def objective(x: np.ndarray) -> float:
        '''
        objective function to minimize during the calibration of the model parameters.
        
        Keep in mind that the Nelder-Mead algorithm minimizes this objective function!
        '''
        nonlocal last_score, last_parameters
        
        # update the parameters based on the current vector of values
        parameters = vector_to_parameters(x)

        # run the model with the current parameters and get the simulated discharge
        df_sim = runModel(
            parameters=parameters,
            states=states,
            df_forcings=df_forcings,
            date_ini=date_ini,
            date_end=date_end,
            n_warmup_days=n_warmup_days
        )

        # compute the loss function
        score, greater_is_better = loss_function(
                                    Qsim=df_sim["Qsim"].values,
                                    Qobs= df_sim["Qobs"].values
                                    )

        # if the score is not finite (e.g., NaN or Inf), return a large penalty value
        if not np.isfinite(score):
            score=1e6 if greater_is_better else -1e6

        history.append(
            {
                "evaluation": len(history),
                "score": score,
                **{
                    name: getattr(parameters, name)
                    for name in parameter_names
                },
            }
        )
        last_score = score
        last_parameters = parameters

        return -score if greater_is_better else score

    # -- callback ---
    counter = count(1)
    def callback(xk):
        iteration = next(counter)

        print(
            f"Iteration {iteration}: "
            f"Loss = {last_score:.6f}, "
            f"parameters = {last_parameters}"
        ) if iteration % 10 == 0 else None

    # --- run the Nelder-Mead optimization ---
    result = minimize(
        fun=objective,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        callback=callback if verbose else None,
        options={
            "maxiter": max_iterations,
            "xatol": x_tolerance,
            "fatol": objective_tolerance,
            "adaptive": True,
            "disp": verbose,
        }
        )
    if not result.success:
        warnings.warn(
            f"Calibration did not converge: {result.message}",
            RuntimeWarning,
        )
        
    calibrated_parameters = vector_to_parameters(result.x)
    calibration_history = pd.DataFrame(history)

    return calibrated_parameters, calibration_history