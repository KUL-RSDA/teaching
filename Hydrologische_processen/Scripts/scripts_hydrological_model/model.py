from pathlib import Path
import rootutils, sys
ROOT_PATH = rootutils.find_root(search_from=__file__, indicator=".git")
sys.path.append(str(ROOT_PATH))
import numpy as np
import pandas as pd
from attrs import define, field
from Scripts.constants import (
    MILLIMETER_PER_M,
    SECONDS_PER_DAY,
    SQUARE_METERS_PER_SQUARE_KILOMETER,
)
from scipy.integrate import solve_ivp

@define
class Parameters:
    lambda_: float = field(
        default=0.65,
        metadata={
            "description": "Evaporation reduction parameter",
            "bounds": (0.1, 1.5),
            "units": "-",
        },
    )
    S_max: float = field(
        default=350.0,
        metadata={
            "description": "Storage capacity of unsaturated soil reservoir",
            "bounds": (87.0, 730.0),
            "units": "mm",
        },
    )
    b: float = field(
        default=0.86,
        metadata={
            "description": "Infiltration shape parameter",
            "bounds": (0.2, 1.0),
            "units": "-",
        },
    )
    alpha: float = field(
        default=0.8,
        metadata={
            "description": "Partitioning parameter for excess precipitation",
            "bounds": (0.1, 1.0),
            "units": "-",
        },
    )
    Q_p_max: float = field(
        default=34.0,
        metadata={
            "description": "Maximum percolation rate",
            "bounds": (0.75, 75.0),
            "units": "mm/d",
        },
    )
    beta: float = field(
        default=0.06,
        metadata={
            "description": "Percolation shape parameter",
            "bounds": (0.05, 0.1),
            "units": "-",
        },
    )
    gamma: float = field(
        default=9.8,
        metadata={
            "description": "Non-linear reservoir parameter",
            "bounds": (5.0, 10.0),
            "units": "-",
        },
    )
    S_2_max: float = field(
        default=14.0,
        metadata={
            "description": "Storage capacity of fast and slow reacting reservoirs",
            "bounds": (13.0, 26.0),
            "units": "mm",
        },
    )
    kappa_2: float = field(
        default=110.0,
        metadata={
            "description": "Maximum outflow rate of fast reacting reservoir",
            "bounds": (25.0, 250.0),
            "units": "mm/d",
        },
    )
    kappa_1: float = field(
        default=0.32,
        metadata={
            "description": "Reciprocal of residence time for slow reacting reservoir",
            "bounds": (0.16, 0.52),
            "units": "1/d",
        },
    )


@define
class Fluxes:
    E_a: float = field(
        default=None,
        metadata={
            "description": "Actual evaporation",
            "units": "mm/d",
        },
    )
    P_exc: float = field(
        default=None,
        metadata={
            "description": "Excess precipitation (i.e. excluding infiltration)",
            "units": "mm/d",
        },
    )
    P_in: float = field(
        default=None,
        metadata={
            "description": "Precipitation infiltrating into soil",
            "units": "mm/d",
        },
    )
    Q_p: float = field(
        default=None,
        metadata={
            "description": "Percolation from unsaturated soil to slow reservoir",
            "units": "mm/d",
        },
    )
    P_1: float = field(
        default=None,
        metadata={
            "description": "Excess precipitation reaching slow reservoir",
            "units": "mm/d",
        },
    )
    P_2: float = field(
        default=None,
        metadata={
            "description": "Excess precipitation reaching fast reservoir",
            "units": "mm/d",
        },
    )
    Q_1: float = field(
        default=None,
        metadata={
            "description": "Outflow from slow reacting reservoir",
            "units": "mm/d",
        },
    )
    Q_2: float = field(
        default=None,
        metadata={
            "description": "Outflow from fast reacting reservoir",
            "units": "mm/d",
        },
    )
    Q: float = field(
        default=None,
        metadata={
            "description": "Total discharge",
            "units": "mm/d",
        },
    )
    Q_m3s: float = field(
        default=None,
        metadata={
            "description": "Total discharge in m^3/s, using catchment area for conversion",
            "units": "m^3/s",
        },
    )


@define
class States:
    S: float = field(
        default=0.0,
        metadata={
            "description": "Storage in unsaturated soil reservoir",
            "units": "mm",
        },
    )
    S_1: float = field(
        default=0.0,
        metadata={
            "description": "Storage in slow reacting reservoir",
            "units": "mm",
        },
    )
    S_2: float = field(
        default=0.0,
        metadata={
            "description": "Storage in fast reacting reservoir",
            "units": "mm",
        },
    )


@define
class Forcings:
    precipitation: float = field(
        default=None, metadata={"description": "Precipitation", "units": "mm/d"}
    )
    potential_evaporation: float = field(
        default=None,
        metadata={"description": "Potential evaporation", "units": "mm/d"},
    )


@define
class ForcingsTimeSeries:
    precipitation: np.ndarray = field(
        metadata={"description": "Precipitation time series", "units": "mm/d"}
    )
    potential_evaporation: np.ndarray = field(
        metadata={"description": "Potential evaporation time series", "units": "mm/d"}
    )
    time: np.ndarray = field(
        metadata={
            "description": "Time array corresponding to forcings",
            "units": "days since start of simulation",
        }
    )


@define
class RainfallRunoffModel:
    catchment_area: float = field(metadata={"units": "km^2"})
    states: States = field(factory=States)
    parameters: Parameters = field(factory=Parameters)
    forcings: Forcings = field(factory=Forcings)
    fluxes: Fluxes = field(factory=Fluxes)
    clock: float = field(
        default=0.0,
        metadata={
            "description": "Current simulation time",
            "units": "days since start of simulation",
        },
    )

    def get_state(self):
        return np.array([
            self.states.S,
            self.states.S_1,
            self.states.S_2,
        ])

    def set_state(self, x):
        self.states.S = x[0]
        self.states.S_1 = x[1]
        self.states.S_2 = x[2]

    def get_var_by_name(self, var_name):
        if hasattr(self.states, var_name):
            return getattr(self.states, var_name)
        if hasattr(self.parameters, var_name):
            return getattr(self.parameters, var_name)
        if hasattr(self.forcings, var_name):
            return getattr(self.forcings, var_name)
        if hasattr(self.fluxes, var_name):
            return getattr(self.fluxes, var_name)
        raise ValueError(
            f"Variable '{var_name}' not found in model states, parameters, forcings, or fluxes."
        )

    def mmd_to_m3s(self, Q_mmd):
        """Convert discharge from mm/d to m^3/s using catchment area"""
        return (
            Q_mmd
            / MILLIMETER_PER_M
            * self.catchment_area
            * SQUARE_METERS_PER_SQUARE_KILOMETER
            / SECONDS_PER_DAY
        )

    def m3s_to_mmd(self, Q_m3s):
        """Convert discharge from m^3/s to mm/d using catchment area"""
        return (
            Q_m3s
            * SECONDS_PER_DAY
            / (self.catchment_area * SQUARE_METERS_PER_SQUARE_KILOMETER)
            * MILLIMETER_PER_M
        )

    def calculate_fluxes(self, t, x) -> Fluxes:
        """
        Calculate model fluxes based on current states, parameters, and forcings

        Parameters
        ---------
        t: float
            Current simulation time (in days since start of simulation)
        x: array-like
            Current model states (S, S_1, S_2)

        Returns
        -------
        Fluxes
            Calculated model fluxes based on current states, parameters, and forcings
        """
        S, S_1, S_2 = x

        # error handling related to S and S_max
        S = self.parameters.S_max if S > self.parameters.S_max else S

        # Fluxes relevant to soil moisture reservoir
        E_a = (
            1
            / self.parameters.lambda_
            * S
            / self.parameters.S_max
            * self.forcings.potential_evaporation
        )
        P_in = (
            1 - S / self.parameters.S_max
        ) ** self.parameters.b * self.forcings.precipitation
        P_exc = self.forcings.precipitation - P_in
        Q_p = self.parameters.Q_p_max * (
            1 - np.exp(-self.parameters.beta * S / self.parameters.S_max)
        )

        # Partitioning of excess precipitation
        P_2 = self.parameters.alpha * S / self.parameters.S_max * P_exc
        P_1 = P_exc - P_2

        # Outflow fluxes
        Q_1 = self.parameters.kappa_1 * S_1
        Q_2 = (
            self.parameters.kappa_2
            * (S_2 / self.parameters.S_2_max) ** self.parameters.gamma
        )
        Q = Q_1 + Q_2
        Q_m3s = self.mmd_to_m3s(Q)
        return Fluxes(
            E_a=E_a,
            P_exc=P_exc,
            P_in=P_in,
            Q_p=Q_p,
            P_1=P_1,
            P_2=P_2,
            Q_1=Q_1,
            Q_2=Q_2,
            Q=Q,
            Q_m3s=Q_m3s,
        )

    def rhs(self, t, x):
        """Calculate right hand side of ODE based on current states"""
        fluxes = self.calculate_fluxes(t, x)
        dS_dt = fluxes.P_in - fluxes.E_a - fluxes.Q_p
        dS_1_dt = fluxes.P_1 - fluxes.Q_1
        dS_2_dt = fluxes.P_2 + fluxes.Q_p - fluxes.Q_2
        return np.array([dS_dt, dS_1_dt, dS_2_dt])

    def euler_step(self, dt):
        """Advance model states by one time step using Euler method"""
        # Update states using Euler method and ensure they remain within physical bounds
        dS_dt, dS_1_dt, dS_2_dt = self.rhs(self.clock, self.get_state())
        self.set_state([
            np.clip(self.states.S + dS_dt * dt, 0, self.parameters.S_max),
            np.clip(self.states.S_1 + dS_1_dt * dt, 0, self.parameters.S_2_max),
            np.clip(self.states.S_2 + dS_2_dt * dt, 0, self.parameters.S_2_max),
        ])
        # Update fluxes given the new state
        self.fluxes = self.calculate_fluxes(self.clock, self.get_state())
        # Advance clock
        self.clock += dt

    def scipy_step(self, dt, **kwargs):
        # Use scipy to advance model over one time step
        t_span = (self.clock, self.clock + dt)
        x0 = self.get_state()
        sol = solve_ivp(
            fun=self.rhs, t_span=t_span, y0=x0, t_eval=[self.clock + dt], **kwargs
        )
        # Update state
        self.set_state(sol.y[:, -1])
        # Update fluxes given the new state
        self.fluxes = self.calculate_fluxes(self.clock, self.get_state())
        # Advance clock
        self.clock += dt


@define
class Simulation:
    model: RainfallRunoffModel = field(factory=RainfallRunoffModel)
    time_step: float = field(
        default=1.0,
        metadata={
            "description": "Time step for model simulation",
            "units": "days",
        },
    )
    time_stepping_method: str = field(
        default="euler",
        metadata={
            "description": "Time stepping method to use, options are 'euler' or 'scipy'"
        },
    )
    forcing_time_series: ForcingsTimeSeries = field(factory=ForcingsTimeSeries)
    start_time: pd.Timestamp = field(default=None)
    end_time: pd.Timestamp = field(default=None)
    output_vars: list = field(
        factory=lambda: ["S", "S_1", "S_2", "Q", "Q_m3s"],
        metadata={
            "description": "List of variable names to include in simulation output. "
        },
    )
    output: pd.DataFrame = field(default=None)
    scipy_solvers_kwargs: dict = field(
        factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to scipy.integrate.solve_ivp if using 'scipy' time stepping method"
        },
    )
    _initial_states: States = field(init=False)

    def __attrs_post_init__(self):
        # Initialize start_time and end_time from forcing_time_series if not provided
        if self.start_time is None:
            self.start_time = pd.Timestamp(self.forcing_time_series.time[0])
        if self.end_time is None:
            self.end_time = pd.Timestamp(self.forcing_time_series.time[-1])

        self._initial_states = States(
            S=self.model.states.S,
            S_1=self.model.states.S_1,
            S_2=self.model.states.S_2,
        )

    def run(self):
        """Run the simulation over the specified time period using the given method"""
        # Make sure model states are reset to initial values at start of simulation
        self.model.states = self._initial_states
        # Create time array for simulation outputs
        time_array = pd.date_range(
            start=self.start_time,
            end=self.end_time,
            freq=pd.Timedelta(value=self.time_step, unit="D"),
        )
        n_steps = len(time_array)
        output = {var: np.empty(n_steps) for var in self.output_vars}
        # Loop over time steps and advance model states
        for i, current_time in enumerate(time_array):
            # Update model forcings based on current time step
            # Assumes piecewise constant interpolation of forcings between time steps
            idx = np.where(self.forcing_time_series.time == current_time)[0][0]
            self.model.forcings.precipitation = self.forcing_time_series.precipitation[
                idx
            ]
            self.model.forcings.potential_evaporation = (
                self.forcing_time_series.potential_evaporation[idx]
            )
            # For first time step, calculate fluxes given the state
            # Other timesteps: done within time stepping methods after state update
            if i == 0:
                self.model.fluxes = self.model.calculate_fluxes(
                    current_time, self.model.get_state()
                )

            # Store output variables (at beginning of time interval)
            for var in self.output_vars:
                output[var][i] = self.model.get_var_by_name(var)

            # Advance model states by one time step
            if self.time_stepping_method == "euler":
                self.model.euler_step(self.time_step)
            elif self.time_stepping_method == "scipy":
                self.model.scipy_step(self.time_step, **self.scipy_solvers_kwargs)
            else:
                raise ValueError(
                    f"Invalid time stepping method '{self.time_stepping_method}'. "
                    "Options are 'euler' or 'scipy'."
                )
        self.output = pd.DataFrame(output, index=time_array)

    def save_output(self, filepath: Path):
        """Save simulation output to CSV file"""
        if self.output is not None:
            self.output.to_csv(filepath)
        else:
            raise ValueError("No simulation output to save. Run the simulation first.")