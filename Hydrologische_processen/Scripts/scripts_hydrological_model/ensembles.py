# --- modules ---
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
from joblib import Parallel, delayed
from attrs import define, field
from attrs import define, field
import rootutils
root_path = rootutils.find_root(search_from=Path.cwd(), indicator=".git")
sys.path.append(str(root_path))
from Scripts.scripts_hydrological_model.model import (
    ForcingsTimeSeries,
    RainfallRunoffModel,
    Simulation,
)

# --- functions ---
def PerturbForcingsMember(
    df_forcings:pd.DataFrame,
    P_column:str = "precipitation",
    ET_column:str = "potential_evapotranspiration",
    sigma_P: float = 0.25,
    sigma_ET: float = 0.25,
    base_seed: int = 0
    ) -> pd.DataFrame:
    """
    Function to perturb the forcing data using a lognormal distribution.
    Currently, both precipitation and evapotranspiration are being perturbed. 

    NOTE: the sigma values are the standard deviations of the underlying normal distribution, not the lognormal distribution itself.

    To go from normal to lognormal:
    If X is normally distributed with mean mu and standard deviation sigma, then Y = exp(X) is lognormally distributed. The mean of Y is given by:
        E[Y] = mu_normal = exp(mu_log + (sigma_log^2)/2) -> thus (given a desired mean of 1 for the normal distribution) mu_log = - (sigma_normal^2)/2
    
    For the lognormal distribution, CV = sqrt(exp(sigma^2) - 1), where sigma is the standard deviation of the lognormal distribution.
    Given that mu_normal=1, CV^2=SD^2/mu=SD^2. Thus SD=sigma_log=exp(sigma_normal^2) - 1 -> sigma_log=sqrt(log(1 + sigma_normal^2))

    -> thus: sigma_log=sqrt(log(1 + sigma_normal^2)) -> mu_log= - (sigma_log^2)/2

    """
    df_to_perturb=df_forcings[[P_column, ET_column]].copy()

    #--- determine the number of simulation steps ---
    n_steps=len(df_forcings)

    #--- perturb the forcings ---
    # compute sigma in lognormal distibution 
    sigma_P = np.sqrt(np.log(1 + sigma_P**2))
    sigma_ET = np.sqrt(np.log(1 + sigma_ET**2))

    #compute mu in lognormal distribution
    mu_P = -0.5 * sigma_P**2
    mu_ET = -0.5 * sigma_ET**2

    # generate random numbers for precipitation and evapotranspiration
    rng_P = np.random.default_rng(base_seed)
    rng_ET = np.random.default_rng(base_seed + 10_000)
    mfactor_P = rng_P.lognormal(mean=mu_P, sigma=sigma_P, size=n_steps)
    mfactor_ET = rng_ET.lognormal(mean=mu_ET, sigma=sigma_ET, size=n_steps)

    # perturb the forcings by multiplying the original values with the generated random numbers
    df_to_perturb[P_column] = df_to_perturb[P_column].to_numpy() * mfactor_P
    df_to_perturb[ET_column] = df_to_perturb[ET_column].to_numpy() * mfactor_ET

    return df_to_perturb

def RunMemberOL(
    model: RainfallRunoffModel,
    forcings: pd.DataFrame,
    date_ini:pd.Timestamp = pd.Timestamp("2010-01-01"),
    date_end:pd.Timestamp = pd.Timestamp("2021-12-31"),
    member_id: int = 0,
    saved_variables: list = ["S", "S_1", "S_2", "Q_m3s", "Q_1", "Q_2"], #HARD CODED
    n_warmup_days:int = 50
    ) -> pd.DataFrame:
    """
    Function to run a single member of the RainfallRunoffModel with given forcings

    Currently, the function is hardcoded to only output the discharge and renames 
    the output column to include the member_id.
    """
    #--- error handling ---
    if not isinstance(forcings.index, pd.DatetimeIndex):
        raise ValueError("The index of forcings must be a DatetimeIndex!")
    if not isinstance(model, RainfallRunoffModel):
        raise ValueError("The model must be an instance of RainfallRunoffModel!")
    
    #--- set up the forcing time series for the member ---
    forcing_timeseries = ForcingsTimeSeries(
        precipitation=forcings[f"precipitation_{member_id}"].values,
        potential_evaporation=forcings[f"potential_evapotranspiration_{member_id}"].values,
        time=forcings.index.values,
        )

    #--- setup the simulation for the member ---
    # change the start date to account for the warmup period
    date_ini_wrmup = date_ini-pd.Timedelta(days=n_warmup_days)
    date_ini_wrmup = forcings.index[0] if date_ini_wrmup < forcings.index[0] else date_ini_wrmup

    # generate the simulation object and run the simulation
    sim=Simulation(
        model=model,
        forcing_time_series=forcing_timeseries,
        output_vars=saved_variables,
        start_time=date_ini_wrmup,
        end_time=date_end
    )
    sim.run()

    # --- return the simulated discharge together with the measured discharge ---
    output=sim.output.copy()
    output=output.loc[(output.index >= date_ini) & (output.index <= date_end)] # clip the output to the desired time period (no warm up days)
    output=output[["Q_m3s", "S"]].rename(columns={"Q_m3s": f"Q_m3s_{member_id}", 
                                                      "S": f"S_{member_id}"})
    del forcings, forcing_timeseries, sim

    return output

@define
class Perturbations():
    """Class to handle the generation of perturbed forcings and running open loop simulations for each member."""
    df_forcings: pd.DataFrame = field(init=True, 
                                      default=None, 
                                      metadata={"description": "DataFrame containing the original forcings to be perturbed", 
                                                "units": "-"
                                    })
    sigma_P: float = field(default=0.25,
                          metadata={"description": "standard deviation used for the lognormal distribution to perturb precipitation", 
                                    "units": "-"
                                    })
    sigma_ET: float = field(default=0.25,
                           metadata={"description": "standard deviation used for the lognormal distribution to perturb evapotranspiration", 
                                     "units": "-"
                                     })
    n_perturbations: int = field(default=30, 
                                metadata={"description": "number of perturbed forcings to generate", 
                                          "units": "-"
                                          })
    seed: int = field(default=42,
                      metadata={"description": "seed for the random number generator", 
                                  "units": "-"
                                  })
    date_ini: pd.Timestamp = field(default=pd.Timestamp("2010-01-01"),
                                   metadata={"description": "start date for the simulation",
                                                "units": "-"
                                                })
    date_end: pd.Timestamp = field(default=pd.Timestamp("2021-12-31"),
                                    metadata={"description": "end date for the simulation",
                                                "units": "-"
                                                })
    n_warmup_days: int = field(default=50,
                               metadata={"description": "number of warmup days for the simulation",
                                             "units": "-"
                                             })
    model: RainfallRunoffModel = field(factory=RainfallRunoffModel)

    #instantiate some dataframes
    perturbed_forcings: pd.DataFrame = field(init=False)
    open_loop_results: pd.DataFrame = field(init=False)

    def GeneratePerturbedForcings(self):
        """
        Function to generate multiple perturbed forcings using the PerturbForcingsMember function.
        The function uses parallel processing to speed up the generation of perturbed forcings.

        Note that the mean for the lognormal distributions are set to their default (=1)
        """
        # generate the base seed for the members
        np.random.seed(self.seed) #set the seed for reproducibility
        base_seed = np.random.randint(0, np.max([1000, self.n_perturbations]), 
                                    size=self.n_perturbations)

        # perturb the forcings -> daily perturbations
        perturbations = pd.concat([PerturbForcingsMember(
                                                    df_forcings=self.df_forcings, 
                                                    sigma_P=self.sigma_P, 
                                                    sigma_ET=self.sigma_ET, 
                                                    base_seed=s).add_suffix(f"_{i}")
                for i, s in enumerate(base_seed)], 
                axis=1)
            
        perturbations = perturbations.reindex(sorted(perturbations.columns), 
                                              axis=1
                                              )
        self.perturbed_forcings = perturbations #HACK: store the perturbed forcings in the class attribute for later use

        return perturbations


    def open_loop(self):
        """
        Function to run the open loop simulation for each perturbed forcing member.
        """
        #--- error handling ---
        if not hasattr(self, 'perturbed_forcings'):
            print("Perturbed forcings have not been generated yet. Generating them...")
            self.GeneratePerturbedForcings()

        #---run the open loop simulation for each member---
        n_jobs=12 if os.cpu_count() > 12 else os.cpu_count() #HACK: use 12 cores if available, otherwise use all available cores
        
        open_loop=Parallel(n_jobs=n_jobs)(
            delayed(RunMemberOL)(
                model=self.model,
                forcings=self.perturbed_forcings.loc[:, [c for c in self.perturbed_forcings.columns 
                                                         if c.split("_")[-1]==str(member)]],
                member_id=member,
                date_ini=self.date_ini,
                date_end=self.date_end,
                n_warmup_days=self.n_warmup_days
            )
            for member in range(self.n_perturbations)
        )
        open_loop = pd.concat(open_loop, axis=1)

        #---compute statistics---
        # statistics for the discharge
        OLQ = open_loop[[c for c in open_loop.columns if c.startswith("Q_m3s_")]].to_numpy()
        OLQ = pd.DataFrame({
                    "Qmean": OLQ.mean(axis=1),
                    "Qstd": OLQ.std(axis=1, ddof=1),  # matches pandas' default
                    "Qmin": OLQ.min(axis=1),
                    "Qmax": OLQ.max(axis=1),
                }, index=open_loop.index)
        
        # statistics for the storage
        OLS = open_loop[[c for c in open_loop.columns if c.startswith("S_")]].to_numpy()
        OLS = pd.DataFrame({
            "Smin": OLS.min(axis=1),
            "Smean": OLS.mean(axis=1),
            "Sstd": OLS.std(axis=1, ddof=1),
            "Smax": OLS.max(axis=1)
        }, index=open_loop.index)


        open_loop=pd.concat([OLQ, OLS], axis=1)

        # --- add the measured discharge to the open loop results ---
        Qobs = self.df_forcings["river_discharge"].rename("Qobs")
        Qobs = Qobs.loc[(Qobs.index >= self.date_ini) & (Qobs.index <= self.date_end)]
        open_loop = pd.merge(open_loop, Qobs, left_index=True, right_index=True)

        self.open_loop_results = open_loop #HACK: store the open loop results in the class attribute for later use


    def PlotPerturbedForcings(self,
                              date_ini: pd.Timestamp = None,
                              date_end: pd.Timestamp = None):
        """
        Function to plot the perturbed forcings.
        """
        if not hasattr(self, 'perturbed_forcings'):
            self.GeneratePerturbedForcings()

        if date_ini is None:
            date_ini = self.date_ini
        if date_end is None:
            date_end = self.date_end
        
        # --- generate the dataframe to plot ---
        df_plot=[]
        for k,v in {"potential": "ET", "precip": "P"}.items():
            cols = [c for c in self.perturbed_forcings.columns if k in c.lower()]
            arr = self.perturbed_forcings[cols].to_numpy()

            df_tmp = pd.DataFrame({
                f"{v}_mean": arr.mean(axis=1),
                f"{v}_std": arr.std(axis=1, ddof=1),  # matches pandas' default
            }, index=self.perturbed_forcings.index)
            df_plot.append(df_tmp)
        df_plot = pd.concat(df_plot, axis=1)

        # --- plot the figures ---
        # select the time period to plot
        df_plot = df_plot.loc[(df_plot.index >= date_ini) & 
                            (df_plot.index <= date_end)
                            ]

        # plot P
        fig,ax=plt.subplots(figsize=(12, 5))
        ax.plot(df_plot.index, df_plot["P_mean"], color="tab:purple", lw=1.4, label="Mean P")
        ax.fill_between(df_plot.index, df_plot["P_mean"]-df_plot["P_std"], df_plot["P_mean"]+df_plot["P_std"], alpha=0.5, color="tab:purple", label="Mean P")
        ax.tick_params(axis="y", labelcolor="tab:purple", labelsize=15)
        ax.set_ylabel("P mean (mm)", color="tab:purple", fontsize=18)
        ax.tick_params(axis="x", labelsize=15)
        ax.set_xlabel("Date [-]", fontsize=18)

        ax2=ax.twinx()
        ax2.invert_yaxis()
        ax2.plot(df_plot.index, df_plot["P_std"], color="tab:orange", lw=1.4, label="std P", alpha=0.4)
        ax2.set_ylabel("P std (mm)", color="tab:orange", fontsize=18)
        ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=15)

        # plot ET
        fig,ax=plt.subplots(figsize=(12, 5))
        ax.plot(df_plot.index, df_plot["ET_mean"], color="tab:purple", lw=1.4, label="Mean ET")
        ax.fill_between(df_plot.index, df_plot["ET_mean"]-df_plot["ET_std"], df_plot["ET_mean"]+df_plot["ET_std"], alpha=0.3, color="tab:purple", label="Mean ET")
        ax.tick_params(axis="y", labelcolor="tab:purple", labelsize=15)
        ax.set_ylabel("ET mean (mm)", color="tab:purple", fontsize=18)
        ax.tick_params(axis="x", labelsize=15)
        ax.set_xlabel("Date [-]", fontsize=18)

        ax2=ax.twinx()
        ax2.invert_yaxis()
        ax2.plot(df_plot.index, df_plot["ET_std"], color="tab:orange", lw=1.4, label="std ET", alpha=0.4)
        ax2.set_ylabel("ET std (mm)", color="tab:orange", fontsize=18)
        ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=15)