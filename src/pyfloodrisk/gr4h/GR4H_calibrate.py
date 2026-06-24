"""Link the GR4H model to SPOTPY for DREAM-based calibration."""

import pandas as pd
from spotpy.objectivefunctions import nashsutcliffe, rmse
from spotpy.parameter import Uniform

from .GR4H_model import GR4H


class spot_setup:
    """SPOTPY setup class wrapping GR4H for calibration.

    Defines the calibrated parameter priors as class attributes (the
    convention SPOTPY uses to auto-discover them) and implements the
    ``simulation``/``evaluation``/``objectivefunction`` API SPOTPY calls
    during sampling.
    """

    # Initialise parameter priors
    x1 = Uniform(low=1.0, high=1500.0)  # Maximum production capacity (mm)
    x2 = Uniform(low=-10.0, high=10.0)  # Water exchange coefficient (mm);
    # positive if gaining, negative if losing, or null
    x3 = Uniform(low=1.0, high=1000.0)  # Routing maximum capacity (mm)
    x4 = Uniform(low=0.5, high=500)  # Unit hydrograph time base (hours)

    def __init__(
        self,
        climatefile,
        area,
        obj_func=None,
        ps0=1.0,
        rs0=0.5,
        warmup=365,
        events=None,
        obj_freq=None,
    ):
        """
        Parameters
        ----------
        climatefile : str or Path
            Path to a climate CSV with ``prec``, ``pet``, and ``qt`` columns.
        area : float
            Catchment area in km2.
        obj_func : callable or str, optional
            Objective function. One of ``None`` (RMSE, negated), ``"nnse"``,
            ``"-nnse"``, or a callable ``f(evaluation, simulation)``.
        ps0, rs0 : float, optional
            Initial production/routing storage fractions.
        warmup : int, optional
            Warm-up period in days excluded from the objective function.
        events : array_like, optional
            Indices to restrict the objective function to specific events.
            If None, all data after the warm-up period is used.
        obj_freq : str, optional
            If ``"D"``, resample simulation/evaluation to daily totals
            before scoring.
        """
        # Load observation data from file
        climatedata = pd.read_csv(
            climatefile, index_col=0, parse_dates=True, dayfirst=True
        )
        self.forcings = climatedata[["prec", "pet"]]
        self.trueObs = climatedata["qt"]
        # Catchment size
        self.area = area  # km2
        # Transformation factor
        self.Factor = self.area / 3.6  # Convert units mm/hr to m3/s
        # Objective function
        self.obj_freq = obj_freq
        self.obj_func = obj_func
        # Warm-up period in days
        self.warmup = warmup
        # Calibration events
        self.events = events
        # Initial conditions
        self.ps0 = ps0
        self.rs0 = rs0

    def simulation(self, x):
        """Run GR4H for one SPOTPY-sampled parameter vector x = [x1, x2, x3, x4]."""
        model = GR4H(
            area=self.area,
            params={
                "ps0": self.ps0,
                "rs0": self.rs0,
                "x1": x[0],
                "x2": x[1],
                "x3": x[2],
                "x4": x[3],
            },
        )
        data = model.run(self.forcings)
        if self.events is None:
            return data.qt.iloc[self.warmup * 24:].values
        return data.qt.iloc[self.events].values

    def evaluation(self):
        """Return the observed streamflow matching `simulation`'s window."""
        if self.events is None:
            return self.trueObs.iloc[self.warmup * 24:].values
        return self.trueObs.iloc[self.events].values

    def objectivefunction(self, simulation, evaluation, params=None):
        """Score a simulation against observations (SPOTPY's required hook)."""
        # SPOTPY expects to get one or multiple values back, that define
        # the performance of the model run
        if self.obj_freq == "D":
            # simulation/evaluation are plain arrays (see simulation/evaluation
            # above), so the DatetimeIndex needed for resampling is reattached
            # here
            if self.events is None:
                idx = self.trueObs.iloc[self.warmup * 24:].index
            else:
                idx = self.trueObs.iloc[self.events].index
            simulation = pd.Series(simulation, index=idx).resample("D").sum().iloc[1:-1]
            evaluation = pd.Series(evaluation, index=idx).resample("D").sum().iloc[1:-1]

        if self.obj_func is None:
            # This is used if not overwritten by user
            like = -rmse(evaluation, simulation)
        elif self.obj_func == "nnse":
            like = 1 / (2 - nashsutcliffe(evaluation, simulation))
        elif self.obj_func == "-nnse":
            like = -1 / (2 - nashsutcliffe(evaluation, simulation))
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like