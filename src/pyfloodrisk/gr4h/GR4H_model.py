# -*- coding: utf-8 -*-
"""
GR4H

Rain-Runoff Model

Based on:
Saul Arciniega Esparza
Hydrogeology Group, Faculty of Engineering,
National Autonomous University of Mexico
zaul.ae@gmail.com | sarciniegae@comunidad.unam.mx
https://gitlab.com/Zaul_AE/lumod
&
Andrew MacDonald (andrew@maccas.net)
https://github.com/amacd31/pygr4j

Reference:
Perrin, C. (2002). Vers une amélioration d'un modèle global pluie-débit au
travers d'une approche comparative. La Houille Blanche, n°6/7, 84-91.
Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a
parsimonious model for streamflow simulation. Journal of Hydrology
279(1-4), 275-289.

Author:
Caleb Dykman

Update to GR4H
 - Unit hydrograph exponent updated to 1.25 from 2.5
 - percolation coefficient updated to 21/4 from 9/4

Reference:
Mathevet, T. (2005). Quels modèles pluie-débit globaux pour le pas de temps horaire ?
Développement empirique et comparaison de modèles sur un large échantillon de bassins versants,
PhD thesis (in French),
ENGREF - Cemagref Antony, Paris, France, 463 pp.
Le Moine, N. (2008). Le bassin versant de surface vu par le souterrain : une
voie d'amélioration des performances et du réalisme des modèles pluie-débit ?
PhD thesis (in French),
UPMC - Cemagref Antony, Paris, France.
"""

import math
import warnings
from math import tanh

import numba as nb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class BaseModel:
    """Minimal base class providing dict-style parameter storage."""

    def __init__(self, area=100):
        self.area = area
        self.params = {}

    def get_parameters(self, asdict=True):
        """Return the model parameters as dict or Pandas Series."""
        if asdict:
            return self.params.copy()
        else:
            return pd.Series(self.params)

    def set_parameters(self, params=None, **kwargs):
        """Set model parameters as dict {"x": 1}, Series, or keyword (x=1)."""
        if isinstance(params, dict):
            params = kwargs.update(params)
        elif isinstance(params, pd.Series):
            params = kwargs.update(params.to_dict())

        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value


# ==============================================================================
# Main class
# ==============================================================================

class GR4H(BaseModel):
    """GR4H hourly rainfall-runoff model.

    A production/routing reservoir model in the GR (modèle du Génie Rural)
    family, adapted from the daily GR4J for hourly simulation: the unit
    hydrograph exponent is relaxed from 2.5 to 1.25 and the percolation
    coefficient from 9/4 to 21/4 (see module docstring for references).
    """

    def __init__(self, area=100, params=None):
        """
        GR4H hydrological model for hourly simulation

        Inputs:
            area    >     [float] catchment area in km2
            params  >     [dict] model parameters

        Model Parameters
            ps0     >     [float] initial production storage as a fraction ps0=(ps/X1)
            rs0     >     [float] initial routing storage as a fraction pr0=(rs/X3)
            x1      >     [float] maximum production capacity (mm)
            x2      >     [float] discharge parameter (mm)
            x3      >     [float] routing maximum capacity (mm)
            x4      >     [float] HU1 unit hydrograph time base (hours)

        Reference:
        Perrin, C. (2002). Vers une amélioration d'un modèle global pluie-débit
        au travers d'une approche comparative. La Houille Blanche, n°6/7, 84-91.
        Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a
        parsimonious model for streamflow simulation. Journal of Hydrology
        279(1-4), 275-289.
        """
        super().__init__(area)

        self.params = {
            "ps0": 1.0,
            "rs0": 0.5,
            "x1": 500.0,
            "x2": 3.0,
            "x3": 200.0,
            "x4": 5.0,
        }

        if params is not None:
            self.set_parameters(**params)

    def __repr__(self):
        text = "\n______________GR4J structure______________\n"
        text += "Catchment properties:\n"
        text += "    Area (km2): {:.3f}\n".format(self.area)
        text += "Model Parameters:\n"
        text += "    x1  > Maximum production capacity (mm)     : {:.3f}\n".format(
            self.params["x1"]
        )
        text += "    x2  > Discharge parameter (mm)             : {:.3f}\n".format(
            self.params["x2"]
        )
        text += "    x3  > Routing maximum capacity (mm)        : {:.3f}\n".format(
            self.params["x3"]
        )
        text += "    x4  > Delay (hours)                        : {:.3f}\n".format(
            self.params["x4"]
        )
        text += "    ps0 > Initial production storage (ps/x1)   : {:.3f}\n".format(
            self.params["ps0"]
        )
        text += "    rs0 > Initial routing storage (rs/x3)      : {:.3f}\n".format(
            self.params["rs0"]
        )
        return text

    def __str__(self):
        x1 = self.params["x1"]
        x2 = self.params["x2"]
        x3 = self.params["x3"]
        x4 = self.params["x4"]
        ps0 = self.params["ps0"]
        rs0 = self.params["rs0"]
        text = f"GR4J(area={self.area:.2f},"
        text += f"x1={x1:.3f},x2={x2:.3f},x3={x3:.3f},x4={x4:.3f},"
        text += f"ps0={ps0:.3f},rs0={rs0:.3f})"
        return text

    def run(self, forcings, save_state=False, **kwargs):
        """
        Run the GR4H model

        Parameters
        ----------
        forcings : DataFrame
            Input data with columns prec (precipitation), tmean, and
            pet(potential evapotranspiration, optional)
        save_state : bool, optional
            If True (default), last storage is saved as w0 parameter
        **kwargs :
            Model parameters can be changed for the simulation
                area    >     [float] catchment area in km2
                ps0     >     [float] initial production storage as a fraction ps0=(ps/X1)
                rs0     >     [float] initial routing storage as a fraction pr0=(rs/X3)
                x1      >     [float] maximum production capacity (mm)
                x2      >     [float] discharge parameter (mm)
                x3      >     [float] routing maximum capacity (mm)
                x4      >     [float] HU1 unit hydrograph time base (hours)

        Returns
        -------
        Simulations : DataFrame
            qt       > Streamflow (qd+qb) at catchment output (m3/s)
            qd       > Directflow at catchment output (m3/s)
            qb       > Baseflow at catchment output (m3/s)
            pet      > Potential Evapotranspiration (mm)
            gwe      > Groundwater Exchange (mm)
            ps       > Production storage as a fraction of x1 (-)
            rs       > Routing storage as a fraction of x3 (-)
        """
        # Load new parameters
        if kwargs:
            self.area = kwargs.get("area", self.area)
            self.set_parameters(**kwargs)

        # Get Forcings
        prec = forcings["prec"].values
        pet = forcings["pet"].values

        simulations = _gr4h(
            prec,
            pet,
            self.params["x1"],
            self.params["x2"],
            self.params["x3"],
            self.params["x4"],
            self.params["ps0"],
            self.params["rs0"],
        )

        # Create Output DataFrame
        outputs = pd.DataFrame(
            {
                "qt": simulations[0],
                "qt_mm": simulations[0],
                "qd": simulations[1],
                "qb": simulations[2],
                "pet": pet,
                "prec": prec,
                "gwe": simulations[3],
                "ps": simulations[4],
                "rs": simulations[5],
            },
            index=forcings.index,
        )

        # Convert units mm/hr to m3/s
        outputs.loc[:, "qt"] = outputs.loc[:, "qt"] * self.area / 3.6
        outputs.loc[:, "qd"] = outputs.loc[:, "qd"] * self.area / 3.6
        outputs.loc[:, "qb"] = outputs.loc[:, "qb"] * self.area / 3.6

        # Save final storage state
        if save_state:
            psto = simulations[4][-1]
            rsto = simulations[5][-1]
            self.params["ps0"] = psto
            self.params["rs0"] = rsto

        return outputs


# ==============================================================================
# Subroutines for model processes
# ==============================================================================

@nb.jit(nopython=True)
def _reservoirs_evaporation(prec, pet, ps, x1):
    """Estimate net evapotranspiration and reservoir production."""
    if prec > pet:
        evap = 0.0
        snp = (prec - pet) / x1  # scaled net precipitation
        snp = min(snp, 13.0)
        tsnp = tanh(snp)  # tanh_scaled_net_precip
        # reservoir production
        res_prod = ((x1 * (1.0 - (ps / x1) ** 2.0) * tsnp)
                    / (1.0 + ps / x1 * tsnp))
        # routing pattern
        rout_pat = prec - pet - res_prod
    else:
        sne = (pet - prec) / x1  # scaled net evapotranspiration
        sne = min(sne, 13.0)
        tsne = tanh(sne)  # tanh_scaled_net_evap
        ps_div_x1 = (2.0 - ps / x1) * tsne
        evap = ps * ps_div_x1 / (1.0 + (1.0 - ps / x1) * tsne)

        res_prod = 0  # reservoir_production
        rout_pat = 0  # routing_pattern

    return evap, res_prod, rout_pat


@nb.jit(nopython=True)
def _s_curves1(t, x4):
    """Unit hydrograph ordinates for UH1 derived from S-curves."""
    if t <= 0:
        return 0
    elif t < x4:
        return (t / x4) ** 1.25  # 1.25 instead of 2.5 for GR4H
    else:  # t >= x4
        return 1


@nb.jit(nopython=True)
def _s_curves2(t, x4):
    """Unit hydrograph ordinates for UH2 derived from S-curves."""
    if t <= 0:
        return 0
    elif t < x4:
        return 0.5 * (t / x4) ** 1.25  # 1.25 instead of 2.5 for GR4H
    elif t < 2 * x4:
        return 1 - 0.5 * (2 - t / x4) ** 1.25  # 1.25 instead of 2.5 for GR4H
    else:  # t >= x4
        return 1


@nb.jit(nopython=True)
def _compute_unitary_hydrograph(x4):
    """Build the UH1/UH2 unit hydrograph ordinates and working buffers."""
    nuh1 = int(math.ceil(x4))
    nuh2 = int(math.ceil(2.0 * x4))
    uh1 = np.zeros(nuh1)
    uh2 = np.zeros(nuh2)
    uh1_ordinates = np.zeros(nuh1)
    uh2_ordinates = np.zeros(nuh2)

    for t in range(1, nuh1 + 1):
        uh1_ordinates[t - 1] = _s_curves1(t, x4) - _s_curves1(t - 1, x4)

    for t in range(1, nuh2 + 1):
        uh2_ordinates[t - 1] = _s_curves2(t, x4) - _s_curves2(t - 1, x4)

    ouh1 = uh1_ordinates
    ouh2 = uh2_ordinates
    return ouh1, ouh2, uh1, uh2


@nb.jit(nopython=True)
def _compute_hydrograph(rout_pat, ouh1, ouh2, uh1, uh2):
    """Convolve routed production with the UH1/UH2 unit hydrographs."""
    for i in range(0, len(uh1) - 1):
        uh1[i] = uh1[i + 1] + ouh1[i] * rout_pat
    uh1[-1] = ouh1[-1] * rout_pat

    for j in range(0, len(uh2) - 1):
        uh2[j] = uh2[j + 1] + ouh2[j] * rout_pat
    uh2[-1] = ouh2[-1] * rout_pat

    return uh1, uh2


@nb.jit(nopython=True)
def _compute_exchange(uh1, rout_sto, x2, x3):
    """Apply groundwater exchange and update the routing store."""
    # groundwater exchange
    cr = 1
    gw_exc = x2 * ((1 / cr) * (rout_sto / x3)) ** 3.5
    rout_sto = max(0, rout_sto + uh1[0] * 0.9 + gw_exc)
    return gw_exc, rout_sto


@nb.jit(nopython=True)
def _compute_discharge(uh2, gw_exc, rout_sto, x3):
    """Compute the routing-store outflow and direct-flow discharge."""
    new_rout_sto = rout_sto / (1.0 + (rout_sto / x3) ** 4.0) ** 0.25
    qr = rout_sto - new_rout_sto
    rout_sto = new_rout_sto
    qd = max(0, uh2[0] * 0.1 + gw_exc)
    return qr, qd, rout_sto


@nb.jit(nopython=True)
def _gr4h(prec, pet, x1, x2, x3, x4, ps0, rs0):
    """Run the GR4H production/routing scheme over a forcing time series.

    Parameters
    ----------
    prec, pet : ndarray
        Precipitation and potential evapotranspiration (mm/hr).
    x1, x2, x3, x4 : float
        GR4H model parameters (see :class:`GR4H`).
    ps0, rs0 : float
        Initial production/routing storage fractions.

    Returns
    -------
    Tuple of arrays (qt, qd, qb, gw_exchange, ps, rs), each the same
    length as prec/pet.
    """
    # Create empty arrays
    n = len(prec)
    qtarray = np.zeros(n, dtype=np.float32)
    qdarray = np.zeros(n, dtype=np.float32)
    qrarray = np.zeros(n, dtype=np.float32)
    gwarray = np.zeros(n, dtype=np.float32)
    psarray = np.zeros(n, dtype=np.float32)
    rsarray = np.zeros(n, dtype=np.float32)

    # Initial parameters
    ouh1, ouh2, uh1, uh2 = _compute_unitary_hydrograph(x4)
    psto = ps0 * x1
    rsto = rs0 * x3

    # Compute water partioning
    for t in range(n):
        evap, res_prod, rout_pat = _reservoirs_evaporation(
            prec[t], pet[t], psto, x1
        )
        psto = psto - evap + res_prod
        # 21/4 (5.25) instead of 9/4 (2.25) for GR4H
        perc = psto * (1 - (1.0 + (psto / 5.25 / x1) ** 4.0) ** -0.25)
        rout_pat = perc + rout_pat
        psto = psto - perc

        uh1, uh2 = _compute_hydrograph(rout_pat, ouh1, ouh2, uh1, uh2)

        gw_exc, rsto = _compute_exchange(uh1, rsto, x2, x3)

        qr, qd, rsto = _compute_discharge(uh2, gw_exc, rsto, x3)
        qt = qr + qd

        # Save outputs
        qtarray[t] = qt  # total flow
        qdarray[t] = qd  # runoff
        qrarray[t] = qr  # baseflow
        gwarray[t] = gw_exc  # groundwater exchange
        psarray[t] = psto / x1  # production storage
        rsarray[t] = rsto / x3  # routing storage

    return qtarray, qdarray, qrarray, gwarray, psarray, rsarray