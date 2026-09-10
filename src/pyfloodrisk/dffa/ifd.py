"""
Design rainfall frequency curve (IFD) and areal reduction.

The Monte Carlo framework needs a *continuous* rainfall frequency curve, whereas
ARR/BoM IFD data are tabulated at a handful of AEPs.  ``IFDCurve`` fits a
monotone curve through the tabulated points in the (standard normal variate,
log depth) domain -- i.e. a log-normal-like curve that is allowed local
curvature -- and extrapolates log-linearly in the normal variate beyond the
rarest tabulated point.  That is the same transformation the ARR joint
probability framework uses to stratify the rainfall domain, so the curve and the
stratification live in the same coordinate system.

Notes / things to check against your own data
---------------------------------------------
* Depths are *burst* depths (as IFD data are), not complete storms.  Everything
  downstream inherits that limitation (see README).
* If you have ARR "rare" (1 in 200 to 1 in 2000) and "very frequent" design
  rainfalls, add them as extra columns of the table rather than relying on the
  extrapolation.
* The ARF implementation reproduces the *functional form* of the ARR 2019
  long-duration equation, but no coefficients are bundled -- you must supply the
  nine region-specific coefficients from ARR Book 2, Chapter 4.  Verify both the
  form and the coefficients before using it in anger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

__all__ = ["IFDCurve", "unit_arf", "ARR2019LongDurationARF", "ifd_table_from_csv"]

#: AEPs the bundled demo tables are tabulated at, and a reasonable default for
#: :meth:`IFDCurve.as_frame`.
DEFAULT_IFD_AEPS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def _as_array(x):
    return np.atleast_1d(np.asarray(x, dtype=float))


class IFDCurve:
    """Continuous, invertible design rainfall frequency curve.

    Parameters
    ----------
    table
        ``DataFrame`` whose index is storm duration in **hours** and whose
        columns are AEPs expressed as annual exceedance probabilities
        (e.g. ``0.5, 0.2, 0.1, 0.05, 0.02, 0.01``).  Values are point burst
        **depths in mm**.  Columns need not be sorted.
    arf
        Callable ``arf(area_km2, duration_h, aep) -> float`` giving the areal
        reduction factor.  Defaults to 1.0 (point rainfall).
    duration_interp
        ``"pchip"`` (default) or ``"linear"`` interpolation of log depth against
        log duration, used when a requested duration is not tabulated.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        arf: Callable[[float, float, float], float] | None = None,
        duration_interp: str = "pchip",
    ):
        table = table.copy()
        table.columns = [float(c) for c in table.columns]
        table = table.reindex(sorted(table.columns), axis=1)  # ascending AEP
        table = table.sort_index()
        if (table.values <= 0).any():
            raise ValueError("IFD depths must be positive")
        # depths must decrease as AEP increases
        if not np.all(np.diff(table.values, axis=1) <= 1e-9):
            raise ValueError("IFD depths must be non-increasing with increasing AEP")

        self.table = table
        self.durations = table.index.to_numpy(dtype=float)
        self.aeps = np.asarray(table.columns, dtype=float)
        self.arf = arf if arf is not None else unit_arf
        self.duration_interp = duration_interp

        # z = standard normal variate of non-exceedance probability
        self._z = norm.ppf(1.0 - self.aeps)          # ascending AEP -> descending z
        order = np.argsort(self._z)
        self._z = self._z[order]
        self._logd = np.log(table.values[:, order])   # (n_dur, n_aep) ascending z

        # per-duration monotone fit of log depth against z, with linear tails
        self._fits = [PchipInterpolator(self._z, self._logd[i], extrapolate=False)
                      for i in range(len(self.durations))]
        # tail slopes (log mm per unit z) from the two outermost knots
        self._slope_hi = (self._logd[:, -1] - self._logd[:, -2]) / (self._z[-1] - self._z[-2])
        self._slope_lo = (self._logd[:, 1] - self._logd[:, 0]) / (self._z[1] - self._z[0])

    # ------------------------------------------------------------------ core
    def _logdepth_at_tabulated_durations(self, z: np.ndarray) -> np.ndarray:
        """(n_dur, n_z) log point depth."""
        out = np.empty((len(self.durations), z.size))
        for i, fit in enumerate(self._fits):
            y = fit(z)
            hi = z > self._z[-1]
            lo = z < self._z[0]
            y[hi] = self._logd[i, -1] + self._slope_hi[i] * (z[hi] - self._z[-1])
            y[lo] = self._logd[i, 0] + self._slope_lo[i] * (z[lo] - self._z[0])
            out[i] = y
        return out

    def point_depth(self, duration_h, aep) -> np.ndarray:
        """Point burst depth (mm) for scalar/array duration and AEP."""
        aep = _as_array(aep)
        duration_h = _as_array(duration_h)
        z = norm.ppf(1.0 - aep)
        logd = self._logdepth_at_tabulated_durations(z)      # (n_dur, n_z)

        ld_req = np.log(duration_h)
        ld_tab = np.log(self.durations)
        if duration_h.size == 1 and np.any(np.isclose(ld_tab, ld_req[0])):
            i = int(np.argmin(np.abs(ld_tab - ld_req[0])))
            return np.squeeze(np.exp(logd[i]))

        if self.duration_interp == "pchip" and len(ld_tab) >= 3:
            out = np.array([PchipInterpolator(ld_tab, logd[:, j], extrapolate=True)(ld_req)
                            for j in range(z.size)]).T
        else:
            out = np.array([np.interp(ld_req, ld_tab, logd[:, j]) for j in range(z.size)]).T
        return np.exp(np.squeeze(out))

    def depth(self, duration_h: float, aep, area_km2: float | None = None) -> np.ndarray:
        """Catchment-average burst depth (mm) = point depth x ARF."""
        aep = _as_array(aep)
        d_point = _as_array(self.point_depth(duration_h, aep))
        if area_km2 is None:
            return np.squeeze(d_point)
        f = np.array([float(self.arf(area_km2, duration_h, a)) for a in aep])
        return np.squeeze(d_point * f)

    def aep_of_depth(self, duration_h: float, depth_mm) -> np.ndarray:
        """Inverse: AEP of a given *point* burst depth."""
        depth_mm = _as_array(depth_mm)
        # dense monotone lookup in z
        zg = np.linspace(-3.0, 6.0, 2001)
        dg = _as_array(self.point_depth(duration_h, 1.0 - norm.cdf(zg)))
        z = np.interp(np.log(depth_mm), np.log(dg), zg)
        return np.squeeze(1.0 - norm.cdf(z))

    # --------------------------------------------------------------- helpers
    def as_frame(self, aeps: Sequence[float], area_km2: float | None = None) -> pd.DataFrame:
        """Tabulate the fitted curve (handy for checking against the source IFD)."""
        rows = {}
        for d in self.durations:
            rows[d] = _as_array(self.depth(d, aeps, area_km2))
        return pd.DataFrame(rows, index=list(aeps)).T.rename_axis("duration_h")


# ------------------------------------------------------------------- loaders
def ifd_table_from_csv(path, duration_col: str = "duration_h",
                       duration_units: str = "hours") -> pd.DataFrame:
    """Read a tidy design rainfall table into the layout :class:`IFDCurve` wants.

    The expected file is one row per duration and one column per AEP::

        duration_h,0.5,0.2,0.1,0.05,0.02,0.01
        1,22.1,30.5,36.0,41.4,48.9,54.8
        ...

    AEP column headers may be fractions (``0.01``), percentages (``1%``) or
    average recurrence intervals (``1 in 100``, ``100y``).  Depths are mm.
    Lines beginning ``#`` are ignored, so a table can carry its provenance
    at the top of the file (the bundled demo tables do).

    This is the only way design rainfalls enter the framework.  A BoM IFD
    download is *not* in the layout above -- it carries a metadata preamble
    and its own column naming, and the layout has changed between releases
    -- so extract the depths you want into a tidy table like the above
    rather than pointing this at the download unedited.

    Parameters
    ----------
    path :
        CSV file path.
    duration_col :
        Name of the duration column.
    duration_units :
        ``"hours"`` (default) or ``"minutes"``.

    Returns
    -------
    DataFrame indexed by duration in hours, columns AEP as a fraction.
    """
    raw = pd.read_csv(path, comment="#", skip_blank_lines=True)
    cols = {str(c).strip().lower(): c for c in raw.columns}
    key = duration_col.strip().lower()
    if key not in cols:
        raise ValueError(
            f"no duration column '{duration_col}'; got {list(raw.columns)}")
    dur = pd.to_numeric(raw[cols[key]], errors="coerce").to_numpy(float)
    if duration_units.startswith("min"):
        dur = dur / 60.0
    elif not duration_units.startswith("hour"):
        raise ValueError("duration_units must be 'hours' or 'minutes'")

    depth_cols = [c for c in raw.columns if c != cols[key]]
    aeps = [_parse_aep(c) for c in depth_cols]
    table = pd.DataFrame(
        {a: pd.to_numeric(raw[c], errors="coerce").to_numpy(float)
         for a, c in zip(aeps, depth_cols)},
        index=pd.Index(dur, name="duration_h"))
    return table.sort_index()


def _parse_aep(label) -> float:
    """AEP as a fraction from a column header."""
    s = str(label).strip().lower().replace(" ", "")
    if s.startswith("1in"):                       # 1 in 100
        return 1.0 / float(s[3:])
    if s.endswith("y") or s.endswith("ari"):      # 100y, 100ari
        return 1.0 / float(s.rstrip("ari").rstrip("y"))
    if s.endswith("%"):                           # 1%
        return float(s[:-1]) / 100.0
    value = float(s)
    if value > 1.0:                               # bare percentage
        return value / 100.0
    return value


# --------------------------------------------------------------------- ARF
def unit_arf(area_km2: float, duration_h: float, aep: float) -> float:
    """No areal reduction (point rainfall)."""
    return 1.0


@dataclass
class ARR2019LongDurationARF:
    """ARR 2019 long-duration areal reduction factor.

    Functional form (ARR Book 2, Chapter 4)::

        ARF = min[1,
                  1 - a*(A**b - c*log10(D)) * D**(-d)
                    + e * A**f * D**g * (0.3 + log10(AEP))
                    + h * 10**(i*A*D/1440) * (0.3 + log10(AEP))]

    with ``A`` in km2, ``D`` in **minutes** and ``AEP`` as a fraction.  Note the
    internal consistency check that at AEP = 0.5 the two AEP-dependent terms
    vanish (``0.3 + log10(0.5) = 0``).

    **No coefficients are bundled.**  Supply the nine region-specific values
    (keys ``a`` ... ``i``) from the ARR table for your region, and verify the
    equation against the current guideline before use.  ARF is held at 1 for
    ``area <= 1 km2`` and the AEP is clamped to the range over which the ARR
    equations are stated to apply.
    """

    coeffs: Mapping[str, float]
    aep_range: tuple[float, float] = (0.005, 0.5)
    duration_range_min: tuple[float, float] = (720.0, 10080.0)
    warn_outside_duration: bool = True

    def __post_init__(self):
        missing = set("abcdefghi") - set(self.coeffs)
        if missing:
            raise ValueError(f"missing ARF coefficients: {sorted(missing)}")

    def __call__(self, area_km2: float, duration_h: float, aep: float) -> float:
        if area_km2 <= 1.0:
            return 1.0
        c = self.coeffs
        D = float(duration_h) * 60.0
        A = float(area_km2)
        p = float(np.clip(aep, *self.aep_range))
        t = 0.3 + np.log10(p)
        arf = (1.0
               - c["a"] * (A ** c["b"] - c["c"] * np.log10(D)) * D ** (-c["d"])
               + c["e"] * A ** c["f"] * D ** c["g"] * t
               + c["h"] * 10.0 ** (c["i"] * A * D / 1440.0) * t)
        return float(np.clip(arf, 0.0, 1.0))
