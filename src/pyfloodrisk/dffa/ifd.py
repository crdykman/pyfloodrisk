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

__all__ = ["IFDCurve", "unit_arf", "ARR2019LongDurationARF", "ifd_table_from_csv",
           "ifd_table_from_bom_csv"]

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


def ifd_table_from_bom_csv(path) -> pd.DataFrame:
    """Read a BoM IFD download as-issued into the layout :class:`IFDCurve` wants.

    The Bureau's "IFD Design Rainfall Depth" CSV carries a metadata preamble
    (copyright, issue date, requested and nearest grid coordinates), then a
    two-row header, then one row per duration::

        ,,Annual Exceedance Probability (AEP)
        Duration,Duration in min,63.2%,50%,20%,10%,5%,2%,1%
        1 hour,60,40.4,46.0,63.4,74.8,85.8,100,111
        ...

    Durations are taken from the numeric ``Duration in min`` column rather
    than the ``1.5 hour`` text label, and the AEP headers are parsed by
    :func:`_parse_aep` (so ``63.2%`` becomes 0.632).

    Use :func:`ifd_table_from_csv` instead for a table you have already
    tidied yourself; this function is for the download as it comes.

    Parameters
    ----------
    path :
        Path to the BoM IFD CSV.

    Returns
    -------
    DataFrame indexed by duration in hours, columns AEP as a fraction.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()

    header_row = next(
        (i for i, line in enumerate(lines)
         if line.replace(" ", "").lower().startswith("duration,durationinmin")),
        None)
    if header_row is None:
        raise ValueError(
            f"{path}: no 'Duration,Duration in min,...' header row found; this "
            "does not look like a BoM IFD download (use ifd_table_from_csv "
            "for an already-tidy table)")

    raw = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig")
    raw = raw.dropna(how="all")
    cols = {str(c).strip().lower(): c for c in raw.columns}
    if "duration in min" not in cols:
        raise ValueError(f"{path}: no 'Duration in min' column")

    minutes = pd.to_numeric(raw[cols["duration in min"]], errors="coerce")
    keep = minutes.notna()

    # every column but the two duration columns is a depth column
    depth_cols = [c for c in raw.columns
                  if c not in (cols["duration in min"], cols.get("duration"))]
    table = pd.DataFrame(
        {_parse_aep(c): pd.to_numeric(raw.loc[keep, c], errors="coerce").to_numpy(float)
         for c in depth_cols},
        index=pd.Index(minutes[keep].to_numpy(float) / 60.0, name="duration_h"))
    table = table.dropna(axis=1, how="all")
    if table.empty:
        raise ValueError(f"{path}: no usable depths parsed")
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


# ------------------------------------------------- ARR 2019 ARF, full method
#: Coefficients ``a``-``i`` of the ARR 2019 **long**-duration ARF equation, by
#: ARF region (ARR Book 2, Chapter 4).  The region is a *map* lookup -- see the
#: ARF region figure in ARR -- not something derivable from a catchment's
#: coordinates or its name, so confirm yours against the map.
ARF_REGIONS: dict[str, dict[str, float]] = {
    "East Coast North":     dict(a=0.327,  b=0.241, c=0.448, d=0.360,
                                 e=9.6e-04, f=0.480, g=-0.210, h=0.01200,
                                 i=-0.00130),
    "Semi-arid Inland QLD": dict(a=0.159,  b=0.283, c=0.250, d=0.308,
                                 e=7.3e-07, f=1.000, g=0.039, h=0.0,
                                 i=0.0),
    "Tasmania":             dict(a=0.0605, b=0.347, c=0.200, d=0.283,
                                 e=7.6e-04, f=0.347, g=0.0877, h=0.01200,
                                 i=-0.00033),
    "SW WA":                dict(a=0.183,  b=0.259, c=0.271, d=0.330,
                                 e=3.845e-06, f=0.410, g=0.550, h=0.00817,
                                 i=-0.00045),
    "Central NSW":          dict(a=0.265,  b=0.241, c=0.505, d=0.321,
                                 e=5.6e-04, f=0.414, g=-0.021, h=0.01500,
                                 i=-0.00033),
    "SE Coast":             dict(a=0.060,  b=0.361, c=0.000, d=0.317,
                                 e=8.11e-05, f=0.651, g=0.0, h=0.0,
                                 i=0.0),
    "Southern Semi-arid":   dict(a=0.254,  b=0.247, c=0.403, d=0.351,
                                 e=1.3e-03, f=0.302, g=0.058, h=0.0,
                                 i=0.0),
    "Southern Temperate":   dict(a=0.158,  b=0.276, c=0.372, d=0.315,
                                 e=1.41e-04, f=0.410, g=0.150, h=0.01000,
                                 i=-0.00270),
    "Northern Coastal":     dict(a=0.326,  b=0.223, c=0.442, d=0.323,
                                 e=1.3e-03, f=0.580, g=-0.374, h=0.01300,
                                 i=-0.00150),
    "Inland Arid":          dict(a=0.297,  b=0.234, c=0.449, d=0.344,
                                 e=1.42e-03, f=0.216, g=0.129, h=0.0,
                                 i=0.0),
}

#: Coefficients of the ARR 2019 **short**-duration ARF equation, which is the
#: same Australia-wide (it has no regional dependence).
ARF_SHORT_COEFFICIENTS: dict[str, float] = dict(
    a=0.287, b=0.265, c=0.439, d=0.360, e=0.00226,
    f=0.226, g=0.125, h=0.0141, i=-0.021, j=0.213)

#: Durations (minutes) bounding the band ARR interpolates across: at or below
#: 720 min (12 h) the short-duration equation applies, at or above 1440 min
#: (24 h) the long-duration one, and between them the two are interpolated.
_ARF_SHORT_MAX_MIN = 720.0
_ARF_LONG_MIN_MIN = 1440.0


def _arf_long(area_km2: float, duration_min: float, aep: float,
              coeffs: Mapping[str, float]) -> float:
    """ARR 2019 long-duration (24 h and over) ARF equation."""
    c, t = coeffs, 0.3 + np.log10(aep)
    arf = (1.0
           - c["a"] * (area_km2 ** c["b"] - c["c"] * np.log10(duration_min))
           * duration_min ** -c["d"]
           + c["e"] * area_km2 ** c["f"] * duration_min ** c["g"] * t
           + c["h"] * 10.0 ** (c["i"] * area_km2 * duration_min / 1440.0) * t)
    return float(min(1.0, arf))


def _arf_short(area_km2: float, duration_min: float, aep: float) -> float:
    """ARR 2019 short-duration (12 h and under) ARF equation, Australia-wide."""
    c, t = ARF_SHORT_COEFFICIENTS, 0.3 + np.log10(aep)
    return float(1.0
                 - c["a"] * (area_km2 ** c["b"] - c["c"] * np.log10(duration_min))
                 * duration_min ** -c["d"]
                 + c["e"] * area_km2 ** c["f"] * duration_min ** c["g"] * t
                 + c["h"] * area_km2 ** c["j"]
                 * 10.0 ** (c["i"] * (1.0 / 1440.0) * (duration_min - 180.0) ** 2)
                 * t)


def _arf_small_area(arf_at_10: float, area_km2: float) -> float:
    """ARR scaling of a 10 km2 ARF down to a catchment of 1-10 km2."""
    return 1.0 - 0.6614 * (1.0 - arf_at_10) * (area_km2 ** 0.4 - 1.0)


@dataclass
class ARR2019ARF:
    """ARR 2019 areal reduction factor -- the complete method.

    Unlike :class:`ARR2019LongDurationARF`, which is the long-duration
    functional form with no coefficients bundled, this implements the whole
    ARR Book 2 Chapter 4 procedure and *does* bundle the coefficients:

    * bursts of 24 h and longer use the region's long-duration equation;
    * bursts of 12 h and shorter use the Australia-wide short-duration
      equation;
    * between 12 and 24 h the two are interpolated linearly in duration,
      between the 12 h short value and the 24 h long value;
    * catchments of 1-10 km2 are scaled down from the 10 km2 value by
      ``1 - 0.6614 (1 - ARF_10)(A**0.4 - 1)``;
    * catchments under 1 km2 get ARF = 1, i.e. no reduction.

    Parameters
    ----------
    region :
        One of :data:`ARF_REGIONS`.  **Look this up on the ARR ARF region
        map**: it is not derivable from the catchment coordinates, and the
        regions do not follow state boundaries.
    coeffs :
        Override the bundled long-duration coefficients, e.g. if ARR is
        revised.  Keys ``a`` to ``i``.
    aep_range :
        AEP is *clamped* to this range rather than rejected, because a
        derived-FFA run samples well outside the range ARR states the
        equations for -- down to 1e-5 and up to 0.9, against ARR's 0.0005
        to 0.5.  Clamping holds the ARF flat outside the fitted range
        instead of extrapolating a fitted form beyond its data; the
        alternative would be to fail the whole Monte Carlo.
    round_to :
        Decimal places, matching the precision ARF is tabulated to.
        ``None`` keeps full precision.

    Notes
    -----
    ARR states the short-duration equation does not apply to catchments over
    1000 km2, and that is enforced for bursts of 12 h and under.  Inside the
    12-24 h interpolation band the short-duration equation is still
    evaluated at 12 h for such catchments, because the interpolation rule
    requires it -- treat results there with care.
    """

    region: str
    coeffs: Mapping[str, float] | None = None
    aep_range: tuple[float, float] = (0.0005, 0.5)
    round_to: int | None = 3

    def __post_init__(self):
        if self.coeffs is None:
            if self.region not in ARF_REGIONS:
                raise ValueError(
                    f"unknown ARF region {self.region!r}; the ARR regions are "
                    f"{sorted(ARF_REGIONS)}")
            self.coeffs = dict(ARF_REGIONS[self.region])
        else:
            missing = set("abcdefghi") - set(self.coeffs)
            if missing:
                raise ValueError(f"missing ARF coefficients: {sorted(missing)}")

    def __call__(self, area_km2: float, duration_h: float, aep: float) -> float:
        area = float(area_km2)
        duration_min = float(duration_h) * 60.0
        if not (np.isfinite(area) and np.isfinite(duration_min)
                and np.isfinite(aep)):
            raise ValueError(
                f"ARF needs finite inputs; got area_km2={area_km2}, "
                f"duration_h={duration_h}, aep={aep}")
        p = float(np.clip(aep, *self.aep_range))

        if area < 0.0 or area > 30000.0:
            raise ValueError(f"area must be 0 to 30000 km2, got {area}")
        if duration_min <= 0.0 or duration_min > 7 * 24 * 60:
            raise ValueError(
                f"duration must be positive and at most 7 days, got {duration_h} h")
        if duration_min <= _ARF_SHORT_MAX_MIN and area > 1000.0:
            raise ValueError(
                "the ARR short-duration ARF equation does not apply to "
                f"catchments over 1000 km2; got {area} km2 at {duration_h} h")

        if area < 1.0:
            return 1.0

        if duration_min >= _ARF_LONG_MIN_MIN:
            arf = (_arf_long(area, duration_min, p, self.coeffs) if area >= 10.0
                   else _arf_small_area(
                       _arf_long(10.0, duration_min, p, self.coeffs), area))
        elif duration_min <= _ARF_SHORT_MAX_MIN:
            arf = (_arf_short(area, duration_min, p) if area >= 10.0
                   else _arf_small_area(_arf_short(10.0, duration_min, p), area))
        else:
            # 12-24 h: interpolate between the 12 h short and 24 h long values
            ref = max(area, 10.0)
            short_12 = _arf_short(ref, _ARF_SHORT_MAX_MIN, p)
            long_24 = _arf_long(ref, _ARF_LONG_MIN_MIN, p, self.coeffs)
            arf = short_12 + (long_24 - short_12) * (
                duration_min - _ARF_SHORT_MAX_MIN) / _ARF_SHORT_MAX_MIN
            if area < 10.0:
                arf = _arf_small_area(arf, area)

        arf = float(np.clip(arf, 0.0, 1.0))
        return arf if self.round_to is None else round(arf, self.round_to)
