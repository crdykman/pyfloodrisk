"""
ARR temporal patterns and pre-burst rainfall.

``TemporalPatternLibrary`` holds the ARR Data Hub ensembles (10 patterns per
duration per AEP band) and samples one at random for a given duration and
rainfall AEP, aggregating it to the hydrological model's timestep in a
mass-conserving way.

``PreBurstSampler`` is optional but matters here.  ARR design rainfalls are
*bursts*; in a conventional event-based Monte Carlo framework the
embedded-burst problem is dealt with by adjusting the sampled initial loss.
Running GR4H in event mode from a sampled
*state* removes the loss parameter but not the problem: the state
distribution derived from continuous simulation is a distribution of states at
the start of a complete storm, whereas the IFD burst begins part way into one.
Two defensible treatments:

1. Prepend sampled pre-burst rainfall and let GR4H wet the stores itself
   (``PreBurstSampler``).  Physically the cleanest and the reason for using a
   continuous model at all.
2. Sample the state distribution conditional on the burst having been preceded
   by pre-burst rainfall of the sampled magnitude (see
   :class:`pyfloodrisk.dffa.states.InitialStateSampler` conditioning).

Do one or the other, not both, or you will double-count antecedent wetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "resample_increments",
    "TemporalPatternLibrary",
    "PreBurstSampler",
    "DEFAULT_AEP_BANDS",
]

#: AEP bands used by the ARR Data Hub temporal patterns, as
#: ``band -> (aep_lower, aep_upper)`` with AEP as a fraction.
#: The boundaries below reflect the ARR grouping of frequent (50%-20% AEP),
#: intermediate (10%-2%) and rare (1% and rarer).  Confirm against the header of
#: your own Data Hub download and override if it differs.
DEFAULT_AEP_BANDS: dict[str, tuple[float, float]] = {
    "frequent": (0.20, 1.00),
    "intermediate": (0.02, 0.20),
    "rare": (0.00, 0.02),
}


def band_for_aep(aep: float, bands: Mapping[str, tuple[float, float]] = None) -> str:
    bands = DEFAULT_AEP_BANDS if bands is None else bands
    for name, (lo, hi) in bands.items():
        if lo <= aep < hi or (aep == hi == 1.0):
            return name
    raise ValueError(f"no AEP band contains {aep}")


def resample_increments(inc: np.ndarray, ts_min: float, dt_min: float) -> np.ndarray:
    """Re-express rainfall increments on a new timestep, conserving total mass.

    Works for both aggregation (dt > ts, e.g. 5-minute ARR patterns to hourly
    GR4H) and disaggregation, by linear interpolation of the cumulative mass
    curve.  The pattern duration is padded to a whole number of new timesteps.
    """
    inc = np.asarray(inc, dtype=float)
    total_min = inc.size * ts_min
    n_new = int(np.ceil(total_min / dt_min - 1e-9))
    t_src = np.arange(inc.size + 1) * ts_min
    cum_src = np.concatenate([[0.0], np.cumsum(inc)])
    t_new = np.minimum(np.arange(n_new + 1) * dt_min, total_min)
    cum_new = np.interp(t_new, t_src, cum_src)
    return np.diff(cum_new)


@dataclass
class TemporalPatternLibrary:
    """Ensembles of ARR temporal patterns keyed by ``(duration_min, band)``.

    ``patterns[(duration_min, band)]`` is a dict with keys ``increments``
    (array ``(n_events, n_steps)`` of percentages of total depth),
    ``timestep_min`` (float) and ``event_ids`` (array of ids).
    """

    patterns: dict[tuple[int, str], dict] = field(default_factory=dict)
    bands: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_AEP_BANDS))

    # ------------------------------------------------------------- loading
    @classmethod
    def from_arr_increments_csv(cls, path, bands=None) -> "TemporalPatternLibrary":
        """Read an ARR Data Hub increments CSV.

        Expected layout (point or areal download): a header row of
        ``EventID, Duration, TimeStep, Region, AEP`` followed by one column per
        rainfall increment (expressed as a percentage of the burst depth), one
        row per pattern.  ``Duration`` and ``TimeStep`` are in minutes and
        ``AEP`` holds the band name.  Comment/metadata lines beginning with
        ``[`` are skipped.

        This loader is deliberately tolerant about column naming, but *do* check
        one duration by hand: the Data Hub file layout has changed between
        releases.
        """
        raw = pd.read_csv(path, header=0, skip_blank_lines=True,
                          comment=None, dtype=str, engine="python",
                          on_bad_lines="skip")
        raw = raw[~raw.iloc[:, 0].astype(str).str.startswith("[")]
        cols = {str(c).strip().lower(): c for c in raw.columns}
        need = ["eventid", "duration", "timestep", "aep"]
        missing = [n for n in need if n not in cols]
        if missing:
            raise ValueError(
                f"increments file is missing column(s) {missing}; got {list(raw.columns)}")
        meta = [cols[n] for n in need] + ([cols["region"]] if "region" in cols else [])
        inc_cols = [c for c in raw.columns if c not in meta]

        lib = cls(bands=dict(bands) if bands else dict(DEFAULT_AEP_BANDS))
        grp = raw.groupby([cols["duration"], cols["aep"], cols["timestep"]], sort=False)
        for (dur, band, ts), block in grp:
            inc = block[inc_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
            # trailing all-NaN columns are padding for shorter durations
            keep = ~np.all(np.isnan(inc), axis=0)
            inc = np.nan_to_num(inc[:, keep])
            key = (int(float(dur)), str(band).strip().lower())
            lib.patterns[key] = {
                "increments": inc,
                "timestep_min": float(ts),
                "event_ids": block[cols["eventid"]].to_numpy(),
            }
        if not lib.patterns:
            raise ValueError("no patterns parsed")
        return lib

    @classmethod
    def from_arrays(cls, mapping: Mapping[tuple[int, str], tuple[np.ndarray, float]],
                    bands=None) -> "TemporalPatternLibrary":
        """Build from ``{(duration_min, band): (increments, timestep_min)}``."""
        lib = cls(bands=dict(bands) if bands else dict(DEFAULT_AEP_BANDS))
        for (dur, band), (inc, ts) in mapping.items():
            inc = np.atleast_2d(np.asarray(inc, float))
            lib.patterns[(int(dur), str(band).lower())] = {
                "increments": inc,
                "timestep_min": float(ts),
                "event_ids": np.arange(inc.shape[0]),
            }
        return lib

    # ------------------------------------------------------------ sampling
    @property
    def durations_min(self) -> list[int]:
        return sorted({d for d, _ in self.patterns})

    def _lookup(self, duration_min: int, band: str) -> dict:
        key = (int(duration_min), band)
        if key in self.patterns:
            return self.patterns[key]
        avail = sorted(b for d, b in self.patterns if d == int(duration_min))
        if avail:
            raise KeyError(f"duration {duration_min} min has bands {avail}, not '{band}'")
        raise KeyError(f"no patterns for duration {duration_min} min "
                       f"(available: {self.durations_min})")

    def sample(self, duration_h: float, aep: float, dt_hours: float,
               rng: np.random.Generator):
        """Sample one pattern; returns ``(fractions_on_dt, event_id, band, index)``.

        Every member of the ensemble is equally likely (weight 1/10 in the
        standard ARR ensembles), so the pattern dimension is handled by plain
        Monte Carlo inside each rainfall stratum.
        """
        band = band_for_aep(float(aep), self.bands)
        entry = self._lookup(round(duration_h * 60), band)
        inc = entry["increments"]
        j = int(rng.integers(inc.shape[0]))
        frac = inc[j] / inc[j].sum()
        frac_dt = resample_increments(frac, entry["timestep_min"], dt_hours * 60.0)
        frac_dt = frac_dt / frac_dt.sum()
        return frac_dt, entry["event_ids"][j], band, j

    def all_on_dt(self, duration_h: float, aep: float, dt_hours: float) -> np.ndarray:
        """The whole ensemble on the model timestep -- for plotting/checking."""
        band = band_for_aep(float(aep), self.bands)
        entry = self._lookup(round(duration_h * 60), band)
        inc = entry["increments"]
        out = [resample_increments(r / r.sum(), entry["timestep_min"], dt_hours * 60.0)
               for r in inc]
        return np.array(out)


@dataclass
class PreBurstSampler:
    """Sample pre-burst rainfall depth as a ratio of the burst depth.

    Parameters
    ----------
    ratio_table
        ``DataFrame`` indexed by burst duration in hours, with columns holding
        non-exceedance percentiles (e.g. ``[10, 25, 50, 75, 90]``) of the
        pre-burst-to-burst depth ratio.  ARR publishes these by duration and
        AEP; if you have the AEP dependence, pass ``ratio_tables`` instead as
        ``{band: DataFrame}``.
    gap_hours
        Dry period inserted between the pre-burst rainfall and the burst.
    duration_ratio
        Pre-burst duration as a multiple of the burst duration.
    shape
        ``"uniform"`` spreads the pre-burst depth evenly, ``"increasing"`` uses
        a linearly rising pattern (wetter immediately before the burst).
    fixed_ratio
        If given, use this ratio deterministically and ignore ``ratio_table``
        (the "median pre-burst" convention).
    """

    ratio_table: pd.DataFrame | None = None
    ratio_tables: Mapping[str, pd.DataFrame] | None = None
    gap_hours: float = 0.0
    duration_ratio: float = 1.0
    shape: str = "uniform"
    fixed_ratio: float | None = None

    def sample_ratio(self, duration_h: float, aep: float,
                     rng: np.random.Generator) -> float:
        if self.fixed_ratio is not None:
            return float(self.fixed_ratio)
        table = self.ratio_table
        if self.ratio_tables is not None:
            table = self.ratio_tables[band_for_aep(aep)]
        if table is None:
            return 0.0
        pct = np.asarray([float(c) for c in table.columns], float) / 100.0
        # interpolate the ratio's own quantile function in the normal-variate
        # domain (smooth, monotone, sensible tails)
        vals = np.array([np.interp(np.log(duration_h),
                                   np.log(table.index.to_numpy(float)),
                                   table.iloc[:, j].to_numpy(float))
                         for j in range(table.shape[1])])
        z = norm.ppf(pct)
        u = rng.random()
        return float(max(0.0, np.interp(norm.ppf(u), z, vals)))

    def series(self, burst_depth_mm: float, duration_h: float, aep: float,
               dt_hours: float, rng: np.random.Generator):
        """Return ``(preburst_increments_mm, ratio)`` on the model timestep."""
        ratio = self.sample_ratio(duration_h, aep, rng)
        depth = ratio * burst_depth_mm
        n = max(1, int(round(self.duration_ratio * duration_h / dt_hours)))
        if self.shape == "increasing":
            w = np.arange(1, n + 1, dtype=float)
        else:
            w = np.ones(n)
        return depth * w / w.sum(), ratio
