# Derived flood frequency analysis with GR4H in event mode

`pyfloodrisk.dffa` is a Monte Carlo (joint probability) framework in the form
used for event-based design flood estimation under ARR, with the sampled
initial loss replaced by a jointly sampled GR4H **state vector** drawn from a
continuous simulation. Rainfall depths come from an IFD curve, temporal
patterns from the ARR ensembles, and antecedent conditions from the state
distribution of a continuous run of the same calibrated model. Sampling of the
rainfall probability domain is stratified, and flood quantiles are estimated
empirically by the total probability theorem.

```
pyfloodrisk/dffa/
  ifd.py              IFD curve fitting/inversion, CSV loader, areal reduction
  stratification.py   strata over the rainfall AEP domain, with weights
  patterns.py         ARR temporal pattern ensembles, pre-burst rainfall
  states.py           joint resampling of GR4H states, PET climatology
  engine.py           EventModel interface and the GR4H wrapper
  mcs.py              the Monte Carlo driver and the results object
  tpt.py              weighted exceedance / quantiles / bootstrap
  diagnostics.py      figures
  workflow.py         bridges to the rest of pyfloodrisk
examples/dffa_demo.py end-to-end run on a bundled demo station
tests/test_dffa*.py   what can actually be checked against something known
```

Requirements beyond the package's own: `matplotlib` for `diagnostics`, and
`pyvinecopulib` (extra `copula`) for the copula and KDE state-sampling
methods. No R.

## How it sits in the package

The framework needs four inputs. Three of them are things `pyfloodrisk`
already produces:

| input | where it comes from |
|---|---|
| event model | a calibrated parameter set through `GR4HEventEngine` — the same `pyfloodrisk.gr4h.GR4H` code the calibration uses |
| initial state distribution | `continuous_state_table(forcings, params, area)`, a continuous GR4H run over the station's climate record |
| temporal patterns | `station_patterns(station)`, the station's ARR Data Hub increments file — the same file `build_design_storm` reads |
| design rainfall (IFD) | **a CSV you supply.** BoM depths in a tidy table, read with `ifd_table_from_csv`. A *demonstration* table is bundled per demo station (`station_ifd`) so the example runs; it is not a design IFD |

The state distribution is the input with no ARR equivalent, and the reason the
model has to be the same on both sides: the event is *hot-started* from a state
the continuous run was actually in, so `x1`, `x3` and the unit-hydrograph
length must be the ones that produced the table.

```python
from pyfloodrisk.dffa import (DerivedFFA, GR4HEventEngine, IFDCurve, MCSConfig,
                              STANDARD_DURATIONS_H, Stratification,
                              TemporalPatternLibrary, continuous_state_table,
                              ifd_table_from_csv, pet_climatology,
                              state_sampler_from_run, diagnostics)

params = {"x1": X1, "x2": X2, "x3": X3, "x4": X4}     # from calibration()

table  = continuous_state_table(forcings, params, AREA, warmup_hours=8760, thin=6)
states = state_sampler_from_run(table, params, method="bootstrap")
engine = GR4HEventEngine(params, area_km2=AREA)

ifd = IFDCurve(ifd_table_from_csv("bom_ifd.csv"), arf=my_arf)
tp  = TemporalPatternLibrary.from_arr_increments_csv("ARR_Point_Increments.csv")

strat = Stratification.uniform_in_z(aep_max=0.9, aep_min=1e-6,   # see below
                                    n_strata=50, n_per_stratum=200)
cfg   = MCSConfig(area_km2=AREA, durations_h=STANDARD_DURATIONS_H)

res = DerivedFFA(ifd, tp, states, engine, cfg, strat,
                 pet=pet_climatology(forcings)).run()
print(res.summary())
diagnostics.plot_all(res, "figures", ifd=ifd, states=states)
res.to_csv("events.csv")
```

`run_dffa(station=...)` does all of the above for a bundled demo station in one
call. It is a demonstration, not a template: its design rainfalls are the
bundled demo table (fitted to six years of one pluviograph), and its parameters
are plausible rather than calibrated.

## The estimator

Each simulated event *i* carries a probability weight `w_i = mass_k / n_k` where
`mass_k` is the probability mass of its rainfall interval. Following ARR Book 4,
Section 4.3.3.3 the estimator is written interval by interval,

```
P(Q > q) = Σ_k p[R_k] · P(Q > q | R_k),    P(Q > q | R_k) ≈ n_k(q) / N_k
```

and inverted (linearly in log P) to give quantiles. Design quantiles are the
envelope over storm durations, with the critical duration recorded at each AEP.

The conditional exceedance probability inside an interval is a **plotting
position** rather than the raw `n/N`:

```
P(Q > q | R_k) = (n_k(q) - a) / (N_k + 1 - 2a)
```

with **Cunnane's a = 0.4** the default (`plotting_position="cunnane"`), since it
is approximately quantile-unbiased across the distributions used in flood
frequency work. `"hazen"` (0.5), `"weibull"` (0), `"gringorten"` (0.44),
`"blom"`, `"beard"`, `"apl"` and bare floats are accepted. Two details: it is
applied *within* each interval and forced to zero where no simulation in the
interval exceeds `q`, so it introduces no probability floor in the far tail;
and the pooled (`estimator="truncated"`) form uses the weight-generalised
version of the same family, `P_i = W(C_i - a·w_i)/(W + (1-2a)·w_i)`, which
reduces exactly to `(i-a)/(n+1-2a)` for equal weights.

**End intervals are treated as ARR specifies**, so the probability domain is
closed rather than truncated:

| | frequent end | rare end |
|---|---|---|
| interval covers | AEP > `edges[0]` | AEP < `edges[-1]` |
| mass `p[R_k]` | `1 - edges[0]` (non-exceedance of the upper-bound rainfall) | `edges[-1]` (exceedance of the lower-bound rainfall) |
| rainfall | held at `edges[0]` | held at `edges[-1]` |
| conditional exceedance | geometric mean of `c` and `0.1·c` = `c·√0.1` | geometric mean of `c` and 1 = `√c` |

The interval masses then sum to exactly 1. `Stratification(open_ends=False)` and
`DFFAResults(estimator="truncated")` give the plain truncated scheme instead,
where the masses sum to `edges[0] - edges[-1]` and the curve is undefined
outside that range.

`tests/test_dffa.py` holds two analytical checks: substituting the identity
map for GR4H must reproduce the rainfall frequency curve exactly (through both
estimators), and the two geometric-mean rules are checked directly. Those are
the tests to keep running as you modify things.

### Check the boundary is actually where ARR assumes it is

ARR justifies the geometric-mean end rules "on the assumption that these
boundary intervals are distant from the probability region of interest". In a
framework where **antecedent wetness is the sampled variable**, that assumption
is worth testing rather than inheriting: a 50% AEP burst on a saturated
catchment can produce a large flood, so the frequent open interval — rainfall
pinned at `edges[0]`, conditional exceedance damped by an arbitrary `√0.1` — can
end up influencing the answer.

`DFFAResults.end_interval_contributions(aeps)` reports the share of each
quantile's exceedance probability coming from the two open intervals, and
`summary()` carries it as `end_share`. On the synthetic catchment the framework
was developed against, with the ARR default `aep_max=0.5`, the frequent open
interval supplied 20% of the exceedance probability at the 20% AEP quantile;
raising `aep_max` to 0.9 cut that to 2% and moved the quantile 11%. So: if you
report quantiles more frequent than about 2% AEP, extend `aep_max` (and supply
design rainfalls that reach there — ARR's very frequent design rainfalls go to
12 EY) rather than debating the value of `first_factor`. The rare end is well
behaved by comparison: with `aep_min = 1e-5` it contributed under 0.6% at the
1 in 1000 AEP quantile. Read `end_share` on your own catchment; both defaults
here (`aep_max=0.9`) are chosen to keep it small, and it is cheap to check.

## Sampling the initial state

The state table is one row per timestep of a long continuous GR4H run, holding
`prod_store`, `rout_store`, the UH memory and the date.
`continuous_state_table` builds it; `thin` keeps every *n*-th row, which costs
almost nothing because consecutive hours are nearly identical states, and
`wet_only=True` restricts it to states at the onset of rainfall.

Which rows belong in the donor pool is a modelling choice, not a detail.
Three defensible answers, in increasing order of conditioning:

```python
table = continuous_state_table(forcings, params, AREA, thin=6)   # any hour
table = continuous_state_table(forcings, params, AREA, wet_only=True)  # rain starts
table = event_onset_states(continuous_state_table(forcings, params, AREA))
```

`event_onset_states` runs the package's own baseflow separation and event
delineation over the continuous run's discharge and keeps the antecedent state
ahead of each delineated event — the same operation `extract_initial_states`
performs for the single-event workflow, so the two paths condition on the same
thing. Delineation finds every rise, most of which are not floods, so the
events are then trimmed to an exceedance-per-year rate — by default the 6
largest per year of record (`ey=6`), ranked on peak flow or on event volume
(`rank_by="peak"` or `"volume"`); `ey=None` keeps the lot. On the bundled demo
station that is the difference between 220 rises and 33 events, and it roughly
doubles the median antecedent production store (73 → 142 mm), which is the
whole point: big events start on wet catchments. It is a far smaller pool (one
row per event rather than per hour), so it is the case where the smoothed and
copula methods earn their keep. Conditioning
on rainfall or on events makes the pool wetter, and the design flood larger;
which is right depends on whether you also prepend pre-burst rainfall — do one
or the other, not both.

The non-bootstrap methods model a small set of **state variables** — the two
stores and `uh_total`, the total water in transit through the unit hydrographs.
`uh_total` is the physically meaningful scalar; the *shape* of the UH memory is
a deterministic consequence of the last few timesteps of rainfall, so it is
taken from a donor row and rescaled to the sampled total
(`uh_profile="scaled_donor"`) rather than sampled ordinate by ordinate, which
would produce states GR4H could never reach.

| `method` | what it does | use it for |
|---|---|---|
| `bootstrap` | draws whole rows | production runs — preserves every dependence exactly |
| `smoothed` | donor + jitter, kernel covariance taken from the pool, donor shrunk by `1/√(1+h²)` so the covariance is preserved | short or heavily conditioned donor pools |
| `empirical_copula` | nonparametric **vine copula** (`pyvinecopulib`, TLL pair copulas) on the pseudo-observations, with kernel or empirical margins | dependence estimated rather than assumed; asymmetric and tail dependence |
| `independent_kde` | each state variable from its own boundary-corrected 1-D KDE (`pv.Kde1d`, which handles the bounded stores and the point mass at zero in `uh_total`), sampled independently | isolating how much the *joint* structure matters, as against the marginals alone |

`InitialStateSampler.dependence(sampled)` reports Kendall's tau between state
variables for the input and the sample — **run it before trusting any
non-bootstrap method**, because it says whether the dependence you meant to
preserve (or destroy) actually was. It earns its keep immediately on the
bundled demo station: `smoothed` reproduces the tau between the two stores
(0.710 against 0.712) but collapses the tau between each store and `uh_total`
(0.19 against 0.79), because in a dry catchment `uh_total` is zero for most
hours and jittering a variable with a point mass at its boundary smears that
mass out. `bootstrap` has no such problem, which is one more reason it is the
default.

## Things to verify before trusting output

These are the places where the code is doing something you should confirm
against your own data rather than take on trust.

**The state vector is the engine's own.** The state dict
`{"prod_store", "rout_store", "uh1", "uh2"}` only means what
`GR4HEventEngine` takes it to mean. `pyfloodrisk`'s GR4H splits the routed
depth 0.9/0.1 between the two unit hydrographs at the point of use, so the
arrays hold *unsplit* depth, and ordinate 0 has already been released by the
step the state was recorded at, so it is dropped rather than released twice.
Another GR4H — airGR's Fortran core, for instance — splits on entry, and
handing it these arrays would quietly double-count or lose water. So: generate
the state table with the engine that will consume it (`continuous_state_table`
does this), and with the same parameter set, because a table built at a
different `x4` is silently zero-padded or truncated to the engine's UH length.
`tests/test_gr4h_state.py` pins the convention down algebraically; if you
plug a different model in behind your own `EventModel` subclass, write the
equivalent test for it first.

**Design rainfalls.** Nothing in the package is a design IFD. They enter one
way only — a tidy CSV read with `ifd_table_from_csv`, so the depths you run on
are depths you put there. The tables under `pyfloodrisk/data/ifd/` that
`station_ifd` reads exist so the demo has something to run on: each was fitted
to six-odd years of that station's own pluviograph, and says so in its header.
That is not a substitute for design rainfall — ARR depths are regionalised from
the whole gauge network and extend to AEPs no single record can support — so
extract BoM depths into a table of the same layout before reporting anything.

**Areal reduction.** `ARR2019LongDurationARF` implements the *functional form*
of the ARR long-duration ARF equation; the nine region-specific coefficients
are not bundled and you must supply them from ARR Book 2, Chapter 4. The form
has one useful internal check: at 50% AEP the two AEP-dependent terms vanish.
There is also a separate short-duration equation in ARR that is not implemented
here. Without an `arf`, depths are point depths.

**ARR temporal pattern file layout.** The loader expects `EventID, Duration,
TimeStep, Region, AEP` followed by the increment columns, with `AEP` holding the
band name — which is the layout of the files bundled with the package. The Data
Hub layout has changed between releases; parse one duration and check the
increments sum to 100 before running anything (`test_dffa_workflow.py` does
this for the bundled files).

**AEP band boundaries.** `DEFAULT_AEP_BANDS` groups frequent (50%–20%),
intermediate (10%–2%) and rare (1% and rarer). Confirm against the header of
your own download and override if it differs, because the band drives which
ensemble is sampled.

**Sub-hourly durations.** GR4H is hourly — its percolation coefficient (21/4)
and unit-hydrograph exponent (1.25) are the hourly values. The 1-hour and
2-hour ARR bursts are aggregated from 5-minute patterns to hourly, which for a
small catchment throws away most of the information in the pattern. If short
durations are critical for your catchment, rescale the parameters for a
sub-hourly step and set `dt_hours` accordingly — the framework is
timestep-agnostic, the model is not, and `GR4HEventEngine` warns when
`dt_hours != 1`.

## Methodological caveats worth arguing about in the paper

**Independence of rainfall and antecedent state.** Depth, pattern and state are
sampled independently. The pattern ensembles are conditioned on the AEP band, so
there is a coarse depth-pattern dependence; there is none between depth and
antecedent wetness. That assumption is the standard one, and it is also the one
your state distribution is uniquely placed to test: pass `state_pool_fn` to
`DerivedFFA` (a callable returning a donor-pool mask given the sampled rainfall
AEP and duration) and compare the resulting curve against the independent case.
If the difference is material, the independent case is an unquantified bias in
the whole event-based framework, not a limitation of this code.

**Bursts, not complete storms.** IFD depths are bursts, and the state
distribution from continuous simulation is a distribution of states at the
*start of a complete storm* — not at the start of an embedded burst. The
conventional event-based framework patches this by adjusting the sampled
initial loss. Here you have two cleaner options: prepend sampled pre-burst
rainfall and let GR4H wet the stores itself (`PreBurstSampler`), or condition
the state distribution on antecedent rainfall
(`continuous_state_table(wet_only=True)`, or `state_pool_fn`). Doing both
double-counts the antecedent wetting.

**Enveloping durations.** Taking the maximum over durations at each AEP is
consistent with design practice but biases the estimate high, because the
maximum of several noisy estimates exceeds the maximum of their expectations —
visible in the convergence diagnostic, where the enveloped quantile sits several
per cent high at small sample sizes. `DFFAResults.pooled_quantiles` gives the
unbiased alternative if you are prepared to assume a probability mass function
over durations. Reporting both is the honest option.

**What the confidence limits cover.** `confidence()` bootstraps events *within*
strata, so it isolates the Monte Carlo sampling error of the simulation. It says
nothing about uncertainty in the IFD, the finite temporal pattern ensemble, the
state distribution, or the calibrated parameters. An outer loop over parameter
sets is the natural extension and the estimator is already weight-based, so it
slots in without changing `tpt.py` — and `behavioural_posterior` /
`robust_calibration` already give you the parameter sets to loop over. Note
that each parameter set needs its own state table, because `x4` sets the length
of the UH memory.

**Baseflow.** Not added separately — GR4H's routing store produces it. This is a
substantive difference from a conventional event-based implementation, where
baseflow is a separately sampled input, and it is one of the arguments for
doing this at all.

## Diagnostics

`diagnostics.plot_all` writes seven figures: the derived frequency curve with
confidence limits, the per-duration curves and envelope, the stratum
contributions to exceedance probability, the sampled inputs, the sensitivity of
the curve to antecedent state, the convergence check, and a sample of
hydrographs.

The stratum-contribution figure is the one to read first. Its right-hand panel
gives the rainfall AEP at which half a flood quantile's exceedance probability
has accumulated: for a catchment where antecedent conditions matter, the 1% AEP
flood is generated mostly by rainfall more frequent than 1% AEP, and by how much
is the result the whole exercise is for.
