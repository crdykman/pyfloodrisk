# pyfloodrisk

Flood Risk Assessment Workflow for GR4H-Based Design Flood Analysis.

`pyfloodrisk` is a Python port of an R-based design flood workflow built
around the GR4H hourly rainfall-runoff model. It covers the full pipeline
from streamflow calibration through to design flood simulation:

- **GR4H** — hourly production/routing rainfall-runoff model
  (`pyfloodrisk.gr4h`). Events can be hot-started from an explicit state
  — both stores *and* the unit-hydrograph memory — so a design event
  continues a continuous run rather than starting from empty
  (`GR4H.run_from_state`, `continuous_state_table`).
- **Calibration** — DREAM-based calibration against observed streamflow,
  plus an optional robust re-calibration step that re-ranks the
  behavioural posterior against design storm events
  (`calibration`, `behavioural_posterior`, `robust_calibration`).
- **Event delineation** — baseflow separation and hydrologic event
  extraction (peaks-over-threshold or local maxima), trimmed to the
  largest events per year by peak flow or volume, 6 EY by default
  (`hydro_event_pipeline`).
- **Antecedent states** — the model state the catchment was actually in
  before its own large rainfall bursts, taken separately for each storm
  duration, so a 72 h design storm starts from the wetness that precedes
  72 h bursts (`rainfall_events`,
  `extract_initial_states_per_duration`). This is used in place of
  design pre-burst rainfall or an adjusted initial loss.
- **Design rainfall** — Bureau of Meteorology IFD downloads read as
  issued (`ifd_table_from_bom_csv`), reduced from point to catchment
  depths by the ARR 2019 areal reduction factors (`ARR2019ARF`), with
  ARR Data Hub temporal patterns (`TemporalPatternLibrary`).
- **Design storms** — built from temporal-pattern increment files, at
  any duration the file covers. The increment timestep is read from the
  file rather than assumed, since ARR coarsens it as the storm
  lengthens (`build_design_storm`).
- **Design flood simulation** — runs GR4H forward across every
  combination of temporal pattern and antecedent state to produce a
  design flood ensemble for one design storm
  (`simulate_design_flood`, `run_demo_workflow`).
- **Derived flood frequency analysis** — a stratified Monte Carlo (joint
  probability) framework in the ARR event-based form, with the sampled
  initial loss replaced by a GR4H state vector drawn jointly from a
  continuous run, producing a full flood frequency curve rather than a
  single design event (`pyfloodrisk.dffa`; see
  [docs/dffa.md](docs/dffa.md)).

A bundled demo dataset — climate records, BoM IFD tables and ARR
temporal-pattern increments for three stations (117002A, 303203 and
405214, 51–357 km²) — lets the whole workflow run end to end without any
external data.

## Installation

```bash
pip install pyfloodrisk
```

Requires Python 3.10 or newer.

Optional extra: `pip install pyfloodrisk[copula]` for the copula and KDE
methods of sampling the initial state distribution. The default
bootstrap method needs none of it.

## Quickstart

Design flood ensemble for one design storm. `duration_hours` sets the
storm duration *and* the bursts the antecedent states are drawn from, so
the two cannot drift apart:

```python
from pyfloodrisk import run_demo_workflow

results = run_demo_workflow(station="117002A", duration_hours=12)
```

Derived flood frequency curve across the whole probability domain, with
one state distribution per storm duration:

```python
from pyfloodrisk import run_dffa

out = run_dffa(station="303203")
print(out["results"].summary())
```

Both are demonstrations. Their design rainfalls are the station's
bundled BoM IFD download, areally reduced with `ARR2019ARF`. Design 
rainfalls enter one way only — `ifd_table_from_bom_csv`, reading a 
Bureau IFD download as it was issued. [docs/dffa.md](docs/dffa.md) 
sets out what to replace before the numbers mean anything, and 
`examples/dffa_demo.py` builds the same analysis input by input.

## Package layout

`pyfloodrisk.gr4h` is a leaf: the model, its calibration, and the state
table it produces. The main package builds on it, and
`pyfloodrisk.dffa` builds on both — never the other way round.

## License

MIT — see [LICENSE](LICENSE).
