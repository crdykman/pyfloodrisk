# pyfloodrisk

Flood Risk Assessment Workflow for GR4H-Based Design Flood Analysis.

`pyfloodrisk` is a Python port of an R-based design flood workflow built
around the GR4H hourly rainfall-runoff model. It covers the full pipeline
from streamflow calibration through to design flood simulation:

- **GR4H** — hourly production/routing rainfall-runoff model
  (`pyfloodrisk.gr4h`).
- **Calibration** — DREAM-based calibration against observed streamflow,
  plus an optional robust re-calibration step that re-ranks the
  behavioural posterior against design storm events
  (`calibration`, `behavioural_posterior`, `robust_calibration`).
- **Event delineation** — baseflow separation and hydrologic event
  extraction (peaks-over-threshold or local maxima), trimmed to the
  largest events per year by peak flow or volume (6 EY by default), and
  used to derive antecedent model states ahead of each event
  (`hydro_event_pipeline`, `extract_initial_states`).
- **Design storms** — builds design storms from temporal-pattern
  increment files (`build_design_storm`).
- **Design flood simulation** — runs GR4H forward across design storm
  patterns and antecedent states to produce a design flood ensemble
  (`simulate_design_flood`, `run_demo_workflow`).
- **Derived flood frequency analysis** — a stratified Monte Carlo (joint
  probability) framework in the ARR event-based form, with the sampled
  initial loss replaced by a GR4H state vector drawn jointly from a continuous
  run, producing a full flood frequency curve rather than a single design
  event (`pyfloodrisk.dffa`; see [docs/dffa.md](docs/dffa.md)).

A small bundled demo dataset (climate data and design storm increments
for two stations) lets the whole workflow run end-to-end without any
external data.

## Installation

```bash
pip install pyfloodrisk
```

Optional extra: `pip install pyfloodrisk[copula]` for the copula and KDE
methods of sampling the initial state distribution.

## Quickstart

Design flood ensemble for one design storm:

```python
from pyfloodrisk import run_demo_workflow

results = run_demo_workflow(station="117002A")
```

Derived flood frequency curve across the whole probability domain:

```python
from pyfloodrisk import run_dffa

out = run_dffa(station="117002A")
print(out["results"].summary())
```

`run_dffa` is a demonstration — its design rainfalls are the station's
bundled BoM IFD download, areally reduced with `ARR2019ARF`, and
its parameters are plausible rather than calibrated. Design rainfalls
enter one way only, from a CSV: `ifd_table_from_bom_csv` for a Bureau
download as-issued, `ifd_table_from_csv` for a table you have tidied
yourself. [docs/dffa.md](docs/dffa.md) sets out what to replace
before the numbers mean anything, and `examples/dffa_demo.py` builds the
same analysis input by input.

## License

MIT — see [LICENSE](LICENSE).