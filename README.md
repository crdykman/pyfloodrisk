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
  extraction (peaks-over-threshold or local maxima), used to derive
  antecedent model states ahead of each event (`hydro_event_pipeline`,
  `extract_initial_states`).
- **Design storms** — builds design storms from temporal-pattern
  increment files (`build_design_storm`).
- **Design flood simulation** — runs GR4H forward across design storm
  patterns and antecedent states to produce a design flood ensemble
  (`simulate_design_flood`, `run_demo_workflow`).

A small bundled demo dataset (climate data and design storm increments
for two stations) lets the whole workflow run end-to-end without any
external data.

## Installation

```bash
pip install -e .
```

For running the test suite:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from pyfloodrisk import run_demo_workflow

results = run_demo_workflow(station="421026")
```

## Development


## License

MIT — see [LICENSE](LICENSE).