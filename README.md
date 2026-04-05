# F1 2026 Strategy Predictor

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Naksh-IIITB/hacknight)

This project is a hackathon-ready baseline for an AI-assisted Formula 1 strategy model built around the 2026 regulations. It predicts qualifying pace, suggests ERS deploy/harvest behavior, estimates qualifying fuel loads, and simulates race stints with tyre and dirty-air effects.

It is intentionally a **hybrid simulator**, not a fake black box. Public F1 data does not expose full telemetry, power-unit maps, tyre carcass temperatures, or complete aero balance data for every team, so the model combines:

- track metadata for all 24 races on the 2026 calendar,
- generated track-map artifacts for all 24 circuits,
- telemetry-backed corner markers for the completed 2026 rounds,
- a trained historical lap regressor built from cached FastF1 session data,
- early-season 2026 competitive priors from Australia, China, and Japan,
- team-level one-lap and race-pace offsets,
- tyre degradation heuristics,
- fuel-mass sensitivity,
- dirty-air penalties and overtaking friction,
- 2026 ERS deployment and harvesting logic.

## Why this fits the HackNite brief

The provided PDF focuses on rubric quality more than domain rules. This project is designed to score well against that rubric:

- **Problem statement and idea**: an explainable race-engineering assistant for 2026 F1.
- **AI/ML implementation**: hybrid predictive model with structured inputs and tunable priors.
- **System design**: clean separation between data, model logic, and API.
- **Code quality**: modular Python package with tests.
- **Deployment**: FastAPI app with simple endpoints.
- **Documentation**: assumptions, inputs, outputs, and limitations are all explicit.

## Project structure

```text
.
├── api.py
├── data/
│   ├── historical_laps_2025_2026.csv
│   ├── track_maps_2026.json
│   ├── teams_2026.json
│   └── tracks_2026.json
├── f1_ai/
│   ├── __init__.py
│   ├── data.py
│   ├── engine.py
│   ├── maps.py
│   └── ml.py
├── models/
│   ├── lap_regressor_metadata.json
│   └── lap_time_regressor.joblib
├── requirements.txt
├── scripts/
│   ├── build_datasets.py
│   └── train_regressor.py
├── static/
│   └── index.html
└── tests/
    └── test_engine.py
```

## Quick start

```bash
python3 -m pip install --user -r requirements.txt
PYTHONPATH=. python3 scripts/build_datasets.py
PYTHONPATH=. python3 scripts/train_regressor.py
python3 -m uvicorn api:app --reload
```

Then open the dashboard at `http://127.0.0.1:8000/` or the API docs at `http://127.0.0.1:8000/docs`.

## Deploy on Render

This repo is prepared for Render with:

- [render.yaml](/Users/naksh/Documents/New%20project/render.yaml)
- [Dockerfile](/Users/naksh/Documents/New%20project/Dockerfile)
- [runtime.txt](/Users/naksh/Documents/New%20project/runtime.txt)

Fastest path:

1. Push this project to GitHub.
2. In Render, choose **New +** then **Blueprint**.
3. Connect the GitHub repo.
4. Render will detect `render.yaml` and create the web service.
5. Open the deployed service URL once the build finishes.

Important note:

- The app depends on the checked-in generated artifacts in `data/` and `models/`, so keep those files in the repo when you deploy.

## Current trained model

After the latest refresh, the saved historical lap regressor was trained on:

- **13,129** historical lap rows,
- dry qualifying and race laps,
- a 2025 season subset plus Australia, China, and Japan in 2026,
- feature inputs spanning track, team, tyre, fuel, and session context.

Held-out validation metrics:

- **MAE**: `0.8907s`
- **RMSE**: `2.7937s`
- **R²**: `0.8938`

## Example requests

Qualifying prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/qualifying \
  -H "Content-Type: application/json" \
  -d '{"team":"mercedes","track":"australia","ambient_temp_c":26,"wind_kph":10,"push_laps":1}'
```

Race simulation:

```bash
curl -X POST http://127.0.0.1:8000/simulate/race \
  -H "Content-Type: application/json" \
  -d '{"team":"ferrari","track":"japan","ambient_temp_c":24,"starting_position":4,"traffic_level":0.55}'
```

Per-track team strategy board:

```bash
curl http://127.0.0.1:8000/strategy/japan
```

## Model logic

### 1. Qualifying predictor

Qualifying pace is built from:

- track-specific baseline pace,
- team one-lap delta,
- sector-fit adjustment using traction, high-speed efficiency, and straight-line efficiency,
- temperature penalty,
- wind penalty,
- fuel penalty,
- track evolution gain,
- historical ML correction blended with the heuristic model.

### 2. ERS deployment and harvesting

The ERS planner reflects the 2026 direction of travel:

- approximately 50/50 ICE-electric split,
- 350kW MGU-K,
- no MGU-H,
- greater driver control over harvesting and deployment.

The output is a zone-level guidance layer that can later be replaced by telemetry-driven optimization.

### 3. Race strategy simulation

For each team-track combination, the simulator evaluates candidate strategies:

- `M-H`
- `H-M`
- `S-M-H`
- `M-M-H`

Each stint model accounts for:

- compound pace offset,
- degradation from tyre stress and ambient temperature,
- team tyre management strength,
- fuel burn,
- dirty-air pace loss,
- pit loss,
- ML-adjusted average stint pace.

### 4. Track maps and dashboard

The frontend dashboard now includes:

- a browser-based control panel for team, track, temperature, and grid position,
- a track map viewer,
- corner markers,
- strategy-board comparisons,
- live access to the ML-backed predictor through the FastAPI backend.

## Important limitations

This version is **viable and explainable**, but not fully telemetry-accurate. To push it closer to a real engineering tool, the next upgrades should be:

1. Expand the training corpus beyond the current cached subset.
2. Add true weather joins per lap instead of nominal event temperatures.
3. Improve the Silverstone/Las Vegas data balance in the training set.
4. Calibrate tyre degradation from full stint traces and safety-car windows.
5. Replace estimated corners on future-only tracks with telemetry or official geometry once available.
6. Add a retrieval layer for technical regulations, tyres, and event notes.

## Sources used

The hackathon brief and season data were aligned to the following sources:

- `Hacknite_AIML.pdf` from your local brief.
- Formula 1 and FIA 2026 calendar announcement, published June 10, 2025: [Formula 1 and FIA announce 2026 calendar](https://corp.formula1.com/formula-1-and-fia-announce-2026-calendar/)
- Official 2026 schedule/results page, accessed April 5, 2026: [F1 Schedule 2026](https://www.formula1.com/en/racing/2026)
- Official Australian pole report, published March 2026: [Russell pole lap in Australia](https://www.formula1.com/en/latest/article/watch-ride-onboard-with-russell-for-his-pole-position-lap-at-the-australian.MkhniVu6PEdq8o0GIeFeC/)
- Official Chinese starting grid/results data, published March 2026: [Chinese GP starting grid](https://www.formula1.com/en/results/2026/races/1280/china/starting-grid)
- Official Japanese results data, published March 2026: [Japanese GP race result](https://www.formula1.com/en/results/2026/races/1281/japan/race-result)
- Formula 1 2026 regulations terminology update, published January 2026: [F1 2026 regulations terminology update](https://corp.formula1.com/f1-2026-regulations-terminology-update/)
- F1 writers’ early-season team assessment, accessed April 4-5, 2026: [Writers reflect on the first three rounds of 2026](https://www.formula1.com/en/latest/article/star-drivers-biggest-surprises-and-who-has-work-to-do-our-writers-reflect-on.1uYVp8xyyEJNgBersLH087.1uYVp8xyyEJNgBersLH087/)

## Assumptions

- Data is calibrated to the first three completed rounds as of **April 5, 2026**.
- Team strengths are directional public-data priors, not private telemetry.
- Track maps are generated from public geometry, with telemetry-backed corners for the completed 2026 rounds and estimated corners where session geometry is unavailable.
- Strategy outputs are meant to be plausible baseline recommendations for a hackathon demo.
