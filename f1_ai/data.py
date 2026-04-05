from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


@dataclass(frozen=True)
class TrackProfile:
    key: str
    grand_prix: str
    country: str
    circuit: str
    round: int
    race_weekend: str
    length_km: float
    corners: int
    drs_zones: int
    qualifying_reference_sec: float
    lap_count: int
    fuel_sensitivity_sec_per_kg: float
    fuel_burn_kg_per_lap: float
    tyre_stress: float
    overtaking_risk: float
    dirty_air_penalty_sec: float
    track_evolution: float
    braking_harvest_potential: float
    traction_demand: float
    high_speed_bias: float
    straightline_demand: float
    nominal_temp_c: int
    sprint_weekend: bool = False


@dataclass(frozen=True)
class TeamProfile:
    key: str
    name: str
    one_lap_delta_sec: float
    race_pace_delta_sec: float
    tyre_management: float
    energy_efficiency: float
    drag_efficiency: float
    traction: float
    high_speed: float
    dirty_air_resilience: float
    reliability: float
    development_trend: float
    notes: str


@dataclass(frozen=True)
class DriverProfile:
    number: int
    code: str
    name: str
    team_key: str
    team_name: str
    experience_tier: str
    qualifying_delta_sec: float
    racecraft: float
    tyre_management: float
    wet_confidence: float


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tracks() -> dict[str, TrackProfile]:
    tracks = _load_json(DATA_DIR / "tracks_2026.json")
    return {item["key"]: TrackProfile(**item) for item in tracks}


def load_teams() -> dict[str, TeamProfile]:
    teams = _load_json(DATA_DIR / "teams_2026.json")
    return {item["key"]: TeamProfile(**item) for item in teams}


def load_drivers() -> list[DriverProfile]:
    drivers = _load_json(DATA_DIR / "drivers_2026.json")
    return [DriverProfile(**item) for item in drivers]


def load_track_maps() -> dict[str, dict[str, Any]]:
    path = DATA_DIR / "track_maps_2026.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    return {item["key"]: item for item in data}


def load_model_metadata() -> dict[str, Any]:
    path = MODEL_DIR / "lap_regressor_metadata.json"
    if not path.exists():
        return {}
    return _load_json(path)
