from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import json
import pandas as pd

from .data import MODEL_DIR, TeamProfile, TrackProfile


class LapTimeRegressor:
    def __init__(self) -> None:
        self.model_path = MODEL_DIR / "lap_time_regressor.joblib"
        self.race_model_path = MODEL_DIR / "lap_time_race_regressor.joblib"
        self.pipeline: Any | None = None
        self.race_pipeline: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.race_metadata: dict[str, Any] = {}
        self.available = False
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
            self._configure_inference_runtime()
            meta_path = MODEL_DIR / "lap_regressor_metadata.json"
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
            self.available = True
        if self.race_model_path.exists():
            self.race_pipeline = joblib.load(self.race_model_path)
            self._configure_inference_runtime(self.race_pipeline)
            race_meta_path = MODEL_DIR / "lap_race_regressor_metadata.json"
            if race_meta_path.exists():
                with race_meta_path.open("r", encoding="utf-8") as handle:
                    self.race_metadata = json.load(handle)

    def predict_qualifying_lap(
        self,
        team: TeamProfile,
        track: TrackProfile,
        *,
        ambient_temp_c: float,
        wind_kph: float,
        push_laps: int,
        fuel_load_kg: float,
    ) -> float | None:
        if not self.available:
            return None
        row = self._base_row(team, track)
        row.update(
            {
                "session_code": "Q",
                "compound_key": "soft",
                "tyre_life": float(max(push_laps, 1)),
                "lap_number_norm": 0.16,
                "stint": 1.0,
                "track_status_code": 1.0,
                "is_fresh_tyre": 1.0,
                "position": 1.0,
                "ambient_temp_c": float(ambient_temp_c),
                "wind_kph": float(wind_kph),
                "fuel_load_kg": float(fuel_load_kg),
                "traffic_penalty_sec": 0.0,
            }
        )
        return float(self.pipeline.predict(self._frame(row, self.metadata))[0])

    def predict_race_lap(
        self,
        team: TeamProfile,
        track: TrackProfile,
        *,
        compound: str,
        tyre_life: float,
        lap_number_norm: float,
        stint: int,
        position: int,
        ambient_temp_c: float,
        wind_kph: float,
        fuel_load_kg: float,
        traffic_penalty_sec: float,
    ) -> float | None:
        if not self.available and self.race_pipeline is None:
            return None
        row = self._base_row(team, track)
        row.update(
            {
                "session_code": "R",
                "compound_key": compound,
                "tyre_life": float(tyre_life),
                "lap_number_norm": float(lap_number_norm),
                "stint": float(stint),
                "track_status_code": 1.0,
                "is_fresh_tyre": 1.0 if tyre_life <= 2 else 0.0,
                "position": float(position),
                "ambient_temp_c": float(ambient_temp_c),
                "wind_kph": float(wind_kph),
                "fuel_load_kg": float(fuel_load_kg),
                "traffic_penalty_sec": float(traffic_penalty_sec),
            }
        )
        pipeline = self.race_pipeline or self.pipeline
        metadata = self.race_metadata or self.metadata
        return float(pipeline.predict(self._frame(row, metadata))[0])

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            **self.metadata,
            "race_model_available": self.race_pipeline is not None,
            "race_model_metrics": {
                "mae_sec": self.race_metadata.get("mae_sec"),
                "rmse_sec": self.race_metadata.get("rmse_sec"),
                "r2": self.race_metadata.get("r2"),
            },
        }

    def _configure_inference_runtime(self, pipeline: Any | None = None) -> None:
        target_pipeline = pipeline or self.pipeline
        if target_pipeline is None:
            return
        model = getattr(target_pipeline, "named_steps", {}).get("model")
        self._force_single_thread(model)

    def _force_single_thread(self, estimator: Any | None) -> None:
        if estimator is None:
            return
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        for nested in getattr(estimator, "estimators", []):
            if isinstance(nested, tuple):
                _, nested_estimator = nested
            else:
                nested_estimator = nested
            self._force_single_thread(nested_estimator)

    def _base_row(self, team: TeamProfile, track: TrackProfile) -> dict[str, float]:
        row = {
            "track_key": track.key,
            "team_key": team.key,
            "year": 2026.0,
            "track_length_km": track.length_km,
            "track_corners": float(track.corners),
            "track_drs_zones": float(track.drs_zones),
            "track_lap_count": float(track.lap_count),
            "fuel_sensitivity": track.fuel_sensitivity_sec_per_kg,
            "fuel_burn_per_lap": track.fuel_burn_kg_per_lap,
            "tyre_stress": track.tyre_stress,
            "overtaking_risk": track.overtaking_risk,
            "dirty_air_penalty": track.dirty_air_penalty_sec,
            "track_evolution": track.track_evolution,
            "braking_harvest": track.braking_harvest_potential,
            "traction_demand": track.traction_demand,
            "high_speed_bias": track.high_speed_bias,
            "straightline_demand": track.straightline_demand,
            "nominal_temp_c": float(track.nominal_temp_c),
            "team_one_lap_delta": team.one_lap_delta_sec,
            "team_race_pace_delta": team.race_pace_delta_sec,
            "team_tyre_management": team.tyre_management,
            "team_energy_efficiency": team.energy_efficiency,
            "team_drag_efficiency": team.drag_efficiency,
            "team_traction": team.traction,
            "team_high_speed": team.high_speed,
            "team_dirty_air_resilience": team.dirty_air_resilience,
            "team_reliability": team.reliability,
            "team_development_trend": team.development_trend,
        }
        return row

    def _frame(self, row: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
        columns = metadata.get("feature_columns")
        if columns:
            ordered = {column: row.get(column) for column in columns}
            return pd.DataFrame([ordered], columns=columns)
        return pd.DataFrame([row])
