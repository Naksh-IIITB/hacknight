from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .data import DriverProfile, TeamProfile, TrackProfile, load_drivers, load_model_metadata, load_teams, load_tracks
from .maps import TrackMapStore
from .ml import LapTimeRegressor


class F1Predictor:
    """Explainable hybrid simulator tuned for early-2026 F1 public data."""

    def __init__(self) -> None:
        self.tracks = load_tracks()
        self.teams = load_teams()
        self.drivers = load_drivers()
        self.track_maps = TrackMapStore()
        self.regressor = LapTimeRegressor()
        self.model_metadata = load_model_metadata()

    def list_tracks(self) -> list[str]:
        return sorted(self.tracks)

    def list_teams(self) -> list[str]:
        return sorted(self.teams)

    def list_drivers(self) -> list[dict[str, Any]]:
        return [
            {
                "number": driver.number,
                "code": driver.code,
                "name": driver.name,
                "team_key": driver.team_key,
                "team_name": driver.team_name,
            }
            for driver in sorted(self.drivers, key=lambda item: item.number)
        ]

    def list_drivers_by_team(self, team_key: str) -> list[dict[str, Any]]:
        return [driver for driver in self.list_drivers() if driver["team_key"] == team_key]

    def compare_drivers(
        self,
        left_team_key: str,
        left_track_key: str,
        *,
        left_driver_number: int | None = None,
        right_team_key: str,
        right_track_key: str,
        right_driver_number: int | None = None,
        ambient_temp_c: float | None = None,
        wind_kph: float = 8.0,
        push_laps: int = 1,
        starting_position: int = 6,
        traffic_level: float = 0.5,
    ) -> dict[str, Any]:
        left_qual = self.predict_qualifying(
            left_team_key,
            left_track_key,
            driver_number=left_driver_number,
            ambient_temp_c=ambient_temp_c,
            wind_kph=wind_kph,
            push_laps=push_laps,
        )
        right_qual = self.predict_qualifying(
            right_team_key,
            right_track_key,
            driver_number=right_driver_number,
            ambient_temp_c=ambient_temp_c,
            wind_kph=wind_kph,
            push_laps=push_laps,
        )
        left_race = self.simulate_race(
            left_team_key,
            left_track_key,
            driver_number=left_driver_number,
            ambient_temp_c=ambient_temp_c,
            starting_position=starting_position,
            traffic_level=traffic_level,
            wind_kph=wind_kph,
        )
        right_race = self.simulate_race(
            right_team_key,
            right_track_key,
            driver_number=right_driver_number,
            ambient_temp_c=ambient_temp_c,
            starting_position=starting_position,
            traffic_level=traffic_level,
            wind_kph=wind_kph,
        )
        return {
            "left": {"qualifying": left_qual, "race": left_race},
            "right": {"qualifying": right_qual, "race": right_race},
            "delta": {
                "qualifying_sec": round(right_qual["predicted_lap_time_sec"] - left_qual["predicted_lap_time_sec"], 3),
                "race_total_sec": round(right_race["total_race_time_sec"] - left_race["total_race_time_sec"], 3),
            },
        }

    def get_track_map(self, track_key: str) -> dict[str, Any]:
        return self.track_maps.get(track_key)

    def model_status(self) -> dict[str, Any]:
        return self.regressor.status()

    def predict_qualifying(
        self,
        team_key: str,
        track_key: str,
        driver_number: int | None = None,
        ambient_temp_c: float | None = None,
        wind_kph: float = 8.0,
        push_laps: int = 1,
    ) -> dict[str, Any]:
        team = self._team(team_key)
        track = self._track(track_key)
        driver = self._driver(driver_number, team_key)
        temp = float(ambient_temp_c if ambient_temp_c is not None else track.nominal_temp_c)

        fuel_load = self._suggest_qualifying_fuel(track, push_laps)
        sector_fit = (
            (0.77 - team.traction) * track.traction_demand
            + (0.77 - team.high_speed) * track.high_speed_bias
            + (0.77 - team.drag_efficiency) * track.straightline_demand
        ) * 4.4
        temperature_penalty = abs(temp - track.nominal_temp_c) * (1.18 - team.tyre_management) * 0.015
        wind_penalty = wind_kph * (0.35 + track.straightline_demand) * (1.0 - team.drag_efficiency) * 0.01
        fuel_penalty = fuel_load * track.fuel_sensitivity_sec_per_kg
        evolution_gain = -0.08 - 0.08 * track.track_evolution
        driver_adjustment = self._driver_qualifying_adjustment(driver, track, temp)

        heuristic_time = (
            track.qualifying_reference_sec
            + team.one_lap_delta_sec
            + driver_adjustment
            + sector_fit
            + temperature_penalty
            + wind_penalty
            + fuel_penalty
            + evolution_gain
        )
        ml_time = self.regressor.predict_qualifying_lap(
            team,
            track,
            ambient_temp_c=temp,
            wind_kph=wind_kph,
            push_laps=push_laps,
            fuel_load_kg=fuel_load,
        )
        if ml_time is not None:
            if self.regressor.qualifying_pipeline is not None:
                predicted_time = heuristic_time * 0.28 + ml_time * 0.72
                model_blend = "heuristic 28% / qualifying ML 72%"
            else:
                predicted_time = heuristic_time * 0.58 + ml_time * 0.42
                model_blend = "heuristic 58% / historical ML 42%"
        else:
            predicted_time = heuristic_time
            model_blend = "heuristic only"

        return {
            "team": team.name,
            "driver": self._driver_label(driver),
            "track": track.grand_prix,
            "predicted_lap_time_sec": round(predicted_time, 3),
            "predicted_lap_time": self._fmt(predicted_time),
            "heuristic_lap_time_sec": round(heuristic_time, 3),
            "ml_lap_time_sec": round(ml_time, 3) if ml_time is not None else None,
            "model_blend": model_blend,
            "suggested_fuel_load_kg": round(fuel_load, 1),
            "ers_plan": self._ers_plan(team, track),
            "explanation": {
                "baseline_track_reference": self._fmt(track.qualifying_reference_sec),
                "team_one_lap_delta_sec": round(team.one_lap_delta_sec, 3),
                "driver_delta_sec": round(driver_adjustment, 3),
                "sector_fit_sec": round(sector_fit, 3),
                "temperature_penalty_sec": round(temperature_penalty, 3),
                "wind_penalty_sec": round(wind_penalty, 3),
                "fuel_penalty_sec": round(fuel_penalty, 3),
                "track_evolution_gain_sec": round(evolution_gain, 3),
            },
        }

    def simulate_race(
        self,
        team_key: str,
        track_key: str,
        driver_number: int | None = None,
        ambient_temp_c: float | None = None,
        starting_position: int = 6,
        traffic_level: float = 0.5,
        wind_kph: float = 8.0,
        include_expected_result: bool = True,
    ) -> dict[str, Any]:
        team = self._team(team_key)
        track = self._track(track_key)
        driver = self._driver(driver_number, team_key)
        temp = float(ambient_temp_c if ambient_temp_c is not None else track.nominal_temp_c)

        candidates = [
            ["medium", "hard"],
            ["hard", "medium"],
            ["soft", "medium", "hard"],
            ["medium", "medium", "hard"],
        ]

        evaluations = [
            self._evaluate_strategy(team, track, driver, temp, starting_position, traffic_level, wind_kph, candidate)
            for candidate in candidates
        ]
        best = min(evaluations, key=lambda item: item["total_race_time_sec"])

        best["team_notes"] = team.notes
        best["driver"] = self._driver_label(driver)
        if include_expected_result:
            best["expected_race_result"] = self._estimate_finish_position(
                team_key=team_key,
                track_key=track_key,
                driver_number=driver.number if driver else None,
                ambient_temp_c=temp,
                wind_kph=wind_kph,
            )
        best["track_model"] = {
            "dirty_air_penalty_sec_per_lap": round(track.dirty_air_penalty_sec, 3),
            "overtaking_risk": round(track.overtaking_risk, 2),
            "tyre_stress": round(track.tyre_stress, 2),
            "ml_available": self.regressor.available,
        }
        return best

    def build_team_strategy_board(self, track_key: str) -> list[dict[str, Any]]:
        board = []
        for team_key in self.list_teams():
            result = self.simulate_race(team_key, track_key, include_expected_result=False)
            board.append(
                {
                    "team": result["team"],
                    "recommended_strategy": result["recommended_strategy"],
                    "predicted_total_time": result["predicted_total_time"],
                    "stint_summary": [f"{stint['compound']} x {stint['laps']}" for stint in result["stints"]],
                    "headline": result["team_notes"],
                }
            )
        return sorted(board, key=lambda item: item["predicted_total_time"])

    def expected_race_order(
        self,
        track_key: str,
        *,
        ambient_temp_c: float | None = None,
        wind_kph: float = 8.0,
    ) -> list[dict[str, Any]]:
        track = self._track(track_key)
        temp = float(ambient_temp_c if ambient_temp_c is not None else track.nominal_temp_c)
        entries = []
        seen_teams: set[str] = set()
        for driver in self.drivers:
            team = self._team(driver.team_key)
            total_time = self._project_race_total(
                team=team,
                track=track,
                driver=driver,
                ambient_temp_c=temp,
                wind_kph=wind_kph,
            )
            entries.append(
                {
                    "position": 0,
                    "driver": driver.name,
                    "number": driver.number,
                    "team": driver.team_name,
                    "predicted_total_time": self._fmt(total_time),
                    "total_race_time_sec": round(total_time, 3),
                }
            )
            seen_teams.add(driver.team_key)
        for team_key in self.list_teams():
            if team_key in seen_teams:
                continue
            team = self._team(team_key)
            total_time = self._project_race_total(
                team=team,
                track=track,
                driver=None,
                ambient_temp_c=temp,
                wind_kph=wind_kph,
            )
            entries.append(
                {
                    "position": 0,
                    "driver": "Team baseline",
                    "number": None,
                    "team": team.name,
                    "predicted_total_time": self._fmt(total_time),
                    "total_race_time_sec": round(total_time, 3),
                }
            )
        entries.sort(key=lambda item: item["total_race_time_sec"])
        for index, item in enumerate(entries, start=1):
            item["position"] = index
        return entries

    def snapshot(self) -> dict[str, Any]:
        return {
            "tracks": [asdict(track) for track in self.tracks.values()],
            "teams": [asdict(team) for team in self.teams.values()],
            "drivers": [asdict(driver) for driver in self.drivers],
        }

    def _evaluate_strategy(
        self,
        team: TeamProfile,
        track: TrackProfile,
        driver: DriverProfile | None,
        temp: float,
        starting_position: int,
        traffic_level: float,
        wind_kph: float,
        compounds: list[str],
    ) -> dict[str, Any]:
        stints = self._split_stints(track.lap_count, len(compounds), track.tyre_stress)
        race_start_fuel = track.lap_count * track.fuel_burn_kg_per_lap
        total_time = 0.0
        stint_outputs = []

        for index, (compound, laps) in enumerate(zip(compounds, stints), start=1):
            laps_completed_before = sum(stints[: index - 1])
            average_fuel = max(race_start_fuel - (laps_completed_before + laps / 2.0) * track.fuel_burn_kg_per_lap, 0.0)
            clean_air_factor = max(starting_position - 1, 0) / 19.0 if index == 1 else 0.15
            traffic_penalty = (
                track.dirty_air_penalty_sec
                * (1.08 - team.dirty_air_resilience)
                * (traffic_level + clean_air_factor)
            )
            avg_lap = self._average_race_lap(
                team,
                track,
                driver,
                compound,
                laps,
                average_fuel,
                temp,
                traffic_penalty,
                wind_kph,
                stint_number=index,
                average_position=max(1, starting_position - (index - 1)),
                average_lap_number=(laps_completed_before + laps / 2.0) / track.lap_count,
            )
            stint_time = avg_lap * laps
            total_time += stint_time
            if index < len(compounds):
                total_time += self._pit_loss(track, compound, compounds[index])
            stint_outputs.append(
                {
                    "compound": compound,
                    "laps": laps,
                    "average_lap_time_sec": round(avg_lap, 3),
                    "average_lap_time": self._fmt(avg_lap),
                    "stint_time_sec": round(stint_time, 3),
                }
            )

        return {
            "team": team.name,
            "track": track.grand_prix,
            "recommended_strategy": "-".join(item[0].upper() for item in compounds),
            "predicted_total_time": self._fmt(total_time),
            "total_race_time_sec": round(total_time, 3),
            "stints": stint_outputs,
        }

    def _average_race_lap(
        self,
        team: TeamProfile,
        track: TrackProfile,
        driver: DriverProfile | None,
        compound: str,
        laps: int,
        fuel_mass: float,
        temp: float,
        traffic_penalty: float,
        wind_kph: float,
        *,
        stint_number: int,
        average_position: int,
        average_lap_number: float,
    ) -> float:
        compound_offset = {"soft": -0.75, "medium": 0.0, "hard": 0.58}[compound]
        degradation_scale = {"soft": 1.0, "medium": 0.72, "hard": 0.56}[compound]
        degradation = (
            track.tyre_stress
            * degradation_scale
            * (1.12 - team.tyre_management)
            * (1.0 + max(temp - track.nominal_temp_c, 0) * 0.01)
        )
        mean_deg = degradation * max(laps - 1, 1) * 0.45
        fuel_effect = fuel_mass * track.fuel_sensitivity_sec_per_kg * 0.45
        driver_adjustment = self._driver_race_adjustment(driver, track, compound, temp, traffic_penalty)
        chassis_fit = (
            (0.77 - team.traction) * track.traction_demand
            + (0.77 - team.high_speed) * track.high_speed_bias
            + (0.77 - team.drag_efficiency) * track.straightline_demand
        ) * 2.8
        heuristic = (
            track.qualifying_reference_sec
            + 3.8
            + team.race_pace_delta_sec
            + compound_offset
            + mean_deg
            + fuel_effect
            + driver_adjustment
            + chassis_fit
            + traffic_penalty
        )

        ml_prediction = self.regressor.predict_race_lap(
            team,
            track,
            compound=compound,
            tyre_life=max(laps / 2.0, 1.0),
            lap_number_norm=average_lap_number,
            stint=stint_number,
            position=average_position,
            ambient_temp_c=temp,
            wind_kph=wind_kph,
            fuel_load_kg=fuel_mass,
            traffic_penalty_sec=traffic_penalty,
        )
        if ml_prediction is None:
            return heuristic
        return heuristic * 0.64 + ml_prediction * 0.36

    def _pit_loss(self, track: TrackProfile, current_compound: str, next_compound: str) -> float:
        delta = 18.5 + track.length_km * 0.55 + track.overtaking_risk * 2.0
        if current_compound == next_compound:
            delta += 0.4
        return delta

    def _project_race_total(
        self,
        *,
        team: TeamProfile,
        track: TrackProfile,
        driver: DriverProfile | None,
        ambient_temp_c: float,
        wind_kph: float,
    ) -> float:
        average_fuel = track.lap_count * track.fuel_burn_kg_per_lap * 0.48
        traffic_penalty = track.dirty_air_penalty_sec * (1.08 - team.dirty_air_resilience) * 0.42
        average_lap = self._average_race_lap(
            team,
            track,
            driver,
            "medium",
            max(round(track.lap_count * 0.46), 1),
            average_fuel,
            ambient_temp_c,
            traffic_penalty,
            wind_kph,
            stint_number=1,
            average_position=10,
            average_lap_number=0.5,
        )
        strategic_overhead = 18.5 + track.length_km * 0.55 + track.overtaking_risk * 2.0
        return average_lap * track.lap_count + strategic_overhead

    def _split_stints(self, lap_count: int, stint_count: int, tyre_stress: float) -> list[int]:
        if stint_count == 2:
            first = round(lap_count * (0.48 if tyre_stress < 0.72 else 0.42))
            return [first, lap_count - first]
        first = round(lap_count * 0.22)
        second = round(lap_count * 0.36)
        third = lap_count - first - second
        return [first, second, third]

    def _ers_plan(self, team: TeamProfile, track: TrackProfile) -> dict[str, Any]:
        deploy_share = round(0.52 + (team.energy_efficiency - 0.7) * 0.3 + track.straightline_demand * 0.12, 2)
        harvest_share = round(0.48 + track.braking_harvest_potential * 0.18 - (team.energy_efficiency - 0.7) * 0.15, 2)
        return {
            "regulation_context": "2026 cars use an approximately 50/50 ICE-electric split with a 350kW MGU-K and no MGU-H.",
            "state_of_charge_window": "Keep battery between 42% and 78% through the lap to avoid clipping late on the straights.",
            "deployment_bias": deploy_share,
            "harvest_bias": harvest_share,
            "zones": [
                "Deploy aggressively on the longest straight and on exits that open into full-throttle sections.",
                "Harvest hardest into the biggest braking events and any stop-start complex where rear stability matters more than launch boost.",
                "Use a lighter deploy map in turbulent air when following within ~1s, then release stored energy once the car reaches cleaner air.",
                "Reserve a late-lap energy margin for the final DRS zone or the run from the last major traction zone to the line.",
            ],
        }

    def _suggest_qualifying_fuel(self, track: TrackProfile, push_laps: int) -> float:
        return 2.8 + push_laps * 1.6 + track.length_km * 0.48

    def _fmt(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = seconds - 60 * minutes
        return f"{minutes}:{remainder:06.3f}"

    def _estimate_finish_position(
        self,
        *,
        team_key: str,
        track_key: str,
        driver_number: int | None,
        ambient_temp_c: float,
        wind_kph: float,
    ) -> str:
        order = self.expected_race_order(track_key, ambient_temp_c=ambient_temp_c, wind_kph=wind_kph)
        for item in order:
            if driver_number is not None and item["number"] == driver_number:
                return self._ordinal(item["position"])
            if driver_number is None and item["driver"] == "Team baseline" and item["team"] == self._team(team_key).name:
                return self._ordinal(item["position"])
        return "N/A"

    def _ordinal(self, value: int) -> str:
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def _driver(self, driver_number: int | None, team_key: str) -> DriverProfile | None:
        if driver_number is None:
            return None
        for driver in self.drivers:
            if driver.number == int(driver_number):
                if driver.team_key != team_key:
                    raise ValueError(f"Driver {driver.number} does not belong to team '{team_key}'.")
                return driver
        raise ValueError(f"Unknown driver number '{driver_number}'.")

    def _driver_label(self, driver: DriverProfile | None) -> str:
        return driver.name if driver else "Team baseline"

    def _driver_qualifying_adjustment(
        self,
        driver: DriverProfile | None,
        track: TrackProfile,
        ambient_temp_c: float,
    ) -> float:
        if driver is None:
            return 0.0
        tyre_temp_effect = max(ambient_temp_c - track.nominal_temp_c, 0.0) * (0.8 - driver.tyre_management) * 0.01
        pressure_bonus = -0.015 if driver.experience_tier in {"senior", "veteran"} and track.track_evolution > 0.72 else 0.0
        return driver.qualifying_delta_sec + tyre_temp_effect + pressure_bonus

    def _driver_race_adjustment(
        self,
        driver: DriverProfile | None,
        track: TrackProfile,
        compound: str,
        ambient_temp_c: float,
        traffic_penalty: float,
    ) -> float:
        if driver is None:
            return 0.0
        tyre_term = (0.79 - driver.tyre_management) * track.tyre_stress * {"soft": 0.26, "medium": 0.2, "hard": 0.14}[compound]
        temp_term = max(ambient_temp_c - track.nominal_temp_c, 0.0) * (0.78 - driver.tyre_management) * 0.008
        traffic_term = traffic_penalty * (0.84 - driver.racecraft) * 0.65
        return tyre_term + temp_term + traffic_term

    def _track(self, key: str) -> TrackProfile:
        try:
            return self.tracks[key]
        except KeyError as exc:
            raise KeyError(f"Unknown track '{key}'. Available: {', '.join(self.list_tracks())}") from exc

    def _team(self, key: str) -> TeamProfile:
        try:
            return self.teams[key]
        except KeyError as exc:
            raise KeyError(f"Unknown team '{key}'. Available: {', '.join(self.list_teams())}") from exc
