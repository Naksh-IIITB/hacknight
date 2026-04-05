from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
import requests

from f1_ai.data import DATA_DIR, load_teams, load_track_maps, load_tracks


CACHE_DIR = DATA_DIR / "fastf1_cache"
LAP_DATA_PATH = DATA_DIR / "historical_laps_2025_2026.csv"
TRACK_MAPS_PATH = DATA_DIR / "track_maps_2026.json"

TEAM_ALIASES = {
    "Mercedes": "mercedes",
    "Ferrari": "ferrari",
    "McLaren": "mclaren",
    "Red Bull Racing": "red_bull",
    "Red Bull": "red_bull",
    "Oracle Red Bull Racing": "red_bull",
    "Alpine": "alpine",
    "Haas F1 Team": "haas",
    "Haas": "haas",
    "Kick Sauber": "audi",
    "Stake F1 Team Kick Sauber": "audi",
    "Sauber": "audi",
    "Racing Bulls": "racing_bulls",
    "RB": "racing_bulls",
    "Visa Cash App RB": "racing_bulls",
    "Cadillac": "cadillac",
    "Williams": "williams",
    "Aston Martin": "aston_martin",
    "Aston Martin Aramco": "aston_martin",
}

TRACK_EVENT_NAMES = {
    "australia": "Australia",
    "china": "China",
    "japan": "Japan",
    "bahrain": "Bahrain",
    "saudi_arabia": "Saudi Arabia",
    "miami": "Miami",
    "canada": "Canada",
    "monaco": "Monaco",
    "barcelona": "Spain",
    "austria": "Austria",
    "great_britain": "British Grand Prix",
    "belgium": "Belgium",
    "hungary": "Hungary",
    "netherlands": "Netherlands",
    "italy": "Italian Grand Prix",
    "azerbaijan": "Azerbaijan",
    "singapore": "Singapore",
    "united_states": "United States",
    "mexico": "Mexico City",
    "brazil": "Sao Paulo",
    "las_vegas": "Las Vegas",
    "qatar": "Qatar",
    "abu_dhabi": "Abu Dhabi",
}

MAP_SESSION_SOURCE = {
    "australia": (2026, "Q"),
    "china": (2026, "Q"),
    "japan": (2026, "Q"),
}
FASTF1_MAP_TRACKS = {"australia", "china", "japan"}
TRAINING_TRACK_KEYS = {
    "australia",
    "china",
    "japan",
    "bahrain",
    "saudi_arabia",
    "miami",
    "monaco",
    "great_britain",
    "belgium",
    "italy",
    "singapore",
    "las_vegas",
}

TRACK_GEOJSON_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
TRACK_GEOJSON_NAMES = {
    "australia": "Albert Park Circuit",
    "china": "Shanghai International Circuit",
    "japan": "Suzuka International Racing Course",
    "bahrain": "Bahrain International Circuit",
    "saudi_arabia": "Jeddah Corniche Circuit",
    "miami": "Miami International Autodrome",
    "canada": "Circuit Gilles-Villeneuve",
    "monaco": "Circuit de Monaco",
    "barcelona": "Circuit de Barcelona-Catalunya",
    "austria": "Red Bull Ring",
    "great_britain": "Silverstone Circuit",
    "belgium": "Circuit de Spa-Francorchamps",
    "hungary": "Hungaroring",
    "netherlands": "Circuit Zandvoort",
    "italy": "Autodromo Nazionale Monza",
    "madrid": "Circuito de Madring",
    "azerbaijan": "Baku City Circuit",
    "singapore": "Marina Bay Street Circuit",
    "united_states": "Circuit of the Americas",
    "mexico": "Autódromo Hermanos Rodríguez",
    "brazil": "Autódromo José Carlos Pace - Interlagos",
    "las_vegas": "Las Vegas Street Circuit",
    "qatar": "Losail International Circuit",
    "abu_dhabi": "Yas Marina Circuit",
}


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    tracks = load_tracks()
    teams = load_teams()
    existing_maps = load_track_maps()

    lap_frames: list[pd.DataFrame] = []
    map_payload: list[dict[str, Any]] = []

    for key, track in tracks.items():
        if key in TRAINING_TRACK_KEYS:
            lap_frames.extend(_collect_track_sessions(key, track, teams))
        try:
            map_payload.append(_build_track_map(key, track))
        except Exception as exc:
            print(f"reuse existing map for {key}: {exc}")
            if key in existing_maps:
                map_payload.append(existing_maps[key])
            else:
                map_payload.append(_offline_fallback_map(key, track))

    if lap_frames:
        combined = pd.concat(lap_frames, ignore_index=True)
        combined.to_csv(LAP_DATA_PATH, index=False)
        print(f"wrote lap data: {LAP_DATA_PATH} rows={len(combined)}")

    with TRACK_MAPS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(map_payload, handle, indent=2)
    print(f"wrote track maps: {TRACK_MAPS_PATH} tracks={len(map_payload)}")


def _offline_fallback_map(track_key: str, track: Any) -> dict[str, Any]:
    points = [
        {"x": 0.18, "y": 0.25},
        {"x": 0.72, "y": 0.22},
        {"x": 0.84, "y": 0.52},
        {"x": 0.65, "y": 0.8},
        {"x": 0.28, "y": 0.78},
        {"x": 0.12, "y": 0.5},
        {"x": 0.18, "y": 0.25},
    ]
    corners = []
    for index in range(track.corners):
        point = points[index % (len(points) - 1)]
        corners.append(
            {
                "x": point["x"],
                "y": point["y"],
                "number": index + 1,
                "letter": "",
                "angle": 0.0,
            }
        )
    return {
        "key": track_key,
        "source": "offline-fallback",
        "polyline": points,
        "corners": corners,
        "metadata": {"circuit": track.circuit, "grand_prix": track.grand_prix, "estimated_corners": True},
    }


def _collect_track_sessions(track_key: str, track: Any, teams: dict[str, Any]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    years = [2025]
    if track_key in {"australia", "china", "japan"}:
        years.append(2026)

    for year in years:
        for session_code in ("Q", "R"):
            try:
                session = fastf1.get_session(year, TRACK_EVENT_NAMES[track_key], session_code)
                session.load(laps=True, telemetry=False, weather=False, messages=False)
            except Exception as exc:
                print(f"skip session {year} {track_key} {session_code}: {exc}")
                continue

            laps = session.laps.copy()
            if laps.empty:
                continue
            laps = laps[
                laps["LapTime"].notna()
                & laps["IsAccurate"].fillna(False)
                & ~laps["PitOutTime"].notna()
                & ~laps["PitInTime"].notna()
                & laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])
            ].copy()
            if laps.empty:
                continue
            if "Deleted" in laps:
                laps = laps[~laps["Deleted"].fillna(False)].copy()

            mapped = laps["Team"].map(TEAM_ALIASES)
            laps = laps[mapped.notna()].copy()
            laps["team_key"] = mapped[mapped.notna()]
            laps["track_key"] = track_key
            laps["year"] = year
            laps["session_code"] = session_code
            laps["lap_time_sec"] = laps["LapTime"].dt.total_seconds()
            laps["compound_key"] = laps["Compound"].str.lower()
            laps["track_status_code"] = (
                laps["TrackStatus"].astype(str).str.extract(r"(\d)").fillna("1").astype(float)
            )
            laps["tyre_life"] = laps["TyreLife"].fillna(1).astype(float)
            laps["lap_number_norm"] = laps["LapNumber"].astype(float) / float(track.lap_count)
            laps["stint"] = laps["Stint"].fillna(1).astype(float)
            laps["is_fresh_tyre"] = laps["FreshTyre"].fillna(False).astype(int).astype(float)
            laps["position"] = laps["Position"].fillna(10).astype(float)
            laps["ambient_temp_c"] = float(track.nominal_temp_c)
            laps["wind_kph"] = 8.0
            laps["fuel_load_kg"] = _estimate_fuel_load(track, laps["LapNumber"].astype(float), session_code)
            laps["traffic_penalty_sec"] = 0.0

            for feature_name, values in _static_track_features(track).items():
                laps[feature_name] = values

            for feature_name, attr_name in _numeric_team_features(teams).items():
                laps[feature_name] = laps["team_key"].map(lambda key: getattr(teams[key], attr_name))

            frames.append(
                laps[
                    [
                        "lap_time_sec",
                        "track_key",
                        "team_key",
                        "year",
                        "session_code",
                        "compound_key",
                        "tyre_life",
                        "lap_number_norm",
                        "stint",
                        "track_status_code",
                        "is_fresh_tyre",
                        "position",
                        "ambient_temp_c",
                        "wind_kph",
                        "fuel_load_kg",
                        "traffic_penalty_sec",
                        *list(_static_track_features(track).keys()),
                        *list(_numeric_team_features(teams).keys()),
                    ]
                ]
            )
            print(f"loaded {year} {track_key} {session_code}: rows={len(laps)}")

    return frames


def _static_track_features(track: Any) -> dict[str, float]:
    return {
        "track_length_km": float(track.length_km),
        "track_corners": float(track.corners),
        "track_drs_zones": float(track.drs_zones),
        "track_lap_count": float(track.lap_count),
        "fuel_sensitivity": float(track.fuel_sensitivity_sec_per_kg),
        "fuel_burn_per_lap": float(track.fuel_burn_kg_per_lap),
        "tyre_stress": float(track.tyre_stress),
        "overtaking_risk": float(track.overtaking_risk),
        "dirty_air_penalty": float(track.dirty_air_penalty_sec),
        "track_evolution": float(track.track_evolution),
        "braking_harvest": float(track.braking_harvest_potential),
        "traction_demand": float(track.traction_demand),
        "high_speed_bias": float(track.high_speed_bias),
        "straightline_demand": float(track.straightline_demand),
        "nominal_temp_c": float(track.nominal_temp_c),
    }


def _numeric_team_features(teams: dict[str, Any]) -> dict[str, str]:
    first = next(iter(teams.values()))
    return {
        "team_one_lap_delta": "one_lap_delta_sec",
        "team_race_pace_delta": "race_pace_delta_sec",
        "team_tyre_management": "tyre_management",
        "team_energy_efficiency": "energy_efficiency",
        "team_drag_efficiency": "drag_efficiency",
        "team_traction": "traction",
        "team_high_speed": "high_speed",
        "team_dirty_air_resilience": "dirty_air_resilience",
        "team_reliability": "reliability",
        "team_development_trend": "development_trend",
    }


def _estimate_fuel_load(track: Any, lap_number: pd.Series, session_code: str) -> pd.Series:
    if session_code == "Q":
        return pd.Series([3.5 + track.length_km * 0.48] * len(lap_number), index=lap_number.index)
    remaining_laps = (track.lap_count - lap_number).clip(lower=0)
    return remaining_laps * track.fuel_burn_kg_per_lap


def _build_track_map(track_key: str, track: Any) -> dict[str, Any]:
    try:
        if track_key == "madrid":
            return _build_madrid_map(track_key, track)
        if track_key not in FASTF1_MAP_TRACKS:
            return _build_geojson_map(track_key, track)
        year, session_code = MAP_SESSION_SOURCE.get(track_key, (2025, "Q"))
        session = fastf1.get_session(year, TRACK_EVENT_NAMES[track_key], session_code)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        fastest = session.laps.pick_fastest()
        pos = fastest.get_pos_data()[["X", "Y"]].dropna()
        corners = session.get_circuit_info().corners[["Number", "Letter", "X", "Y", "Angle"]].dropna()
        normalized_points = _normalize_points(pos.to_dict(orient="records"), x_key="X", y_key="Y")
        normalized_corners = _normalize_points(corners.to_dict(orient="records"), x_key="X", y_key="Y")
        for index, corner in enumerate(normalized_corners):
            raw = corners.iloc[index]
            corner.update(
                {
                    "number": int(raw["Number"]),
                    "letter": str(raw["Letter"] or ""),
                    "angle": round(float(raw["Angle"]), 2),
                }
            )
        return {
            "key": track_key,
            "source": f"fastf1:{year}:{session_code}",
            "polyline": normalized_points[::2],
            "corners": normalized_corners,
            "metadata": {"circuit": track.circuit, "grand_prix": track.grand_prix},
        }
    except Exception as exc:
        print(f"fallback map for {track_key}: {exc}")
        return _build_geojson_map(track_key, track)


def _build_madrid_map(track_key: str, track: Any) -> dict[str, Any]:
    geo = _load_geojson_feature(TRACK_GEOJSON_NAMES[track_key])
    polyline = _extract_geojson_polyline(geo)
    estimated_corners = _estimate_corners_from_polyline(polyline, target_count=track.corners)
    return {
        "key": track_key,
        "source": "geojson:bacinger/f1-circuits",
        "polyline": polyline,
        "corners": estimated_corners,
        "metadata": {"circuit": track.circuit, "grand_prix": track.grand_prix, "estimated_corners": True},
    }


def _build_geojson_map(track_key: str, track: Any) -> dict[str, Any]:
    feature = _load_geojson_feature(TRACK_GEOJSON_NAMES.get(track_key, track.circuit))
    polyline = _extract_geojson_polyline(feature)
    estimated_corners = _estimate_corners_from_polyline(polyline, target_count=track.corners)
    return {
        "key": track_key,
        "source": "geojson:bacinger/f1-circuits",
        "polyline": polyline,
        "corners": estimated_corners,
        "metadata": {"circuit": track.circuit, "grand_prix": track.grand_prix, "estimated_corners": True},
    }


def _load_geojson_feature(circuit_name: str) -> dict[str, Any]:
    response = requests.get(TRACK_GEOJSON_URL, timeout=30)
    response.raise_for_status()
    features = response.json()["features"]
    for feature in features:
        name = feature["properties"]["Name"]
        if circuit_name.lower() in name.lower() or name.lower() in circuit_name.lower():
            return feature
    raise KeyError(f"No GeoJSON circuit match for {circuit_name}")


def _extract_geojson_polyline(feature: dict[str, Any]) -> list[dict[str, float]]:
    geometry = feature["geometry"]
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        coords = coords[0]
    points = [{"X": lon, "Y": lat} for lon, lat in coords]
    normalized = _normalize_points(points, x_key="X", y_key="Y")
    return normalized


def _normalize_points(points: list[dict[str, Any]], *, x_key: str, y_key: str) -> list[dict[str, float]]:
    xs = [float(item[x_key]) for item in points]
    ys = [float(item[y_key]) for item in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    normalized = []
    for item in points:
        normalized.append(
            {
                "x": round((float(item[x_key]) - min_x) / span_x, 5),
                "y": round(1.0 - ((float(item[y_key]) - min_y) / span_y), 5),
            }
        )
    return normalized


def _estimate_corners_from_polyline(polyline: list[dict[str, float]], *, target_count: int) -> list[dict[str, Any]]:
    if len(polyline) < 5:
        return []
    scores: list[tuple[float, int]] = []
    for idx in range(2, len(polyline) - 2):
        p0, p1, p2 = polyline[idx - 1], polyline[idx], polyline[idx + 1]
        v1 = (p1["x"] - p0["x"], p1["y"] - p0["y"])
        v2 = (p2["x"] - p1["x"], p2["y"] - p1["y"])
        angle = abs(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
        angle = min(angle, 2 * math.pi - angle)
        scores.append((angle, idx))
    scores.sort(reverse=True)
    selected = sorted(idx for _, idx in scores[:target_count])
    corners = []
    for number, idx in enumerate(selected, start=1):
        point = polyline[idx]
        corners.append({"x": point["x"], "y": point["y"], "number": number, "letter": "", "angle": 0.0})
    return corners


if __name__ == "__main__":
    main()
