from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from f1_ai import F1Predictor


app = FastAPI(
    title="F1 2026 Strategy Predictor",
    description="Hybrid qualifying and race-strategy simulator for the 2026 F1 regulations.",
    version="0.2.0",
)

predictor = F1Predictor()
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QualifyingRequest(BaseModel):
    team: str = Field(..., description="Team key, for example mercedes or ferrari.")
    driver_number: int | None = Field(default=None, description="Optional driver number for driver-aware predictions.")
    track: str = Field(..., description="Track key, for example australia or japan.")
    ambient_temp_c: float | None = Field(default=None, description="Optional event temperature.")
    wind_kph: float = 8.0
    push_laps: int = 1


class RaceRequest(BaseModel):
    team: str
    driver_number: int | None = None
    track: str
    ambient_temp_c: float | None = None
    starting_position: int = 6
    traffic_level: float = 0.5
    wind_kph: float = 8.0


class CompareRequest(BaseModel):
    left_team: str
    left_driver_number: int | None = None
    right_team: str
    right_driver_number: int | None = None
    track: str
    ambient_temp_c: float | None = None
    wind_kph: float = 8.0
    push_laps: int = 1
    starting_position: int = 6
    traffic_level: float = 0.5


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "F1 2026 strategy predictor is running."}


@app.get("/meta")
def meta() -> dict:
    return {
        "teams": predictor.list_teams(),
        "tracks": predictor.list_tracks(),
        "drivers": predictor.list_drivers(),
    }


@app.get("/teams/{team}/drivers")
def team_drivers(team: str) -> dict:
    return {"team": team, "drivers": predictor.list_drivers_by_team(team)}


@app.post("/compare/drivers")
def compare_drivers(request: CompareRequest) -> dict:
    return predictor.compare_drivers(
        left_team_key=request.left_team,
        left_track_key=request.track,
        left_driver_number=request.left_driver_number,
        right_team_key=request.right_team,
        right_track_key=request.track,
        right_driver_number=request.right_driver_number,
        ambient_temp_c=request.ambient_temp_c,
        wind_kph=request.wind_kph,
        push_laps=request.push_laps,
        starting_position=request.starting_position,
        traffic_level=request.traffic_level,
    )


@app.get("/model/status")
def model_status() -> dict:
    return predictor.model_status()


@app.get("/tracks/{track}/map")
def track_map(track: str) -> dict:
    return predictor.get_track_map(track)


@app.post("/predict/qualifying")
def predict_qualifying(request: QualifyingRequest) -> dict:
    return predictor.predict_qualifying(
        team_key=request.team,
        track_key=request.track,
        driver_number=request.driver_number,
        ambient_temp_c=request.ambient_temp_c,
        wind_kph=request.wind_kph,
        push_laps=request.push_laps,
    )


@app.post("/simulate/race")
def simulate_race(request: RaceRequest) -> dict:
    return predictor.simulate_race(
        team_key=request.team,
        track_key=request.track,
        driver_number=request.driver_number,
        ambient_temp_c=request.ambient_temp_c,
        starting_position=request.starting_position,
        traffic_level=request.traffic_level,
        wind_kph=request.wind_kph,
    )


@app.get("/strategy/{track}")
def strategy_board(track: str) -> dict:
    return {"track": track, "board": predictor.build_team_strategy_board(track)}
