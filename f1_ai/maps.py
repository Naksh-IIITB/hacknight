from __future__ import annotations

from typing import Any

from .data import load_track_maps


class TrackMapStore:
    def __init__(self) -> None:
        self.maps = load_track_maps()

    def get(self, track_key: str) -> dict[str, Any]:
        track_map = self.maps.get(track_key)
        if track_map:
            return track_map
        return {
            "key": track_key,
            "source": "unavailable",
            "polyline": [],
            "corners": [],
            "metadata": {"note": "Run scripts/build_datasets.py to generate track maps."},
        }

    def available(self) -> list[str]:
        return sorted(self.maps)
