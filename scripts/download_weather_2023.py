"""Download and cache locally aligned 2023 PVGIS historical weather data."""

from __future__ import annotations

import json
from pathlib import Path

from pvlib.iotools import get_pvgis_hourly


ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = ROOT / "data" / "weather"
WEATHER_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2023

LOCATIONS = {
    "Vavuniya": (8.7542, 80.4982),
    "Colombo": (6.9271, 79.8612),
    "Jaffna": (9.6615, 80.0255),
}


def main() -> None:
    for name, (latitude, longitude) in LOCATIONS.items():
        print(f"Downloading and aligning {name} weather data...")

        # Download two years so that all local-time hours of 2023 are available.
        data, metadata = get_pvgis_hourly(
            latitude=latitude,
            longitude=longitude,
            start=2022,
            end=2023,
            raddatabase="PVGIS-ERA5",
            components=True,
            surface_tilt=10.0,
            surface_azimuth=180.0,
            usehorizon=True,
        )

        # Convert UTC timestamps to Sri Lankan local time.
        data = data.tz_convert("Asia/Colombo")

        # Keep exactly the local calendar year 2023.
        data = data[data.index.year == TARGET_YEAR].copy()

        if len(data) != 8760:
            raise RuntimeError(
                f"{name}: expected 8760 local hourly rows, received {len(data)}."
            )

        data.index.name = "timestamp_local"

        csv_path = WEATHER_DIR / f"{name.lower()}_{TARGET_YEAR}_pvgis.csv"
        metadata_path = WEATHER_DIR / f"{name.lower()}_{TARGET_YEAR}_metadata.json"

        data.to_csv(csv_path)

        saved_metadata = {
            "location": name,
            "latitude": latitude,
            "longitude": longitude,
            "target_local_year": TARGET_YEAR,
            "timezone": "Asia/Colombo",
            "source": "PVGIS-ERA5 historical hourly data",
            "surface_tilt": 10.0,
            "surface_azimuth": 180.0,
            "original_metadata": metadata,
        }

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(saved_metadata, file, indent=2, default=str)

        print(f"  Rows: {len(data)}")
        print(f"  First local timestamp: {data.index.min()}")
        print(f"  Last local timestamp:  {data.index.max()}")
        print(f"  Saved: {csv_path}")

    print("\nAll weather datasets aligned to Sri Lankan local time.")


if __name__ == "__main__":
    main()