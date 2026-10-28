"""Download TMDB plot descriptions for every MovieLens movie (one-time).

Reads tmdbId from data/ml-latest-small/links.csv, fetches /movie/{id} from the
TMDB API, and appends one JSON line per movie to data/tmdb/movies.jsonl:
    {movieId, tmdbId, overview, original_language, spoken_languages,
     production_countries}
Re-running skips movies already saved, so an interrupted run can resume.

Needs TMDB_API_KEY in .env: either the v3 "API Key" or the v4 "API Read
Access Token" from https://www.themoviedb.org/settings/api.

Usage: .venv/bin/python fetch_tmdb.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
LINKS = ROOT / "data" / "ml-latest-small" / "links.csv"
OUT = ROOT / "data" / "tmdb" / "movies.jsonl"
API = "https://api.themoviedb.org/3/movie/{}"
WORKERS = 8  # TMDB allows roughly 50 requests/second
RETRIES = 5


def _auth(key: str) -> tuple[dict, dict]:
    """v4 read access tokens are JWTs (start with "eyJ") and go in a header;
    v3 API keys go in the query string."""
    if key.startswith("eyJ"):
        return {"Authorization": f"Bearer {key}"}, {}
    return {}, {"api_key": key}


def fetch(session: requests.Session, movie_id: int, tmdb_id: int, params: dict) -> dict:
    record = {"movieId": movie_id, "tmdbId": tmdb_id}
    for attempt in range(RETRIES):
        response = session.get(API.format(tmdb_id), params={**params, "language": "en-US"}, timeout=20)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", 2 ** attempt)))
            continue
        if response.status_code == 404:
            return {**record, "error": "not found"}
        if response.status_code == 401:
            raise SystemExit("TMDB rejected the key (401). Check TMDB_API_KEY in .env.")
        response.raise_for_status()
        movie = response.json()
        return {
            **record,
            "overview": (movie.get("overview") or "").strip(),
            "original_language": movie.get("original_language") or "",
            "spoken_languages": [l["english_name"] for l in movie.get("spoken_languages", [])],
            "production_countries": [c["name"] for c in movie.get("production_countries", [])],
        }
    return {**record, "error": "rate limited"}


def main() -> None:
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("Add TMDB_API_KEY=... to .env (see https://www.themoviedb.org/settings/api).")

    links = pd.read_csv(LINKS).dropna(subset=["tmdbId"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(line)["movieId"] for line in OUT.read_text().splitlines() if line.strip()}
    todo = [(int(r.movieId), int(r.tmdbId)) for r in links.itertuples() if int(r.movieId) not in done]
    print(f"{len(done)} already saved, {len(todo)} to fetch")

    headers, params = _auth(key)
    session = requests.Session()
    session.headers.update(headers)

    saved = errors = 0
    with OUT.open("a") as out, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for record in pool.map(lambda ids: fetch(session, *ids, params), todo):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            errors += "error" in record
            if saved % 500 == 0:
                out.flush()
                print(f"  {saved}/{len(todo)}")

    print(f"Done: {saved} saved ({errors} without data) to {OUT}")


if __name__ == "__main__":
    main()
