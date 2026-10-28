"""Agent 2: retrieve candidate movies from a Chroma vector index of MovieLens.

Input is agent1's preferences as text, e.g.
    "Genre: comedy. Mood: heartfelt. Exclude: slapstick."
Output is {input, fallback, excluded, note, confidence_score, results}, where
results is an ordered list of
    {title, content_type, genre, mood, description}
"""

import json
import math
import os
import re
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "ml-latest-small"
# IMDb's non-commercial dataset: https://datasets.imdbws.com/title.ratings.tsv.gz
IMDB_RATINGS = ROOT / "data" / "imdb" / "title.ratings.tsv.gz"
# TMDB plot descriptions, downloaded once by fetch_tmdb.py.
TMDB_MOVIES = ROOT / "data" / "tmdb" / "movies.jsonl"
# Whether TMDB plots go into the embedded text and the description. Changing
# it needs a rebuild; build_index() does that automatically when the stored
# collection was built the other way. Override with USE_PLOTS=0 in the env.
USE_PLOTS = os.getenv("USE_PLOTS", "1") != "0"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "movies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_TAGS = 15
IGNORED_TAGS = {"in netflix queue"}
MIN_RATINGS = 10
CANDIDATE_MULTIPLIER = 5

# relevance = weighted mix of embedding similarity and genre/mood matches.
SIMILARITY_WEIGHT = 0.7
GENRE_WEIGHT = 0.2
MOOD_WEIGHT = 0.1
# Subtracted from relevance when a movie is tagged with a mood opposite to the
# one asked for (and not also with the requested mood).
OPPOSITE_MOOD_PENALTY = 0.2
# ranking_score = weighted mix of relevance and business priority.
RELEVANCE_WEIGHT = 0.8
BUSINESS_WEIGHT = 0.2
# business_priority = weighted mix of IMDb rating and IMDb popularity (votes).
IMDB_RATING_WEIGHT = 0.5
IMDB_POPULARITY_WEIGHT = 0.5
# Floor for each top movie's share of matched fields in confidence_score.
FALLBACK_CONFIDENCE = 0.1
# How many of the top results the response confidence_score averages over.
CONFIDENCE_TOP_N = 5
# Fewer matched results than this get topped up (see recommend()).
MIN_RESULTS = 5

# Language names (as agent1 writes them, lowercased) -> TMDB original_language
# codes. TMDB uses "cn" for Cantonese and "zh" for Mandarin.
LANGUAGE_CODES = {
    "english": ["en"], "spanish": ["es"], "french": ["fr"], "german": ["de"], "italian": ["it"],
    "portuguese": ["pt"], "japanese": ["ja"], "korean": ["ko"], "chinese": ["zh", "cn"],
    "mandarin": ["zh"], "cantonese": ["cn"], "hindi": ["hi"], "russian": ["ru"], "swedish": ["sv"],
    "danish": ["da"], "norwegian": ["no"], "dutch": ["nl"], "polish": ["pl"], "turkish": ["tr"],
    "persian": ["fa"], "farsi": ["fa"], "arabic": ["ar"], "hebrew": ["he"], "greek": ["el"],
    "thai": ["th"], "czech": ["cs"], "hungarian": ["hu"], "finnish": ["fi"], "romanian": ["ro"],
    "tamil": ["ta"], "telugu": ["te"], "indonesian": ["id"], "vietnamese": ["vi"], "icelandic": ["is"],
}

# TMDB original_language code -> full language name for descriptions. A
# code listed under several names (zh, cn) uses its single-code name
# (Mandarin, Cantonese); codes not in LANGUAGE_CODES get no name.
LANGUAGE_NAMES: dict[str, str] = {}
for _name, _codes in LANGUAGE_CODES.items():
    if len(_codes) == 1:
        LANGUAGE_NAMES.setdefault(_codes[0], _name.capitalize())
for _name, _codes in LANGUAGE_CODES.items():
    for _code in _codes:
        LANGUAGE_NAMES.setdefault(_code, _name.capitalize())

# Field order for the text format agent1 -> agent2.
TEXT_FIELDS = ["genre", "mood", "exclude", "language"]

MOVIELENS_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

# Maps agent1's free-text genre words onto MovieLens genres.
GENRE_ALIASES = {
    **{g.lower(): g for g in MOVIELENS_GENRES},
    "anime": "Animation",
    "cartoon": "Animation",
    "family": "Children",
    "kids": "Children",
    "noir": "Film-Noir",
    "film noir": "Film-Noir",
    "musicals": "Musical",
    "romantic": "Romance",
    "romcom": "Romance",
    "science fiction": "Sci-Fi",
    "scifi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "documentaries": "Documentary",
    "westerns": "Western",
    # Agent 1 writes excludes as bare terms, so "nothing scary" arrives as
    # "scary"; treat these as the genre so they're filtered by genre, not just text.
    "scary": "Horror",
    "spooky": "Horror",
    "creepy": "Horror",
    "terrifying": "Horror",
}

# agent1's mood values -> MovieLens tags that signal them. Used both to label
# each movie's mood at index time and to expand the mood in the query text.
MOOD_TAGS = {
    "heartfelt": ["heartwarming", "touching", "emotional", "bittersweet", "sentimental"],
    "light": ["lighthearted", "feel-good", "cute", "whimsical", "charming"],
    "dark": ["dark", "disturbing", "bleak", "dark comedy", "black comedy", "dark humor", "gritty", "creepy"],
    "intense": ["intense", "tense", "gripping", "violent", "violence", "brutal"],
    "sad": ["sad", "tearjerker", "depressing", "melancholy", "tragedy", "death"],
    "funny": ["funny", "hilarious", "witty", "humorous", "comedy", "satire", "spoof", "quirky"],
    "uplifting": ["uplifting", "inspirational", "inspiring", "feel-good", "hopeful"],
    "romantic": ["romantic", "romance", "love story"],
    "suspenseful": ["suspense", "suspenseful", "twist ending", "thriller", "tense", "mystery"],
}

# How each mood is phrased in the query. Worded like a plot description so it
# matches TMDB plots, not just tags.
MOOD_PHRASES = {
    "heartfelt": "a heartwarming, touching and emotional story",
    "light": "a lighthearted, feel-good and charming story full of fun",
    "dark": "a dark, bleak and disturbing story",
    "intense": "an intense, gripping and tense story",
    "sad": "a sad, heartbreaking tearjerker about loss and grief",
    "funny": "a hilarious, witty and funny story",
    "uplifting": "an uplifting, inspiring and hopeful story",
    "romantic": "a romantic love story",
    "suspenseful": "a suspenseful mystery full of twists",
}

# Moods that contradict the requested one, e.g. a "dark" movie for "light".
OPPOSITE_MOODS = {
    "heartfelt": ["dark"],
    "light": ["dark", "sad", "intense"],
    "dark": ["light", "uplifting"],
    "intense": ["light"],
    "sad": ["light", "funny", "uplifting"],
    "funny": ["sad"],
    "uplifting": ["dark", "sad"],
    "romantic": [],
    "suspenseful": ["light"],
}


def _genre_key(genre: str) -> str:
    return "genre_" + genre.lower().replace("-", "_")


def _mood_key(mood: str) -> str:
    return "mood_" + mood


def _to_movielens_genre(value: str) -> str | None:
    value = value.strip().lower()
    return GENRE_ALIASES.get(value) or GENRE_ALIASES.get(value.replace("-", " "))


def _moods_for(tags: list[str]) -> list[str]:
    tag_set = set(tags)
    return [mood for mood, signals in MOOD_TAGS.items() if tag_set & set(signals)]


# ---------------------------------------------------------------- text format


def prefs_to_text(prefs: dict) -> str:
    """Convert agent1's JSON to agent2's input text, skipping empty fields:
    {"genre": ["comedy"], "mood": "heartfelt", "exclude": ["slapstick"], ...}
    -> "Genre: comedy. Mood: heartfelt. Exclude: slapstick."
    """
    if prefs.get("status") == "insufficient_info":
        return ""
    parts = []
    for field in TEXT_FIELDS:
        value = prefs.get(field)
        if isinstance(value, list):
            value = ", ".join(v.strip() for v in value if v.strip())
        if value and value.strip():
            parts.append(f"{field.capitalize()}: {value.strip()}.")
    return " ".join(parts)


def parse_text(text: str) -> dict:
    """Inverse of prefs_to_text(). Unknown labels (e.g. an old "Setting:") are
    ignored: any "Word:" ends the previous value."""
    labels = "|".join(TEXT_FIELDS)
    pattern = rf"\b({labels})\s*:\s*(.*?)\s*(?=\.?\s*\b[a-z]+\s*:|\.?\s*$)"
    prefs = {"genre": [], "mood": "", "exclude": [], "language": ""}
    for label, value in re.findall(pattern, text, flags=re.IGNORECASE):
        field = label.lower()
        value = value.strip().rstrip(".")
        if isinstance(prefs[field], list):
            prefs[field] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            prefs[field] = value.lower()
    return prefs


# ---------------------------------------------------------------- indexing


def _load_movies() -> pd.DataFrame:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")
    imdb = pd.read_csv(IMDB_RATINGS, sep="\t").rename(
        columns={"tconst": "imdb_id", "averageRating": "imdb_rating", "numVotes": "imdb_votes"}
    )

    stats = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", num_ratings="count")

    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
    tags = tags[~tags["tag"].isin(IGNORED_TAGS)]
    top_tags = (
        tags.groupby(["movieId", "tag"]).size().rename("n").reset_index()
        .sort_values(["movieId", "n"], ascending=[True, False])
        .groupby("movieId")["tag"].apply(lambda t: list(t[:MAX_TAGS]))
        .rename("tags")
    )

    links["imdb_id"] = links["imdbId"].apply(lambda i: f"tt{i:07d}")
    movies = movies.join(stats, on="movieId").join(top_tags, on="movieId")
    movies = movies.merge(links[["movieId", "imdb_id"]], on="movieId", how="left")
    movies = movies.merge(imdb, on="imdb_id", how="left")
    tmdb = pd.read_json(TMDB_MOVIES, lines=True)[["movieId", "overview", "original_language"]]
    movies = movies.merge(tmdb.rename(columns={"overview": "plot"}), on="movieId", how="left")
    movies["original_language"] = movies["original_language"].fillna("")
    movies["plot"] = movies["plot"].fillna("").str.strip().str.rstrip(".")
    movies["avg_rating"] = movies["avg_rating"].fillna(0.0)
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)
    movies["tags"] = movies["tags"].apply(lambda t: t if isinstance(t, list) else [])
    movies["genre_list"] = movies["genres"].apply(lambda g: [] if g == "(no genres listed)" else g.split("|"))
    movies["moods"] = movies["tags"].apply(_moods_for)

    movies["business_priority"] = _business_priority(movies)
    return movies


def _business_priority(movies: pd.DataFrame) -> pd.Series:
    """0-1 score from IMDb rating and IMDb popularity (vote count).

    The rating is IMDb's own weighted rating, which pulls titles with few votes
    toward the average so a 9.5 from 40 votes doesn't outrank a 9.0 from a
    million. Popularity is log-scaled votes, so blockbusters don't swamp
    everything. The few movies with no IMDb match fall back to MovieLens'
    rating and rating count.
    """
    has_imdb = movies["imdb_rating"].notna()
    votes = movies["imdb_votes"].fillna(0)
    mean_rating = movies.loc[has_imdb, "imdb_rating"].mean()
    min_votes = votes[has_imdb].median()
    weighted_rating = (votes * movies["imdb_rating"].fillna(0) + min_votes * mean_rating) / (votes + min_votes)
    imdb_popularity = votes.apply(math.log1p) / math.log1p(votes.max())

    ml_rating = movies["avg_rating"] * 2  # 5-star scale -> 10-point scale
    ml_popularity = movies["num_ratings"].apply(math.log1p) / math.log1p(movies["num_ratings"].max())

    rating = weighted_rating.where(has_imdb, ml_rating) / 10
    popularity = imdb_popularity.where(has_imdb, ml_popularity)
    return IMDB_RATING_WEIGHT * rating + IMDB_POPULARITY_WEIGHT * popularity


def _description(row) -> str:
    parts = [row.title]
    if row.genre_list:
        parts.append("Genres: " + ", ".join(row.genre_list))
    # Only in the returned description, never in the embedded document.
    if LANGUAGE_NAMES.get(row.original_language):
        parts.append(f"Language: {LANGUAGE_NAMES[row.original_language]}")
    if USE_PLOTS and row.plot:
        parts.append("Plot: " + row.plot)
    if row.tags:
        parts.append("Tags: " + ", ".join(row.tags))
    if pd.notna(row.imdb_rating):
        parts.append(f"IMDb rating {row.imdb_rating:.1f}/10 from {int(row.imdb_votes):,} votes")
    if row.num_ratings:
        parts.append(f"MovieLens rating {row.avg_rating:.1f}/5 from {row.num_ratings} ratings")
    return ". ".join(parts) + "."


def _document(row) -> str:
    """Text that gets embedded. Lowercased so where_document exclusions are
    case-insensitive. The plot goes last: the model reads only the first 256
    tokens, so the rare long plot gets cut instead of the genres or tags."""
    parts = [row.title, "genres: " + ", ".join(row.genre_list)]
    if row.moods:
        parts.append("mood: " + ", ".join(row.moods))
    if row.tags:
        parts.append("tags: " + ", ".join(row.tags))
    if USE_PLOTS and row.plot:
        parts.append("plot: " + row.plot)
    return ". ".join(parts).lower()


def _metadata(row) -> dict:
    return {
        "title": row.title,
        "content_type": "movie",
        "genres": ", ".join(row.genre_list),
        "moods": ", ".join(row.moods),
        "description": _description(row),
        "avg_rating": round(float(row.avg_rating), 3),
        "num_ratings": int(row.num_ratings),
        "imdb_id": row.imdb_id if isinstance(row.imdb_id, str) else "",
        "imdb_rating": float(row.imdb_rating) if pd.notna(row.imdb_rating) else 0.0,
        "imdb_votes": int(row.imdb_votes) if pd.notna(row.imdb_votes) else 0,
        "original_language": row.original_language,
        "business_priority": round(float(row.business_priority), 4),
        # One bool per genre and mood, since filters can't test membership in a string.
        **{_genre_key(g): g in row.genre_list for g in MOVIELENS_GENRES},
        **{_mood_key(m): m in row.moods for m in MOOD_TAGS},
    }


def get_collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        COLLECTION,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL),
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"use_plots": USE_PLOTS},
    )


def refresh_descriptions() -> int:
    """Recompute every movie's metadata (including its description) and update
    it in place. Documents aren't passed, so nothing is re-embedded."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_collection(client)
    movies = _load_movies()
    batch = min(client.get_max_batch_size(), 1000)
    for start in range(0, len(movies), batch):
        chunk = movies.iloc[start : start + batch]
        collection.update(
            ids=[str(m) for m in chunk["movieId"]],
            metadatas=[_metadata(r) for r in chunk.itertuples()],
        )
    return len(movies)


def build_index(rebuild: bool = False) -> chromadb.Collection:
    """Embed every MovieLens movie into the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        built_with_plots = (client.get_collection(COLLECTION).metadata or {}).get("use_plots")
        if rebuild or built_with_plots != USE_PLOTS:
            client.delete_collection(COLLECTION)
    collection = get_collection(client)

    movies = _load_movies()
    if collection.count() == len(movies):
        return collection

    batch = min(client.get_max_batch_size(), 1000)
    for start in range(0, len(movies), batch):
        chunk = movies.iloc[start : start + batch]
        collection.upsert(
            ids=[str(m) for m in chunk["movieId"]],
            documents=[_document(r) for r in chunk.itertuples()],
            metadatas=[_metadata(r) for r in chunk.itertuples()],
        )
        print(f"Indexed {min(start + batch, len(movies))}/{len(movies)}")
    return collection


# ---------------------------------------------------------------- retrieval


def _query_text(prefs: dict) -> str:
    """What gets embedded. Excludes are left out on purpose: embedding
    "slapstick" would pull results toward slapstick. They're filters instead."""
    parts = []
    if prefs["genre"]:
        parts.append("genres: " + ", ".join(prefs["genre"]))
    if prefs["mood"]:
        # No "mood:" label: the word "mood" itself matched titles like
        # "In the Mood for Love" regardless of the mood asked for. The phrase
        # matches plots; the tag list matches tags.
        mood = prefs["mood"]
        parts.append(MOOD_PHRASES.get(mood, f"a {mood} story"))
        if MOOD_TAGS.get(mood):
            parts.append("tags: " + ", ".join(MOOD_TAGS[mood]))
    # Language is also matched on TMDB's original_language; these words only
    # nudge the embedding.
    if prefs["language"]:
        parts.append(f"{prefs['language']} language")
    return ". ".join(parts).lower()


def word_pattern(term: str) -> str:
    """Regex matching term as a whole word (plural too), so excluding "war"
    removes "war" and "wars" but not "toward" or "award" in a plot."""
    return rf"\b{re.escape(term.lower())}s?\b"


def _language_codes(language: str) -> list[str]:
    return LANGUAGE_CODES.get(language.strip().lower(), [])


def _filters(prefs: dict, min_ratings: int) -> dict:
    """Builds the Chroma filters. Returns a dict with:
    where_any       base filters + matches ANY requested genre or language
    where_all       base filters + matches EVERY requested genre and language
                    (None when fewer than two of them are requested)
    where_document  exclusions by whole-word text match (or None)
    wanted          requested genres as MovieLens names
    excluded_genres, excluded_terms
                    excluded items split by type, as the user worded them
    Base filters are the minimum rating count and the excluded genres.
    """
    wanted = list(dict.fromkeys(g for g in map(_to_movielens_genre, prefs["genre"]) if g))

    base: list[dict] = [{"num_ratings": {"$gte": min_ratings}}]
    text_excludes: list[str] = []
    excluded_genres: list[str] = []
    excluded_terms: list[str] = []
    for item in dict.fromkeys(i.strip().lower() for i in prefs["exclude"] if i.strip()):
        genre = _to_movielens_genre(item)
        if genre:
            excluded_genres.append(item)
            base.append({_genre_key(genre): False})
            # Also drop movies users tagged with the genre (e.g. Black Swan is
            # tagged "horror" but its MovieLens genres are Drama, Thriller).
            text_excludes += [item, genre.lower()]
        else:
            excluded_terms.append(item)
            text_excludes.append(item)

    structured = [{_genre_key(g): True} for g in wanted]
    codes = _language_codes(prefs["language"])
    if codes:
        structured.append({"original_language": {"$in": codes}})

    def combine(conditions: list[dict]) -> dict:
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    any_of = [] if not structured else [structured[0] if len(structured) == 1 else {"$or": structured}]
    doc_filters = [{"$not_regex": word_pattern(t)} for t in dict.fromkeys(text_excludes)]
    where_document = None
    if doc_filters:
        where_document = doc_filters[0] if len(doc_filters) == 1 else {"$and": doc_filters}
    return {
        "where_any": combine(base + any_of),
        "where_all": combine(base + structured) if len(structured) > 1 else None,
        "where_document": where_document,
        "wanted": wanted,
        "excluded_genres": excluded_genres,
        "excluded_terms": excluded_terms,
    }


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _note(
    fallback: bool,
    excluded_genres: list[str],
    excluded_terms: list[str],
    found: bool,
    matched: int,
    top_ups: list[str],
    target: int,
) -> str:
    """User-facing summary of what was searched, what was left out, and which
    results are top-ups."""
    if not found:
        return "No movies matched these preferences."
    parts = []
    if fallback:
        parts.append("You didn't name a genre or mood, so here are highly rated popular movies.")
    if excluded_genres:
        parts.append(f"None of them are {_join(excluded_genres)} movies.")
    if excluded_terms:
        parts.append(f"Movies tagged or described as {_join(excluded_terms)} were left out.")
    if top_ups:
        parts.append(
            f"{matched} of {target} matched your request; "
            f"the rest are popular picks: {', '.join(top_ups)}."
        )
    return " ".join(parts)


def recommend(
    text: str,
    k: int = 10,
    min_ratings: int = MIN_RATINGS,
    collection: chromadb.Collection | None = None,
) -> dict:
    """Return the top-k movies for agent1's preferences text, as
    {input, fallback, excluded: {genres, terms}, note, confidence_score, results}.

    With nothing to search on (only excludes), falls back to highly rated
    popular movies. Requested genres and language are a filter (a movie needs
    at least one of them; language is TMDB's original language). Excluded genres are filtered out by metadata and by whole-word
    match on title, tags and plot; other excluded terms by whole-word match
    alone.

    Results are ordered by how many requested fields each movie matches (see
    _matched_fields), then by the internal ranking (relevance + business
    priority). If fewer than MIN_RESULTS remain, the list is topped up with
    the next best movies that pass only the exclusions and the minimum-ratings
    filter, appended after the matched ones. confidence_score is for the whole response: the mean over the
    top CONFIDENCE_TOP_N movies of matched fields / requested fields (each at
    least FALLBACK_CONFIDENCE), rounded to 2 decimals, including for the
    exclude-only fallback; 0.0 when there are no results.
    """
    prefs = parse_text(text)
    query = _query_text(prefs)
    if not query and not prefs["exclude"]:
        return {
            "input": text,
            "fallback": False,
            "excluded": {"genres": [], "terms": []},
            "note": "Not enough information to recommend movies.",
            "confidence_score": 0.0,
            "results": [],
        }

    collection = collection or get_collection()
    f = _filters(prefs, min_ratings)
    excluded_genres, excluded_terms = f["excluded_genres"], f["excluded_terms"]

    fallback = not query
    if fallback:
        candidates = _popular(collection, f["where_any"], f["where_document"])
    else:
        # Search twice when there's more than one genre/language: once for
        # movies matching all of them, once for movies matching any. The
        # first pass makes sure full matches (e.g. Spanish-language
        # thrillers) are found even if the similarity search alone would miss
        # them; the second supplies partial matches to follow.
        n_results = k * CANDIDATE_MULTIPLIER
        wheres = [w for w in (f["where_all"], f["where_any"]) if w is not None]
        candidates = _search(collection, query, prefs["mood"], f["wanted"], wheres, f["where_document"], n_results)

    # Candidates arrive in internal ranking order; a stable sort on matched
    # fields keeps that order within each match count.
    requested = _requested_fields(prefs)
    scored = [(movie, _matched_fields(prefs, movie, meta)) for movie, meta in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    scored = scored[:k]
    matched_count = len(scored)

    target = min(MIN_RESULTS, k)

    # Language top-up: when a language was requested and fewer than
    # MIN_RESULTS films in that language pass the filters, return those films
    # followed only by English-language films (see _language_top_up).
    language_hits = None
    if not fallback and prefs["language"]:
        language_hits = _language_matches(collection, query, prefs, f, min_ratings, n_results, scored)
    if language_hits is not None and len(language_hits) < target:
        scored = language_hits
        matched_count = len(scored)
        scored += _language_top_up(collection, query, prefs, f, min_ratings, n_results, scored, target)

    # C. Top-up: with fewer than MIN_RESULTS matches, append the next best
    # movies that pass only the exclusions and the minimum-ratings filter
    # (no genre/language requirement), in internal ranking order. Not used
    # when the language top-up above applied.
    elif not fallback and len(scored) < target:
        base = _filters({**prefs, "genre": [], "language": ""}, min_ratings)
        seen = {movie["description"] for movie, _ in scored}
        extra = _search(collection, query, prefs["mood"], f["wanted"], [base["where_any"]], base["where_document"], n_results)
        for movie, meta in extra:
            if len(scored) >= target:
                break
            if movie["description"] not in seen:
                seen.add(movie["description"])
                scored.append((movie, _matched_fields(prefs, movie, meta)))

    if scored:
        top = scored[:CONFIDENCE_TOP_N]
        confidence = sum(max(matched / len(requested), FALLBACK_CONFIDENCE) for _, matched in top) / len(top)
    else:
        confidence = 0.0
    top_ups = [movie["title"] for movie, _ in scored[matched_count:]]

    return {
        "input": text,
        "fallback": fallback,
        "excluded": {"genres": excluded_genres, "terms": excluded_terms},
        "note": _note(fallback, excluded_genres, excluded_terms, bool(scored), matched_count, top_ups, target),
        "confidence_score": round(confidence, 2),
        "results": [movie for movie, _ in scored],
    }


def _requested_fields(prefs: dict) -> list[str]:
    """The non-empty fields among genre, mood, exclude, language."""
    return [field for field in TEXT_FIELDS if prefs[field]]


def _whole_word(term: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(term.lower())}\b", text.lower()) is not None


def _matches_language(prefs: dict, movie: dict, meta: dict) -> bool:
    """The movie's original language (TMDB) is the requested one; for a
    language not in LANGUAGE_CODES, the name appears as a whole word in the
    movie's description."""
    codes = _language_codes(prefs["language"])
    if codes:
        return meta.get("original_language") in codes
    return _whole_word(prefs["language"], movie["description"])


def _language_matches(
    collection: chromadb.Collection,
    query: str,
    prefs: dict,
    f: dict,
    min_ratings: int,
    n_results: int,
    scored: list[tuple[dict, int]],
) -> list[tuple[dict, int]]:
    """Films in the requested language that pass the exclusions and the
    minimum-ratings filter, as (result, matched fields) ordered by matched
    fields then internal ranking. For a known language this searches the
    whole catalogue in that language; otherwise it keeps the language
    matches already in scored."""
    if not _language_codes(prefs["language"]):
        return [(movie, n) for movie, n in scored if _whole_word(prefs["language"], movie["description"])]
    only_language = _filters({**prefs, "genre": []}, min_ratings)
    found = _search(
        collection, query, prefs["mood"], f["wanted"], [only_language["where_any"]],
        only_language["where_document"], n_results,
    )
    hits = [(movie, _matched_fields(prefs, movie, meta)) for movie, meta in found if _matches_language(prefs, movie, meta)]
    hits.sort(key=lambda pair: pair[1], reverse=True)
    return hits


def _language_top_up(
    collection: chromadb.Collection,
    query: str,
    prefs: dict,
    f: dict,
    min_ratings: int,
    n_results: int,
    scored: list[tuple[dict, int]],
    target: int,
) -> list[tuple[dict, int]]:
    """English-language films (original_language "en") that pass the
    exclusions and the minimum-ratings filter, to fill up to target. Ranked by
    how many of the other requested fields (genre, mood) they match, then by
    internal ranking. They count as unmatched on language."""
    base = _filters({**prefs, "genre": [], "language": ""}, min_ratings)
    english = {"$and": [base["where_any"], {"original_language": "en"}]}
    wheres = [english]
    if f["wanted"]:
        any_genre = [{_genre_key(g): True} for g in f["wanted"]]
        wheres.insert(0, {"$and": [english, any_genre[0] if len(any_genre) == 1 else {"$or": any_genre}]})
    seen = {movie["description"] for movie, _ in scored}
    found = _search(collection, query, prefs["mood"], f["wanted"], wheres, base["where_document"], n_results)
    fills = [(movie, _matched_fields(prefs, movie, meta)) for movie, meta in found if movie["description"] not in seen]
    fills.sort(key=lambda pair: pair[1], reverse=True)
    return fills[: max(target - len(scored), 0)]


def _matched_fields(prefs: dict, movie: dict, meta: dict) -> int:
    """How many requested fields the movie matches (all compared lowercased):
    genre    the movie's genres contain every requested genre (requested
             genres mapped to MovieLens names first, e.g. "family" -> Children)
    mood     the movie's moods contain the requested mood
    language the movie's original language (TMDB) is the requested one; for
             a language not in LANGUAGE_CODES, the name appears as a whole
             word in the movie's description
    exclude  matched for every returned movie (exclusions are enforced as
             filters, so every result satisfies them)
    """
    genres = {g.lower() for g in movie["genre"]}
    checks = {
        "genre": lambda: all((_to_movielens_genre(g) or g).lower() in genres for g in prefs["genre"]),
        "mood": lambda: prefs["mood"].lower() in {m.lower() for m in movie["mood"]},
        "language": lambda: _matches_language(prefs, movie, meta),
        "exclude": lambda: True,
    }
    return sum(checks[field]() for field in _requested_fields(prefs))


def _search(
    collection: chromadb.Collection,
    query: str,
    mood: str,
    wanted: list[str],
    wheres: list[dict],
    where_document: dict | None,
    n_results: int,
) -> list[tuple[dict, dict]]:
    """All candidates retrieved by any of the wheres (deduplicated), as
    (result, metadata) pairs in internal ranking order (relevance + business
    priority). The score itself isn't returned."""
    ranked: dict[str, tuple[float, dict]] = {}
    for where in wheres:
        result = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["metadatas", "distances"],
        )
        for movie_id, meta, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0]):
            similarity = 1 - distance
            genre_match = sum(meta[_genre_key(g)] for g in wanted) / len(wanted) if wanted else 0.0
            relevance = SIMILARITY_WEIGHT * similarity + GENRE_WEIGHT * genre_match + _mood_score(meta, mood)
            ranking_score = RELEVANCE_WEIGHT * relevance + BUSINESS_WEIGHT * meta["business_priority"]
            ranked[movie_id] = (ranking_score, meta)

    ordered = sorted(ranked.values(), key=lambda pair: pair[0], reverse=True)
    return [(_result(meta), meta) for _, meta in ordered]


def _mood_score(meta: dict, mood: str) -> float:
    """Bonus for movies tagged with the requested mood; penalty for movies
    tagged only with an opposite one. Untagged movies get 0."""
    if not mood:
        return 0.0
    if meta.get(_mood_key(mood)):
        return MOOD_WEIGHT
    if any(meta.get(_mood_key(m)) for m in OPPOSITE_MOODS.get(mood, [])):
        return -OPPOSITE_MOOD_PENALTY
    return 0.0


def _popular(collection: chromadb.Collection, where: dict, where_document: dict | None) -> list[tuple[dict, dict]]:
    """Fallback when there's nothing to search on (e.g. only excludes): every
    movie that passes the filters, highest business priority first."""
    metas = collection.get(where=where, where_document=where_document, include=["metadatas"])["metadatas"]
    metas = sorted(metas, key=lambda m: m["business_priority"], reverse=True)
    return [(_result(meta), meta) for meta in metas]


def _result(meta: dict) -> dict:
    return {
        "title": meta["title"],
        "content_type": meta["content_type"],
        "genre": [g for g in meta["genres"].split(", ") if g],
        "mood": [m for m in meta["moods"].split(", ") if m],
        "description": meta["description"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="free-text request (runs agent1 first)")
    parser.add_argument("--text", help='agent2 input text, skipping agent1, e.g. "Genre: comedy. Mood: heartfelt."')
    parser.add_argument("--build", action="store_true", help="(re)build the Chroma index and exit")
    parser.add_argument(
        "--refresh-descriptions", action="store_true",
        help="update stored descriptions/metadata without re-embedding, and exit",
    )
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    if args.refresh_descriptions:
        print(f"{refresh_descriptions()} descriptions refreshed in {CHROMA_DIR}")
        raise SystemExit

    if args.build:
        print(f"{build_index(rebuild=True).count()} movies indexed in {CHROMA_DIR}")
        raise SystemExit

    if args.text is not None:
        text = args.text
    else:
        from agent1 import extract_preferences

        text = prefs_to_text(extract_preferences(" ".join(args.query) or input("What are you in the mood to watch? ")))
    print(f"Agent 2 input: {text!r}\n")
    print(json.dumps(recommend(text, k=args.k, collection=build_index()), indent=2))
