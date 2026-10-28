# Curately

Curately is a small movie recommender you run on your own machine. You type what
you feel like watching ("something funny but not too silly, maybe with some
heart") and it returns five movies, each with a short explanation of why it fits.

It is built as three agents in a row, uses Claude for the language steps, and a
local vector database of about 9,700 MovieLens movies for the search.

## How it works

1. **Agent 1 – understand the request** (`agent1.py`). Claude turns your message
   into four fields: genres you want, a mood, things to avoid, and a language.
   Anything you didn't say stays empty. If there is nothing to go on (for example
   "something good"), it says so instead of guessing.
2. **Agent 2 – find the movies** (`agent2.py`). It searches a local Chroma
   database of movie descriptions (title, genres, user tags and a TMDB plot
   summary, embedded with `all-MiniLM-L6-v2`). Things to avoid are hard
   filters. Movies that match more of what you asked for come first; ties are
   broken by text similarity plus IMDb rating and popularity. If fewer than five
   movies match, it fills the list with popular picks and says so. It also
   returns a confidence score (how many of your requested fields the top five
   match).
3. **Agent 3 – explain the picks** (`agent3.py`). Claude writes one or two
   sentences per movie, using only the information Agent 2 returned. Code
   checks every explanation (format, no invented language or nationality, no
   "popular pick" label unless it really is one) and retries once if a check
   fails.

`app.py` is a Streamlit page that runs all three agents for each search.

## Setup

You need Python 3.12, an Anthropic API key and a free TMDB API key
(https://www.themoviedb.org/settings/api).

1. Install the dependencies:

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your keys (`.env` is git-ignored):

   ```
   ANTHROPIC_API_KEY=your-key-here
   TMDB_API_KEY=your-tmdb-key-here
   ```

3. Download the data into `data/` (it is git-ignored; the data licences don't
   allow redistributing it):

   ```
   mkdir -p data/imdb
   curl -L -o data/ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
   unzip data/ml-latest-small.zip -d data && rm data/ml-latest-small.zip
   curl -L -o data/imdb/title.ratings.tsv.gz https://datasets.imdbws.com/title.ratings.tsv.gz
   python fetch_tmdb.py
   ```

   `fetch_tmdb.py` downloads a plot summary and the original language for each
   movie into `data/tmdb/movies.jsonl` (about 5 minutes; it resumes if
   interrupted).

4. Build the vector database in `chroma_db/` (downloads the embedding model on
   first run):

   ```
   python agent2.py --build
   ```

   If you only change how descriptions are written, `python agent2.py
   --refresh-descriptions` updates them without re-embedding.

## Run

The app (opens at http://localhost:8501, on your machine only):

```
streamlit run app.py
```

From the command line:

```
python agent3.py "something in Spanish, a good thriller"       # all three agents
python agent2.py --text "Genre: comedy. Mood: heartfelt."       # Agent 2 only
python agent1.py "definitely not a horror movie, something light"  # Agent 1 only
```

## Evaluations

Both evaluations use the 10 queries in `golden_dataset.py` and call the
Anthropic API.

```
python eval_agent1.py   # Agent 1's extraction vs the golden answers
python eval_agent2.py   # Agent 1 -> Agent 2, and Agent 2 alone on the golden answers
```

### Latest results

**Agent 1** (`eval_agent1.py`)

| Metric | Result |
|---|---|
| Examples passed | 10/10 |
| Accuracy | 100.0% (12/12 mentioned fields correct) |
| Hallucination rate | 0.0% (0/40 fields filled that should be empty) |
| Precision / recall / F1 | 1.000 / 1.000 / 1.000 |

**Agent 2** (`eval_agent2.py`)

| Check | Agent 1 → Agent 2 | Agent 2 alone |
|---|---|---|
| Queries passed | 10/10 | 10/10 |
| Schema / insufficient info / result count | 10/10, 2/2, 8/8 | 10/10, 2/2, 8/8 |
| Genre match / exclusions / note | 6/6, 2/2, 2/2 | 6/6, 2/2, 2/2 |
| Fallback / confidence / ordering | 8/8, 10/10, 8/8 | 8/8, 10/10, 8/8 |
| Results tagged with the requested mood | 14/30 (47%) | 14/30 (47%) |
| Results tagged only with a contradicting mood | 0/30 | 0/30 |
| Confidence score (non-fallback) | mean 0.95, range 0.70–1.00 | mean 0.95, range 0.70–1.00 |

**Agent 3** has no automated evaluation yet; its explanations have been checked
by hand.

## Known limitations

- **No topic field.** Agent 1 extracts genre, mood, exclusions and language only,
  so a topic like "about space exploration" is dropped (you get documentaries,
  not space documentaries).
- **Mood tags are sparse.** Moods come from MovieLens user tags, and only about
  47% of results carry the requested mood tag.
- **Explanations can add a small unsupported detail.** Agent 3 occasionally adds
  a detail that isn't in the movie's description (for example "a frozen
  planet"). Code checks catch invented languages and nationalities, not every
  detail.
- **Language requests can return fewer matches.** Only movies with at least 10
  MovieLens ratings are searched, so for languages with few such films (for
  example Korean) fewer than five match and the rest are English-language
  popular picks.
- **No search by actor or director.**
- Places ("set in Paris") are ignored.

## Data and licences

- MovieLens: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens
  Datasets: History and Context. ACM Transactions on Interactive Intelligent
  Systems (TiiS) 5, 4: 19:1–19:19.
- IMDb ratings: IMDb non-commercial datasets (personal and non-commercial use).
- TMDB descriptions and original languages via the TMDB API.

This product uses the TMDB API but is not endorsed or certified by TMDB.

## License

The code is released under the MIT License (see `LICENSE`). The movie data is not
included and keeps its own licences (see above).
