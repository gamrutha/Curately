"""Evaluate agent2.recommend() on the 10 queries from golden_dataset.py.

Each query runs two ways:
    end-to-end    query -> agent1 -> prefs_to_text -> agent2
    golden input  golden expected prefs -> prefs_to_text -> agent2
The golden-input run isolates agent2: if a check fails end-to-end but passes
with golden input, agent1 caused it.

Both runs are checked against the golden expected prefs (what the user meant),
accepting a golden alternative wording for exclude when agent2 applied one:
    schema          response has the right fields; every result has exactly
                    title, content_type, genre, mood, description, with the
                    right types (no per-movie scores)
    insufficient    insufficient_info queries return no results
    has_results     other queries return k results, or exactly MIN_RESULTS
                    when fewer than that matched and the rest are top-ups
    top_up          top-ups (results with none of the requested genres and
                    not the requested language) only appear after all matched
                    results, and only when fewer than MIN_RESULTS matched
    language_top_up when a language was requested and fewer than MIN_RESULTS
                    results are in that language, the films in that language
                    come first and every top-up after them is an
                    English-language film (TMDB original_language "en")
    genre_match     every matched (non-top-up) result has at least one
                    requested genre or the requested language
    exclusions      no result is an excluded genre, or has an excluded genre
                    or term as a whole word in its title, tags or plot
    note            the note names every excluded item
    fallback        fallback is used exactly when there's nothing to search on
    confidence      the response confidence_score is 1.0 for the exclude-only
                    fallback (every movie matches the one requested field,
                    exclude), 0.0 with no results, otherwise in
                    [FALLBACK_CONFIDENCE, 1]
    ordered         matched (non-top-up) results are sorted by number of
                    matched requested fields (genre, mood, language; exclude
                    counts for every movie so doesn't affect order), most
                    first. Language is checked against TMDB's original language
Soft metrics (reported, not pass/fail): mood hit rate, opposite-mood results
(tagged only with a mood that contradicts the request), confidence_score.

Usage: .venv/bin/python eval_agent2.py [-k 10]
"""

import argparse
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from agent1 import extract_preferences
from agent2 import (
    FALLBACK_CONFIDENCE, MIN_RESULTS, OPPOSITE_MOODS, _language_codes, _to_movielens_genre, build_index, parse_text, prefs_to_text,
    recommend, word_pattern,
)
from golden_dataset import GOLDEN_DATASET, INSUFFICIENT_INFO

CHECKS = [
    "schema", "insufficient", "has_results", "top_up", "language_top_up", "genre_match", "exclusions", "note",
    "fallback", "confidence", "ordered",
]
RESPONSE_KEYS = {"input", "fallback", "excluded", "note", "confidence_score", "results"}
RESULT_TYPES = {"title": str, "content_type": str, "genre": list, "mood": list, "description": str}
SHOW_TOP = 5
# Title -> TMDB original_language, filled in main() from the index, so the
# order check can verify language matches independently of agent2.
LANGUAGE_BY_TITLE: dict[str, str] = {}


def accepted_excludes(example: dict, response: dict) -> list[str]:
    """The golden exclude list, or the golden alternative that agent2 actually
    applied (e.g. "silly" for "slapstick"), so exclusions and the note are
    checked against an accepted wording."""
    options = [example["expected"]["exclude"], *example.get("alternatives", {}).get("exclude", [])]
    applied = {i.lower() for group in response.get("excluded", {}).values() for i in group}
    return next((o for o in options if {i.lower() for i in o} == applied), options[0])


def check(expected: dict, response: dict, k: int) -> dict[str, tuple[bool | None, str]]:
    """Run every check. Each maps to (passed, detail); passed is None when the
    check doesn't apply to this query."""
    results = response.get("results", [])
    out: dict[str, tuple[bool | None, str]] = {name: (None, "") for name in CHECKS}

    problems = [f"missing {key}" for key in RESPONSE_KEYS - set(response)]
    problems += [f"unexpected {key}" for key in set(response) - RESPONSE_KEYS]
    if not isinstance(response.get("confidence_score"), float):
        problems.append(f"confidence_score={response.get('confidence_score')!r}")
    for r in results:
        problems += [f"{r.get('title')}: {f}" for f, t in RESULT_TYPES.items() if not isinstance(r.get(f), t)]
        problems += [f"{r.get('title')}: unexpected {f}" for f in set(r) - set(RESULT_TYPES)]
        if r.get("content_type") != "movie":
            problems.append(f"{r.get('title')}: content_type={r.get('content_type')!r}")
    out["schema"] = (not problems, "; ".join(problems[:3]))

    confidence = response.get("confidence_score")
    if expected == INSUFFICIENT_INFO:
        out["insufficient"] = (not results, f"{len(results)} results")
        out["confidence"] = (confidence == 0.0, f"confidence_score={confidence}")
        return out

    target = min(MIN_RESULTS, k)
    received = parse_text(response.get("input", ""))
    flags = [is_top_up(received, r) for r in results]
    language_codes = _language_codes(received["language"])
    in_language = [LANGUAGE_BY_TITLE.get(r["title"]) in language_codes for r in results]
    language_top_up = bool(language_codes) and sum(in_language) < target
    if language_top_up:
        # Language top-ups: anything not in the requested language.
        flags = [not hit for hit in in_language]
    head = results[: flags.index(True)] if True in flags else results
    top_ups = results[len(head):]
    if language_top_up:
        not_english = [r["title"] for r in top_ups if LANGUAGE_BY_TITLE.get(r["title"]) != "en"]
        ok = all(flags[len(head):]) and not not_english
        out["language_top_up"] = (
            ok, f"{len(head)} in {received['language']}, then top-ups; not English: {not_english[:3]}"
        )
    out["has_results"] = (
        len(results) == k or (len(head) < target and len(results) == target),
        f"{len(results)}/{k} results ({len(head)} matched, {len(top_ups)} top-ups)",
    )
    if top_ups or len(head) < target:
        ok = all(flags[len(head):]) and len(head) < target and len(results) == target
        out["top_up"] = (ok, f"{len(head)} matched then top-up flags {flags[len(head):]}")

    wanted = {g for g in map(_to_movielens_genre, expected["genre"]) if g}
    codes = _language_codes(expected["language"])
    if wanted:
        off = [r["title"] for r in head
               if not wanted & set(r["genre"]) and LANGUAGE_BY_TITLE.get(r["title"]) not in codes]
        out["genre_match"] = (not off, f"no {'/'.join(sorted(wanted))}: {off[:3]}" if off else "")

    if expected["exclude"]:
        violations = []
        for item in expected["exclude"]:
            item = item.lower()
            genre = _to_movielens_genre(item)
            for r in results:
                if genre and genre in r["genre"]:
                    violations.append(f"{r['title']} is {genre}")
                elif any(re.search(word_pattern(t), r["description"].lower()) for t in {item, genre or item}):
                    violations.append(f"{r['title']} mentions {item!r}")
        out["exclusions"] = (not violations, "; ".join(violations[:3]))
        missing = [i for i in expected["exclude"] if i.lower() not in response.get("note", "").lower()]
        out["note"] = (not missing, f"note doesn't name {missing}" if missing else "")

    should_fallback = not any(expected[f] for f in ("genre", "mood", "language"))
    out["fallback"] = (response.get("fallback") == should_fallback, f"fallback={response.get('fallback')}, expected {should_fallback}")

    if not results:
        ok = confidence == 0.0
    elif response.get("fallback"):
        ok = confidence == 1.0
    else:
        ok = isinstance(confidence, float) and FALLBACK_CONFIDENCE <= confidence <= 1
    out["confidence"] = (ok, f"confidence_score={confidence}")

    if head:
        counts = [matched_count(received, r) for r in head]
        out["ordered"] = (counts == sorted(counts, reverse=True), f"matched-field counts {counts}")
    return out


def is_top_up(prefs: dict, movie: dict) -> bool:
    """A top-up has none of the requested genres and isn't in the requested
    language. Only possible when a genre or a known language was requested."""
    wanted = {g for g in map(_to_movielens_genre, prefs["genre"]) if g}
    codes = _language_codes(prefs["language"])
    if not wanted and not codes:
        return False
    return not wanted & set(movie["genre"]) and LANGUAGE_BY_TITLE.get(movie["title"]) not in codes


def matched_count(prefs: dict, movie: dict) -> int:
    """Requested fields the movie matches, excluding exclude (same for every
    movie). Recomputed here from the output, independently of agent2."""
    genres = {g.lower() for g in movie["genre"]}
    description = movie["description"].lower()
    count = 0
    if prefs["genre"]:
        count += all((_to_movielens_genre(g) or g).lower() in genres for g in prefs["genre"])
    if prefs["mood"]:
        count += prefs["mood"].lower() in {m.lower() for m in movie["mood"]}
    if prefs["language"]:
        codes = _language_codes(prefs["language"])
        if codes:
            count += LANGUAGE_BY_TITLE.get(movie["title"]) in codes
        else:
            count += re.search(rf"\b{re.escape(prefs['language'].lower())}\b", description) is not None
    return count


def mood_hits(expected: dict, response: dict) -> tuple[int, int, int] | None:
    """(results with the requested mood, results with only an opposite mood, total)."""
    if expected == INSUFFICIENT_INFO or not expected["mood"] or not response["results"]:
        return None
    mood, opposites = expected["mood"], OPPOSITE_MOODS.get(expected["mood"], [])
    hits = sum(mood in r["mood"] for r in response["results"])
    opposite = sum(mood not in r["mood"] and any(o in r["mood"] for o in opposites) for r in response["results"])
    return hits, opposite, len(response["results"])


def run(text: str, example: dict, k: int, collection) -> dict:
    expected = example["expected"]
    start = time.perf_counter()
    response = recommend(text, k=k, collection=collection)
    elapsed = time.perf_counter() - start
    if expected != INSUFFICIENT_INFO:
        expected = {**expected, "exclude": accepted_excludes(example, response)}
    checks = check(expected, response, k)
    return {
        "text": text,
        "response": response,
        "checks": checks,
        "passed": all(ok is not False for ok, _ in checks.values()),
        "mood": mood_hits(expected, response),
        "seconds": elapsed,
    }


def agent1(example: dict) -> tuple[dict | None, str | None]:
    try:
        return extract_preferences(example["input"]), None
    except Exception as e:  # report the failure and run agent2 on empty prefs
        return None, f"{type(e).__name__}: {e}"


def print_run(label: str, r: dict) -> None:
    mark = "PASS" if r["passed"] else "FAIL"
    resp = r["response"]
    print(f"  [{mark}] {label:<12} input={r['text']!r}")
    if resp["note"]:
        print(f"         note: {resp['note']}")
    print(f"         confidence_score: {resp['confidence_score']}")
    for i, m in enumerate(resp["results"][:SHOW_TOP], 1):
        print(f"         {i}. {m['title']}  [{', '.join(m['genre'])}]" + (f"  mood={m['mood']}" if m["mood"] else ""))
    for name, (ok, detail) in r["checks"].items():
        if ok is False:
            print(f"         ✗ {name}: {detail}")


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for mode in ("e2e", "golden"):
        label = "End-to-end (agent1 → agent2)" if mode == "e2e" else "Golden input (agent2 only)"
        runs = [row[mode] for row in rows]
        print(f"\n{label}: {sum(r['passed'] for r in runs)}/{len(runs)} queries passed")
        for name in CHECKS:
            applicable = [r["checks"][name][0] for r in runs if r["checks"][name][0] is not None]
            if applicable:
                print(f"  {name:<13} {sum(applicable)}/{len(applicable)}")
        hits = [r["mood"] for r in runs if r["mood"]]
        if hits:
            hit, opposite, total = map(sum, zip(*hits))
            print(f"  mood hit rate {hit}/{total} results tagged with the requested mood ({hit / total:.0%})")
            print(f"  opposite mood {opposite}/{total} results tagged only with a contradicting mood")
        real = [r["response"]["confidence_score"] for r in runs if not r["response"]["fallback"] and r["response"]["results"]]
        if real:
            print(f"  confidence    mean {sum(real) / len(real):.2f}, range {min(real):.2f}-{max(real):.2f} "
                  f"(response confidence_score, non-fallback)")
        print(f"  latency       mean {sum(r['seconds'] for r in runs) / len(runs) * 1000:.0f} ms per agent2 call")

    blamed = [row["example"]["id"] for row in rows if not row["e2e"]["passed"] and row["golden"]["passed"]]
    if blamed:
        print(f"\nFail end-to-end but pass with golden input (agent1 caused it): #{', #'.join(map(str, blamed))}")
    with_language = [row["example"]["id"] for row in rows
                     if row["example"]["expected"] != INSUFFICIENT_INFO and row["example"]["expected"]["language"]]
    if with_language:
        print(f"Language checked against TMDB original language: #{', #'.join(map(str, with_language))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent2 on the golden dataset queries.")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    collection = build_index()
    LANGUAGE_BY_TITLE.update(
        (m["title"], m["original_language"]) for m in collection.get(include=["metadatas"])["metadatas"]
    )
    with ThreadPoolExecutor(max_workers=5) as pool:
        extracted = list(pool.map(agent1, GOLDEN_DATASET))

    rows = []
    print("=" * 78)
    print("PER-QUERY RESULTS")
    print("=" * 78)
    for example, (prefs, error) in zip(GOLDEN_DATASET, extracted):
        row = {
            "example": example,
            "e2e": run(prefs_to_text(prefs or INSUFFICIENT_INFO), example, args.k, collection),
            "golden": run(prefs_to_text(example["expected"]), example, args.k, collection),
        }
        rows.append(row)
        print(f"\n#{example['id']} {example['category']}: {example['input']!r}")
        if error:
            print(f"  agent1 ERROR: {error}")
        print_run("end-to-end", row["e2e"])
        if row["golden"]["text"] != row["e2e"]["text"]:
            print_run("golden input", row["golden"])
        else:
            print(f"  [{'PASS' if row['golden']['passed'] else 'FAIL'}] golden input (same as end-to-end)")

    print_summary(rows)


if __name__ == "__main__":
    main()
