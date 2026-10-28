"""Agent 3: explain why each of Agent 2's top movies fits the user's request.

explain(user_query, prefs, agent2_response, client) returns
    {"query", "recommendations": [{"rank", "title", "explanation"}],
     "explanations_ok", "debug": {"confidence_score"}}
Rank is Agent 2's order (1-5), assigned in code; the LLM never sees or sets it.
"""

import json
import logging
import re
import unicodedata

import anthropic

from agent1 import MODEL
from agent2 import LANGUAGE_CODES

logger = logging.getLogger(__name__)

TOP_N = 5
MAX_ATTEMPTS = 2  # first try + one retry

# Language and nationality words an explanation may only use when that movie's
# description contains them.
NATIONALITY_WORDS = sorted(
    set(LANGUAGE_CODES)
    | {"american", "british", "french", "german", "japanese", "chinese", "indian", "italian", "russian"}
)
_POPULAR_PICK = re.compile(r"\bpopular picks?\b", re.IGNORECASE)

SYSTEM_PROMPT = """\
You write short explanations for a movie recommendation system. You get the user's \
request, the preferences extracted from it, a note from the recommender, and a list of \
movies with their title, genres, mood tags and description.

For each movie, write 1 to 2 sentences on why it fits what the user asked for. Rules:
- Use only the fields given for that movie. Do not add cast, director, plot or award \
facts from memory.
- Ignore tags that carry no meaning for the request, such as character names, \
catchphrases or brand names.
- If the movie fits only part of the request, say which part it fits and what it is \
more like instead. Do not overclaim.
- If the note says a movie is one of the popular picks filling out the list, say so.
- Call a movie a "popular pick" only if its title is listed in the note. Otherwise never \
use that phrase.
- Never claim a language, country or setting unless the movie's description states it. \
Name a language, nationality or country only if the movie's description states it (a \
line like "Language: Korean." counts).
- Do not assume where the user lives: never use words like "overseas", "abroad" or \
"close to home". Describe a movie's setting only as its description states it.
- Mention a movie's language only when the user asked for a language, and then in your \
own words (e.g. "a Korean-language thriller"). Never copy a label like "Language: Korean." \
into the explanation.
- No praise or reputation words (well-known, famous, acclaimed, classic) unless they \
are in the description.

Return only JSON: {"explanations": {"<exact title>": "<explanation>"}}, with one entry \
per movie, keyed by its title exactly as given."""

# Words ending in "." that don't end a sentence (e.g. "Dr. Strangelove").
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "vs", "jr", "sr", "no", "vol", "mt"}


def _sentence_count(text: str) -> int:
    """Sentences = terminators (. ! ?) followed by a capital letter or the end,
    ignoring common abbreviations and ellipses inside titles."""
    count = 0
    for match in re.finditer(r"[.!?]+(?=\s+[\"'(]?[A-Z]|\s*$)", text.strip()):
        before = text[: match.start()].split()
        word = before[-1].lower().strip("\"'(") if before else ""
        if match.group() == "." and word in _ABBREVIATIONS:
            continue
        count += 1
    return count


def _schema(titles: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "explanations": {
                "type": "object",
                "properties": {title: {"type": "string"} for title in titles},
                "required": titles,
                "additionalProperties": False,
            }
        },
        "required": ["explanations"],
        "additionalProperties": False,
    }


def _payload(user_query: str, prefs: dict, note: str, movies: list[dict]) -> str:
    # Agent 2's order, unchanged; no ranks or scores are sent.
    listed = [
        {"title": m["title"], "genre": m["genre"], "mood": m["mood"], "description": m["description"]}
        for m in movies
    ]
    return json.dumps(
        {"user_query": user_query, "prefs": prefs, "note": note, "movies": listed},
        indent=2,
        ensure_ascii=False,
    )


def _word_in(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def _validate(
    raw: str, titles: list[str], descriptions: dict[str, str], note: str
) -> tuple[dict | None, str | None]:
    """Returns (explanations by title, None) or (None, reason it's invalid).
    Checks format, cleanliness (no newline, tab, other control character or
    backslash), then content: "popular pick" only for titles in the note, and
    language/nationality words only when the movie's description has them."""
    try:
        explanations = json.loads(raw)["explanations"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, f"not valid JSON in the required shape ({type(e).__name__})"
    if not isinstance(explanations, dict) or set(explanations) != set(titles) or len(explanations) != len(titles):
        return None, f"titles must be exactly {titles}, got {list(explanations) if isinstance(explanations, dict) else explanations}"
    for title, text in explanations.items():
        if not isinstance(text, str) or not text.strip():
            return None, f"empty explanation for {title!r}"
        dirty = [c for c in text if c == "\\" or unicodedata.category(c) == "Cc"]
        if dirty:
            return None, (
                f"explanation for {title!r} contains a newline, tab, control character or "
                f"backslash ({', '.join(sorted({repr(c) for c in dirty}))})"
            )
        sentences = _sentence_count(text)
        if not 1 <= sentences <= 2:
            return None, f"explanation for {title!r} has {sentences} sentences, must be 1 to 2"
        if _POPULAR_PICK.search(text) and title not in note:
            return None, f"explanation for {title!r} calls it a popular pick, but the note doesn't list it"
        named = [w for w in NATIONALITY_WORDS if _word_in(w, text) and not _word_in(w, descriptions[title])]
        if named:
            return None, (
                f"explanation for {title!r} names {', '.join(w.capitalize() for w in named)}, "
                "which its description doesn't state"
            )
    return explanations, None


def _ask(client: anthropic.Anthropic, content: str, titles: list[str]) -> str | None:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": _schema(titles)},
        },
        # Same refusal fallback as agent1: rerun on Anthropic's recommended
        # fallback model if Opus 5's safety classifiers decline.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    return next((block.text for block in response.content if block.type == "text"), None)


def explain(
    user_query: str,
    prefs: dict,
    agent2_response: dict,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Explain Agent 2's top TOP_N results, keeping Agent 2's order as the rank."""
    movies = agent2_response.get("results", [])[:TOP_N]
    titles = [m["title"] for m in movies]
    output = {
        "query": user_query,
        "recommendations": [{"rank": i, "title": title, "explanation": None} for i, title in enumerate(titles, 1)],
        "explanations_ok": False,
        "debug": {"confidence_score": agent2_response.get("confidence_score")},
    }
    if not movies:
        output["explanations_ok"] = True  # nothing to explain
        return output

    client = client or anthropic.Anthropic()
    note = agent2_response.get("note", "")
    descriptions = {m["title"]: m["description"] for m in movies}
    content = _payload(user_query, prefs, note, movies)
    problem = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        message = content if problem is None else (
            f"{content}\n\nYour previous answer was rejected: {problem}. Follow the format exactly."
        )
        raw = _ask(client, message, titles)
        explanations, problem = (
            _validate(raw, titles, descriptions, note) if raw is not None else (None, "no usable response")
        )
        if explanations is None:
            logger.warning("Agent 3 attempt %d rejected: %s", attempt, problem)
        else:
            for rec in output["recommendations"]:
                rec["explanation"] = explanations[rec["title"]].strip()
            output["explanations_ok"] = True
            return output
    return output


if __name__ == "__main__":
    import sys

    from agent1 import extract_preferences
    from agent2 import build_index, prefs_to_text, recommend

    query = " ".join(sys.argv[1:]) or input("What are you in the mood to watch? ")
    client = anthropic.Anthropic()
    prefs = extract_preferences(query, client)
    response = recommend(prefs_to_text(prefs), collection=build_index())
    print(json.dumps(explain(query, prefs, response, client), indent=2, ensure_ascii=False))
