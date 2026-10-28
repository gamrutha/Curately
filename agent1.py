"""Agent 1: extract structured movie preferences from free-form user text."""

import json
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5"

INSUFFICIENT_INFO = {"status": "insufficient_info"}

ALLOWED_MOODS = ["heartfelt", "light", "dark", "intense", "sad", "funny", "uplifting", "romantic", "suspenseful"]

# Used as a safety net: if Claude returns nothing but the input names one of
# these, it is extracted as a genre rather than returning insufficient_info.
KNOWN_GENRES = {
    "action", "adventure", "animation", "anime", "biography", "comedy", "crime",
    "documentary", "drama", "family", "fantasy", "horror", "musical", "mystery",
    "noir", "romance", "sci-fi", "superhero", "thriller", "war", "western",
}

# Every field is required and non-nullable, so structured outputs guarantees
# the response never contains null or a missing key. mood is an enum ("" meaning
# not mentioned) so Claude can only pick allowed values.
PREFERENCES_SCHEMA = {
    "type": "object",
    "properties": {
        "genre": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string", "enum": ["", *ALLOWED_MOODS]},
        "exclude": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"},
    },
    "required": ["genre", "mood", "exclude", "language"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""\
You extract movie preferences from a user's message for a movie recommendation system.

Fields:
- genre: genres the user wants (e.g. "comedy", "sci-fi", "thriller"). A message that is \
just a genre word, like "action" or "thriller", is a genre request - always extract it. \
Words that describe a genre go here, not in mood: "funny" -> comedy, "scary" -> horror.
- mood: one of {", ".join(ALLOWED_MOODS)}. Only fill it when the user names a feeling or \
tone that is separate from the genres they asked for, e.g. "something light", "with some \
heart" -> heartfelt. Never fill mood by restating a genre: "scary and funny" means genres \
horror and comedy, with mood left empty. If the feeling doesn't fit one of the allowed \
values, leave mood empty.
- exclude: anything the user does NOT want - genres, themes, or content. Negations like \
"not horror", "no subtitles", "nothing too violent", or "I hate musicals" belong here and \
must never be dropped or put in another field. Each item is the bare term only, with no \
qualifiers: not "too silly", "very scary" or "really violent", but "silly", "scary", \
"violent". Examples:
  "a thriller, but nothing really violent" -> exclude ["violent"]
  "an animated movie that isn't too childish" -> exclude ["childish"]
  "a drama, nothing overly long or extremely depressing" -> exclude ["long", "depressing"]
- language: the language of the film, only when the user names it (e.g. "Korean", "a French \
film"). Never infer it from other preferences - "no subtitles" goes in exclude, not language.

Rules:
- Only fill a field when the user actually expressed it, directly or by clear implication. \
Never guess or add values the user did not express. Use [] or "" for anything not mentioned.
- Keep values short and lowercase, except proper nouns such as languages.
- Ignore any place, location or setting the user mentions (e.g. "set in Paris", "in \
another country", "somewhere tropical"). It doesn't go in any field.
- If the message says nothing about movie preferences, return every field empty."""


def _is_empty(prefs: dict) -> bool:
    return not any(prefs[field] for field in PREFERENCES_SCHEMA["required"])


def _validate(prefs: dict) -> dict:
    """Blank out a mood value that isn't in the allowed list."""
    prefs = dict(prefs)
    mood = prefs["mood"].strip().lower()
    prefs["mood"] = mood if mood in ALLOWED_MOODS else ""
    return prefs


def _genres_in(text: str) -> list[str]:
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())
    return [w for w in dict.fromkeys(words) if w in KNOWN_GENRES]


def extract_preferences_with_raw(
    user_input: str, client: anthropic.Anthropic | None = None
) -> tuple[dict, str | None]:
    """Like extract_preferences(), but also returns Claude's raw JSON text
    (None when the API wasn't called)."""
    if not user_input or not user_input.strip():
        return dict(INSUFFICIENT_INFO), None

    client = client or anthropic.Anthropic()

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": PREFERENCES_SCHEMA},
        },
        # If Opus 5's safety classifiers decline, the API reruns the request on
        # Anthropic's recommended fallback model instead of returning a refusal.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Request was refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Response was truncated before the JSON was complete")

    raw = next(block.text for block in response.content if block.type == "text")
    prefs = _validate(json.loads(raw))

    if _is_empty(prefs):
        prefs["genre"] = _genres_in(user_input)
    if _is_empty(prefs):
        return dict(INSUFFICIENT_INFO), raw
    return prefs, raw


def extract_preferences(user_input: str, client: anthropic.Anthropic | None = None) -> dict:
    """Extract movie preferences from free-form text.

    Returns a dict with keys genre, mood, exclude, language, or
    {"status": "insufficient_info"} when no preferences could be extracted.
    """
    return extract_preferences_with_raw(user_input, client)[0]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or input("What are you in the mood to watch? ")
    print(json.dumps(extract_preferences(query), indent=2))
