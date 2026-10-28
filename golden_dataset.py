"""Golden dataset for evaluating agent1.extract_preferences().

Each example has:
- category: which input style it covers
- input: the raw user text
- expected: the correct extraction, or {"status": "insufficient_info"}
- alternatives: other acceptable values per field, for places where more
  than one wording is equally correct (optional)
- tests: which extraction rules (see CRITERIA) the example exercises
"""

FIELDS = ["genre", "mood", "exclude", "language"]

INSUFFICIENT_INFO = {"status": "insufficient_info"}

CRITERIA = {
    "schema": "All four fields present; [] or \"\" for unmentioned fields, never null",
    "no_invention": "No values for fields the user didn't mention",
    "negation": "Negated preferences go in exclude, not dropped or put elsewhere",
    "insufficient_info": "Returns {status: insufficient_info} when nothing is extractable",
    "no_bias": "Doesn't infer preferences from demographics or stereotypes",
    "multi_field": "Extracts several fields from one message correctly",
    "language_explicit": "Fills language only when a language is named",
}

GOLDEN_DATASET = [
    {
        "id": 1,
        "category": "clear request",
        "input": "I want something funny but not too silly, maybe with some heart",
        "expected": {"genre": ["comedy"], "mood": "heartfelt", "exclude": ["slapstick"], "language": ""},
        "alternatives": {"exclude": [["silly"]]},
        "tests": ["multi_field", "negation"],
    },
    {
        "id": 2,
        "category": "vague input",
        "input": "something good",
        "expected": INSUFFICIENT_INFO,
        "tests": ["insufficient_info", "no_invention"],
    },
    {
        "id": 3,
        "category": "bias-testing",
        "input": "I want to watch a documentary about space exploration",
        "expected": {"genre": ["documentary"], "mood": "", "exclude": [], "language": ""},
        "tests": ["no_bias", "no_invention"],
    },
    {
        "id": 4,
        "category": "minimal input",
        "input": "action",
        "expected": {"genre": ["action"], "mood": "", "exclude": [], "language": ""},
        "tests": ["schema", "no_invention"],
    },
    {
        "id": 5,
        "category": "negation",
        "input": "definitely not a horror movie, something light",
        "expected": {"genre": [], "mood": "light", "exclude": ["horror"], "language": ""},
        "tests": ["negation", "no_invention"],
    },
    {
        "id": 6,
        "category": "multi-attribute",
        "input": "a mix of comedy and romance, maybe set in another country",
        "expected": {"genre": ["comedy", "romance"], "mood": "", "exclude": [], "language": ""},
        "tests": ["multi_field", "language_explicit"],
    },
    {
        "id": 7,
        "category": "contradictory input",
        "input": "I want something that's both the scariest and funniest movie ever",
        "expected": {"genre": ["horror", "comedy"], "mood": "", "exclude": [], "language": ""},
        "tests": ["multi_field", "no_invention"],
    },
    {
        "id": 8,
        "category": "language-specific",
        "input": "something in Spanish, a good thriller",
        "expected": {"genre": ["thriller"], "mood": "", "exclude": [], "language": "Spanish"},
        "tests": ["language_explicit", "multi_field"],
    },
    {
        "id": 9,
        "category": "single-word input",
        "input": "sad",
        "expected": {"genre": [], "mood": "sad", "exclude": [], "language": ""},
        "tests": ["schema", "no_invention"],
    },
    {
        "id": 10,
        "category": "off-topic/malformed input",
        "input": "asdkjaslkdj what's the weather today",
        "expected": INSUFFICIENT_INFO,
        "tests": ["insufficient_info", "schema"],
    },
]
