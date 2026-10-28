"""Evaluate agent1.extract_preferences() against the golden dataset.

Each of the four fields in each example is scored as one of:
    correct        golden filled, output filled, values match       (TP)
    wrong          golden filled, output filled, values differ      (FP + FN)
    missed         golden filled, output empty                      (FN)
    hallucinated   golden empty,  output filled                     (FP)
    correct_empty  golden empty,  output empty                      (TN)

Metrics:
    Accuracy           = correct / fields filled in golden
    Hallucination rate = hallucinated / all fields
    Precision          = TP / (TP + FP)
    Recall             = TP / (TP + FN)
    F1                 = harmonic mean of precision and recall

Usage: .venv/bin/python eval_agent1.py
"""

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from agent1 import extract_preferences_with_raw
from golden_dataset import CRITERIA, FIELDS, GOLDEN_DATASET, INSUFFICIENT_INFO

SYNONYMS = {
    "science fiction": "sci fi",
    "scifi": "sci fi",
    "rom com": "romantic comedy",
    "romcom": "romantic comedy",
}


def normalize(value: str) -> str:
    """Lowercase, unify punctuation, apply synonyms, and singularize words."""
    text = value.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    text = SYNONYMS.get(text, text)
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in text.split()]
    return " ".join(words)


def strings_match(expected: str, actual: str) -> bool:
    """Match free-text fields when one's words are a subset of the other's,
    e.g. "space" vs "outer space" or "cozy" vs "cozy and heartwarming"."""
    exp, act = set(normalize(expected).split()), set(normalize(actual).split())
    return bool(exp and act) and (exp <= act or act <= exp)


def lists_match(expected: list, actual: list) -> bool:
    """List fields match when the normalized sets are equal."""
    return {normalize(v) for v in expected} == {normalize(v) for v in actual}


def values_match(field: str, expected, actual) -> bool:
    if isinstance(expected, list):
        return lists_match(expected, actual)
    return strings_match(expected, actual)


def as_fields(result: dict) -> tuple[dict, list[str]]:
    """Convert an extraction (or insufficient_info) into a full field dict.

    Returns the fields plus a list of schema problems (missing keys, nulls,
    wrong types) found in the raw output.
    """
    if result == INSUFFICIENT_INFO:
        return {f: [] if f in ("genre", "exclude") else "" for f in FIELDS}, []

    problems = []
    fields = {}
    for f in FIELDS:
        empty = [] if f in ("genre", "exclude") else ""
        if f not in result:
            problems.append(f"missing '{f}'")
            fields[f] = empty
        elif result[f] is None:
            problems.append(f"'{f}' is null")
            fields[f] = empty
        elif type(result[f]) is not type(empty):
            problems.append(f"'{f}' has type {type(result[f]).__name__}")
            fields[f] = empty
        else:
            fields[f] = result[f]
    return fields, problems


def score_field(field: str, expected, actual, alternatives: list) -> str:
    if not expected:
        return "hallucinated" if actual else "correct_empty"
    if not actual:
        return "missed"
    if any(values_match(field, option, actual) for option in [expected, *alternatives]):
        return "correct"
    return "wrong"


def evaluate_example(example: dict) -> dict:
    error, raw = None, None
    try:
        output, raw = extract_preferences_with_raw(example["input"])
    except Exception as e:  # record the failure and score it as an empty extraction
        output, error = INSUFFICIENT_INFO, f"{type(e).__name__}: {e}"

    expected, _ = as_fields(example["expected"])
    actual, schema_problems = as_fields(output)
    alternatives = example.get("alternatives", {})

    scores = {
        f: score_field(f, expected[f], actual[f], alternatives.get(f, []))
        for f in FIELDS
    }
    status_ok = (example["expected"] == INSUFFICIENT_INFO) == (output == INSUFFICIENT_INFO)
    passed = (
        error is None
        and not schema_problems
        and status_ok
        and all(s in ("correct", "correct_empty") for s in scores.values())
    )
    return {
        "example": example,
        "output": output,
        "raw": raw,
        "scores": scores,
        "schema_problems": schema_problems,
        "status_ok": status_ok,
        "error": error,
        "passed": passed,
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def print_report(results: list[dict]) -> None:
    print("=" * 72)
    print("PER-EXAMPLE RESULTS")
    print("=" * 72)
    for r in results:
        ex = r["example"]
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{mark}] #{ex['id']} {ex['category']}: {ex['input']!r}")
        print(f"  output:   {r['output']}")
        if r["error"]:
            print(f"  ERROR:    {r['error']}")
        if r["schema_problems"]:
            print(f"  schema:   {', '.join(r['schema_problems'])}")
        if not r["status_ok"]:
            print(f"  status:   expected {'insufficient_info' if ex['expected'] == INSUFFICIENT_INFO else 'extraction'}")
        expected, _ = as_fields(ex["expected"])
        actual, _ = as_fields(r["output"])
        for f in FIELDS:
            score = r["scores"][f]
            if score == "correct_empty":
                continue
            print(f"  {f:<9} {score:<13} expected={expected[f]!r} actual={actual[f]!r}")

    counts = Counter(s for r in results for s in r["scores"].values())
    total_fields = len(results) * len(FIELDS)
    mentioned = counts["correct"] + counts["wrong"] + counts["missed"]
    tp = counts["correct"]
    fp = counts["hallucinated"] + counts["wrong"]
    fn = counts["missed"] + counts["wrong"]
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Examples passed:     {sum(r['passed'] for r in results)}/{len(results)}")
    print(f"Field outcomes:      " + ", ".join(f"{k}={counts[k]}" for k in
          ("correct", "wrong", "missed", "hallucinated", "correct_empty")))
    print()
    print(f"Accuracy:            {ratio(tp, mentioned):.1%}  ({tp}/{mentioned} mentioned fields correct)")
    print(f"Hallucination rate:  {ratio(counts['hallucinated'], total_fields):.1%}  "
          f"({counts['hallucinated']}/{total_fields} fields filled that should be empty)")
    print()
    print(f"Precision:           {precision:.3f}  (TP={tp}, FP={fp})")
    print(f"Recall:              {recall:.3f}  (TP={tp}, FN={fn})")
    print(f"F1:                  {f1:.3f}")

    print("\nPass rate by criterion:")
    for key, description in CRITERIA.items():
        tested = [r for r in results if key in r["example"]["tests"]]
        if tested:
            passed = sum(r["passed"] for r in tested)
            print(f"  {key:<18} {passed}/{len(tested)}  {description}")


def print_failures_raw(results: list[dict]) -> None:
    failures = [r for r in results if not r["passed"]]
    if not failures:
        return
    print("\n" + "=" * 72)
    print("RAW JSON FOR FAILING EXAMPLES")
    print("=" * 72)
    for r in failures:
        ex = r["example"]
        print(f"\n#{ex['id']} {ex['category']}: {ex['input']!r}")
        print(f"  golden:        {json.dumps(ex['expected'])}")
        print(f"  claude (raw):  {r['raw'] if r['raw'] is not None else '(no API call)'}")
        print(f"  final output:  {json.dumps(r['output'])}")


def main() -> None:
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(evaluate_example, GOLDEN_DATASET))
    print_report(results)
    print_failures_raw(results)


if __name__ == "__main__":
    main()
