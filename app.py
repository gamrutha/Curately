"""Curately: a local Streamlit app running Agent 1 -> Agent 2 -> Agent 3.

Run with: streamlit run app.py
"""

import anthropic
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # ANTHROPIC_API_KEY from .env

from agent1 import extract_preferences  # noqa: E402
from agent2 import build_index, prefs_to_text, recommend  # noqa: E402
from agent3 import explain  # noqa: E402

MAX_SEARCHES = 20
MORE_DETAIL = (
    "I couldn't find enough to go on. I can search by genre, mood, language, or things to avoid "
    "(for example 'a light comedy, nothing scary'). I can't search by actor or director yet."
)
FOOTER = (
    "Movie data: MovieLens, IMDb ratings, TMDB descriptions. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@st.cache_resource(show_spinner="Loading the movie index...")
def get_collection():
    return build_index()


def run_pipeline(query: str) -> dict:
    client = get_client()
    prefs = extract_preferences(query, client)
    response = recommend(prefs_to_text(prefs), collection=get_collection())
    return {"prefs": prefs, "result": explain(query, prefs, response, client)}


def show(run: dict) -> None:
    result = run["result"]
    recommendations = result["recommendations"]

    if not recommendations:
        st.info(MORE_DETAIL)
    else:
        if not result["explanations_ok"]:
            st.caption("Descriptions are unavailable right now.")
        for rec in recommendations:
            st.subheader(f"{rec['rank']}. {rec['title']}")
            if result["explanations_ok"] and rec["explanation"]:
                st.write(rec["explanation"])

    with st.expander("How this was decided", expanded=False):
        st.write("What we understood from your request (Agent 1):")
        st.json(run["prefs"])
        st.write(f"Confidence score: {result['debug']['confidence_score']}")


st.set_page_config(page_title="Curately")
st.title("Curately")

st.session_state.setdefault("searches", 0)
query = st.text_input("What do you feel like watching?")

if st.button("Search"):
    if st.session_state.searches >= MAX_SEARCHES:
        st.warning(f"You've reached the limit of {MAX_SEARCHES} searches for this session.")
    else:
        st.session_state.searches += 1
        with st.spinner("Finding movies for you..."):
            try:
                st.session_state.last_run = run_pipeline(query)
            except Exception:  # API or network failure: keep the page usable
                st.session_state.pop("last_run", None)
                st.error("Something went wrong while searching. Please try again in a moment.")

if "last_run" in st.session_state:
    show(st.session_state.last_run)

st.caption(f"{MAX_SEARCHES - st.session_state.searches} of {MAX_SEARCHES} searches left this session.")
st.divider()
st.caption(FOOTER)
