"""
frontend/app.py
---------------
Streamlit web interface for the SFL Text Complexity Analyzer.

This app provides a user-friendly interface for the FastAPI backend.
Users can paste text, click Analyze, and see a detailed complexity
report including the predicted level, confidence score, and SFL
feature breakdown with visualizations.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# --- Configuration ---

# Pointing directly to your live Render API URL
API_BASE_URL = "https://nlp-text-complexity-analyzer.onrender.com"

# --- Page Configuration ---

st.set_page_config(
    page_title="SFL Text Complexity Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
 )

# --- Custom CSS for a cleaner look ---

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .level-elementary { color: #2e7d32; font-weight: 700; font-size: 1.8rem; }
    .level-intermediate { color: #e65100; font-weight: 700; font-size: 1.8rem; }
    .level-advanced { color: #b71c1c; font-weight: 700; font-size: 1.8rem; }
    .feature-label { font-weight: 600; color: #1f4e79; }
    .interpretation-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f4e79;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/R_logo.svg/200px-R_logo.svg.png", width=50 )
    st.markdown("## About This Tool")
    st.markdown("""
    This tool analyzes text complexity using features grounded in
    **Systemic Functional Linguistics (SFL)** theory.

    Unlike traditional readability formulas (Flesch-Kincaid) that
    simply count syllables and words, this analyzer measures the
    **grammatical mechanisms** that make text cognitively demanding.
    """)

    st.markdown("---")
    st.markdown("### The 5 SFL Features")

    features_info = {
        "Lexical Density": "Proportion of content words (nouns, verbs, adjectives) to total tokens. Higher = more information-dense.",
        "Nominalization Density": "Frequency of process-to-noun conversions (e.g., 'destroy' → 'destruction'). The key marker of academic writing.",
        "Clausal Complexity": "Ratio of subordinate clauses to total clauses. Higher = more embedded, complex sentence structures.",
        "Mean Length of Clause": "Average number of tokens per clause. Longer clauses = more information packed per unit.",
        "Passive Voice Ratio": "Proportion of passive sentences. Higher = more formal, impersonal register."
    }

    for feature, description in features_info.items():
        with st.expander(feature):
            st.write(description)

    st.markdown("---")
    st.markdown("**Built by:** Mojtaba Sayyad Mahernia")
    st.markdown("**GitHub:** [nlp-text-complexity-analyzer](https://github.com/mojtabasayyadmahernia/nlp-text-complexity-analyzer )")

# --- Main Content ---

st.markdown('<p class="main-header">📝 SFL Text Complexity Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Classify text complexity using Systemic Functional Linguistics, developed by Mojtaba Sayyad Mahernia.</p>', unsafe_allow_html=True)

# --- Input Section ---

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Paste Your Text Below")
    user_text = st.text_area(
        label="Input text",
        placeholder="Paste any text here — an academic paper, a news article, a children's book, or anything else you want to analyze...",
        height=250,
        label_visibility="collapsed"
    )

    analyze_button = st.button("🔍 Analyze Complexity", type="primary", use_container_width=True)

with col2:
    st.markdown("### Example Texts")
    st.markdown("Click to load an example:")

    if st.button("📚 Academic (Advanced)", use_container_width=True):
        st.session_state["example_text"] = (
            "The operationalization of theoretical constructs within the framework "
            "of systemic functional linguistics necessitates a reconceptualization "
            "of the relationship between grammatical metaphor and ideational meaning. "
            "The realization of experiential meaning through nominalization has been "
            "extensively documented in the literature on academic discourse. "
            "The transformation of verbal processes into nominal groups constitutes "
            "a fundamental mechanism of register variation in written language."
        )

    if st.button("📰 News Article (Intermediate)", use_container_width=True):
        st.session_state["example_text"] = (
            "Scientists have discovered a new species of deep-sea fish that can "
            "survive at extreme depths. The creature, found in the Pacific Ocean, "
            "has adapted to the high pressure environment over millions of years. "
            "Researchers from three universities collaborated on the study, which "
            "was published in the journal Nature. The discovery could help scientists "
            "better understand how life evolves in extreme conditions."
        )

    if st.button("📖 Children's Book (Elementary)", use_container_width=True):
        st.session_state["example_text"] = (
            "The little cat sat on the mat. It was a warm sunny day. "
            "The cat liked to play in the garden. It ran after the red ball. "
            "The ball went under the big tree. The cat looked for the ball. "
            "It found the ball and was very happy. Then it went home for dinner."
        )

    # Load example text if button was clicked
    if "example_text" in st.session_state:
        user_text = st.session_state["example_text"]
        del st.session_state["example_text"]

# --- Analysis and Results ---

if analyze_button and user_text:
    if len(user_text) < 50:
        st.error("⚠️ Text is too short. Please provide at least 50 characters for a meaningful analysis.")
    else:
        with st.spinner("Analyzing text complexity... (Note: The free API on Render may take 30-50 seconds to wake up if inactive)"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={"text": user_text},
                    timeout=60 # Increased timeout to handle Render's cold start
                )

                if response.status_code == 200:
                    result = response.json()

                    st.markdown("---")
                    st.markdown("## Analysis Results")

                    # --- Top-level result ---
                    col_level, col_confidence, col_spacer = st.columns([1, 1, 2])

                    level = result["predicted_level"]
                    confidence = result["confidence"]
                    level_class = f"level-{level.lower()}"

                    with col_level:
                        st.markdown("**Predicted Level**")
                        st.markdown(f'<p class="{level_class}">{level}</p>', unsafe_allow_html=True)

                    with col_confidence:
                        st.markdown("**Model Confidence**")
                        st.markdown(f'<p style="font-size:1.8rem; font-weight:700; color:#1f4e79;">{confidence:.0%}</p>', unsafe_allow_html=True)

                    # --- Confidence Gauge ---
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        title={"text": "Confidence Score (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1f4e79"},
                            "steps": [
                                {"range": [0, 50], "color": "#ffcccc"},
                                {"range": [50, 75], "color": "#fff3cc"},
                                {"range": [75, 100], "color": "#ccffcc"},
                            ],
                        }
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # --- SFL Feature Breakdown ---
                    st.markdown("### SFL Feature Breakdown")

                    features = result["sfl_features"]
                    feature_display_names = {
                        "lexical_density": "Lexical Density",
                        "nominalization_density": "Nominalization Density\n(per 100 tokens)",
                        "clausal_complexity": "Clausal Complexity",
                        "mean_length_of_clause": "Mean Length of Clause\n(tokens)",
                        "passive_voice_ratio": "Passive Voice Ratio"
                    }

                    feature_df = pd.DataFrame({
                        "Feature": [feature_display_names.get(k, k) for k in features.keys()],
                        "Value": [round(v, 3) for v in features.values()]
                    })

                    fig_bar = px.bar(
                        feature_df,
                        x="Value",
                        y="Feature",
                        orientation="h",
                        color="Value",
                        color_continuous_scale=["#cce5ff", "#1f4e79"],
                        title="SFL Feature Values"
                    )
                    fig_bar.update_layout(
                        height=350,
                        showlegend=False,
                        coloraxis_showscale=False,
                        margin=dict(t=40, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # --- Interpretation ---
                    st.markdown("### Linguistic Interpretation")
                    st.markdown(
                        f'<div class="interpretation-box">{result["interpretation"]}</div>',
                        unsafe_allow_html=True
                    )

                    # --- Raw Feature Table ---
                    with st.expander("View Raw Feature Values"):
                        st.dataframe(feature_df, use_container_width=True)

                elif response.status_code == 422:
                    error_detail = response.json().get("detail", "Validation error.")
                    st.error(f"⚠️ {error_detail}")
                else:
                    st.error(f"API returned an unexpected error (status {response.status_code}). Please try again.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "⚠️ Could not connect to the API. "
                    "Make sure the Render service is running."
                )
            except requests.exceptions.Timeout:
                st.error("⚠️ The request timed out. The free Render server might be waking up from sleep. Please wait a minute and try again.")

elif analyze_button and not user_text:
    st.warning("Please paste some text before clicking Analyze.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<small>Built with FastAPI + Streamlit + scikit-learn | Deployed on Render | "
    "Theoretical framework: Halliday's Systemic Functional Linguistics</small>",
    unsafe_allow_html=True
)