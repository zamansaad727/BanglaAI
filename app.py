
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from io import BytesIO

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="BanglaAI | Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "bangla_sentiment_final.joblib"
    return joblib.load(model_path)

model = load_model()

sentiment_labels = {
    0: "Neutral",
    1: "Positive",
    2: "Negative"
}

sentiment_colors = {
    "Positive": "#087F65",
    "Negative": "#D94F4F",
    "Neutral": "#D6A340"
}

# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #102B32, #155D57);
    border-radius: 22px;
    padding: 42px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #D6EAE5;
    font-size: 17px;
}

.brand-accent {
    color: #E8B86D;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #102B32;
    margin-bottom: 12px;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8E5;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #64748B;
    padding-top: 25px;
    font-size: 13px;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    padding: 0.65rem 1.5rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HERO SECTION
# ==========================================

st.markdown("""
<div class="hero">
    <h1>বাংলা<span class="brand-accent">AI</span></h1>
    <p><strong>Rooted in Bangla. Built for the World.</strong></p>
    <p>
        Discover the sentiment behind Bangla text.
        Analyze individual opinions and customer reviews
        with Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. NAVIGATION TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "✍️ Single Analysis",
    "📁 Batch Analysis",
    "ℹ️ About"
])

# ==========================================
# 6. SINGLE TEXT ANALYSIS
# ==========================================

with tab1:

    st.markdown("## Sentiment Studio")

    st.caption(
        "Analyze Bangla customer reviews, "
        "feedback, comments, and opinions."
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:

        st.subheader("Enter Bangla Text")

        user_text = st.text_area(
            "Your Bangla sentence",
            placeholder="এই পণ্যটির মান খুব ভালো। আমি সন্তুষ্ট।",
            height=180
        )

        analyze = st.button(
            "Analyze Sentiment",
            type="primary",
            use_container_width=True
        )

    with right:

        st.subheader("Analysis Result")

        if analyze:

            if user_text.strip():

                prediction = model.predict([user_text])[0]

                probabilities = model.predict_proba(
                    [user_text]
                )[0]

                sentiment = sentiment_labels[int(prediction)]

                confidence = max(probabilities) * 100

                selected_color = sentiment_colors[sentiment]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <p style="color:#64748B;">
                            Predicted Sentiment
                        </p>
                        <h2 style="color:{selected_color};">
                            {sentiment}
                        </h2>
                        <p>
                            Model Confidence:
                            <strong>{confidence:.2f}%</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### Class Probabilities")

                for label, probability in zip(
                    model.classes_, probabilities
                ):

                    class_name = sentiment_labels[int(label)]

                    st.write(
                        f"{class_name}: {probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

            else:
                st.warning("Please enter Bangla text first.")

        else:
            st.info(
                "Enter Bangla text and click Analyze "
                "to see the sentiment prediction."
            )

# ==========================================
# 7. BATCH SENTIMENT ANALYSIS
# ==========================================

with tab2:

    st.markdown("## Batch Sentiment Analysis")

    st.write(
        "Upload a CSV or Excel file containing Bangla "
        "customer reviews to analyze multiple texts at once."
    )

    uploaded_file = st.file_uploader(
        "Upload your review dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            # Read uploaded file
            if uploaded_file.name.lower().endswith(".csv"):

                try:
                    df = pd.read_csv(uploaded_file)

                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file,
                        encoding="utf-8-sig"
                    )

            else:
                df = pd.read_excel(uploaded_file)

            st.markdown("### Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            # Select review column
            text_column = st.selectbox(
                "Select the column containing Bangla reviews",
                options=df.columns.tolist()
            )

            if st.button(
                "Analyze All Reviews",
                type="primary",
                use_container_width=True
            ):

                # Remove empty reviews
                results = df.copy()

                results = results.dropna(
                    subset=[text_column]
                )

                results[text_column] = (
                    results[text_column]
                    .astype(str)
                    .str.strip()
                )

                results = results[
                    results[text_column] != ""
                ].copy()

                if results.empty:

                    st.warning("No valid reviews found.")

                else:

                    # Predict all reviews
                    predictions = model.predict(
                        results[text_column].tolist()
                    )

                    results["Predicted_Sentiment"] = [
                        sentiment_labels[int(p)]
                        for p in predictions
                    ]

                    st.success(
                        f"Successfully analyzed {len(results)} reviews!"
                    )

                    # Sentiment counts
                    counts = results[
                        "Predicted_Sentiment"
                    ].value_counts()

                    # Summary cards
                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "🟢 Positive",
                        int(counts.get("Positive", 0))
                    )

                    c2.metric(
                        "🔴 Negative",
                        int(counts.get("Negative", 0))
                    )

                    c3.metric(
                        "🟡 Neutral",
                        int(counts.get("Neutral", 0))
                    )

                    # ==================================
                    # SENTIMENT OVERVIEW
                    # ==================================

                    st.markdown("### Sentiment Overview")

                    total = len(results)

                    positive_pct = (
                        counts.get("Positive", 0) / total * 100
                    )

                    negative_pct = (
                        counts.get("Negative", 0) / total * 100
                    )

                    neutral_pct = (
                        counts.get("Neutral", 0) / total * 100
                    )

                    st.markdown(
                        f"""
                        <div style="
                            display:flex;
                            height:18px;
                            border-radius:12px;
                            overflow:hidden;
                            background:#E5E7EB;
                            margin:16px 0 15px 0;
                        ">
                            <div style="
                                width:{positive_pct}%;
                                background:#087F65;
                            "></div>

                            <div style="
                                width:{negative_pct}%;
                                background:#D94F4F;
                            "></div>

                            <div style="
                                width:{neutral_pct}%;
                                background:#D6A340;
                            "></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    p1, p2, p3 = st.columns(3)

                    p1.caption(
                        f"🟢 Positive: {positive_pct:.1f}%"
                    )

                    p2.caption(
                        f"🔴 Negative: {negative_pct:.1f}%"
                    )

                    p3.caption(
                        f"🟡 Neutral: {neutral_pct:.1f}%"
                    )

                    # ==================================
                    # ANALYSIS RESULTS
                    # ==================================

                    st.markdown("### Analysis Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # ==================================
                    # DOWNLOAD RESULTS
                    # ==================================

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "⬇️ Download Results (CSV)",
                        data=csv_data,
                        file_name="banglaai_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to process file: {e}"
            )

# ==========================================
# 8. ABOUT SECTION
# ==========================================

with tab3:

    st.markdown("## About BanglaAI")

    st.write(
        "BanglaAI is a Machine Learning-based "
        "Bangla text sentiment classification system."
    )

    st.markdown("### Technology")

    st.write("**Language:** Python")
    st.write("**Framework:** Streamlit")
    st.write("**Text Representation:** TF-IDF")
    st.write("**Classifier:** Logistic Regression")

    st.markdown("### Sentiment Categories")

    st.write("🟢 Positive")
    st.write("🔴 Negative")
    st.write("🟡 Neutral")

    st.markdown("### Model Performance")

    st.write("BLUGE Test Accuracy: 76.19%")
    st.write("BLUGE Test Macro F1: 76.21%")

    st.info(
        "Predictions are generated by a statistical "
        "Machine Learning model and may contain errors."
    )

# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>BanglaAI</strong><br>
        Rooted in Bangla. Built for the World.<br>
        Powered by TF-IDF + Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)
