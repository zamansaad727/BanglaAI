import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from html import escape

st.set_page_config(page_title="BanglaAI | Sentiment Intelligence", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "bangla_sentiment_final.joblib")

model = load_model()
LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}
COLORS = {"Positive": "#087F65", "Negative": "#D95C59", "Neutral": "#C18C32"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Hind+Siliguri:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans','Hind Siliguri',sans-serif;}
[data-testid="stAppViewContainer"] {
 background: radial-gradient(ellipse at 7% 5%, rgba(171,211,190,.33), transparent 35%),
 radial-gradient(ellipse at 95% 20%, rgba(235,207,171,.29), transparent 36%),
 linear-gradient(155deg,#F9F8F2 0%,#F3F7F2 58%,#F9F4EC 100%);
}
.block-container {width:94% !important;max-width:1600px !important;padding:2.4rem 1rem 3rem !important;margin-left:auto !important;margin-right:auto !important;}
[data-testid="stHeader"] {background:transparent;}
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#102B32,#14594F 75%,#1A6D5D);padding:3.3rem 3.4rem;border-radius:26px;color:#fff;box-shadow:0 20px 48px rgba(13,58,54,.14);margin-bottom:1.4rem;}
.hero:after {content:'অ';position:absolute;right:4%;top:-60px;font-family:'Hind Siliguri',sans-serif;font-size:260px;font-weight:700;color:rgba(255,255,255,.055);pointer-events:none;line-height:1.4;}
.hero .eyebrow {letter-spacing:.2em;text-transform:uppercase;color:#BFE5D4;font-size:12px;font-weight:700;}
.hero h1 {font-family:'Hind Siliguri','DM Sans',sans-serif;font-size:clamp(42px,5vw,68px);font-weight:700;color:#fff;line-height:1.15;margin:.7rem 0 .4rem;}
.hero h1 span {color:#F0C58A;}
.hero .tagline {font-size:18px;color:#F5E9D5;font-weight:600;}
.hero .description {color:#D4E7E0;font-size:15px;max-width:650px;line-height:1.7;margin-top:1.1rem;}
.hero .mini-pill {display:inline-block;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.08);padding:6px 12px;border-radius:50px;font-size:12px;color:#E8F4ED;margin-top:10px;}
[data-baseweb="tab-list"] {gap:12px;}
button[data-baseweb="tab"] {font-weight:700;border-radius:12px 12px 0 0;padding:12px 18px;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:18px;}
[data-testid="stMetric"] {background:rgba(255,255,255,.78);border:1px solid #E4E9E1;border-radius:15px;padding:18px;box-shadow:0 5px 18px rgba(30,68,56,.035);}
[data-testid="stMetricLabel"] {font-weight:700;}
.stButton button[kind="primary"], .stDownloadButton button {border-radius:12px;font-weight:700;min-height:46px;}
.stTextArea textarea, [data-testid="stFileUploader"] {border-radius:12px;}
.section-lead {color:#64766F;font-size:14px;margin-bottom:1.2rem;}
.result-card {background:#fff;border:1px solid #E0E9E2;border-radius:18px;padding:24px 26px;box-shadow:0 10px 25px rgba(23,70,59,.06);margin-bottom:18px;}
.result-card .eyebrow {color:#64766F;font-size:12px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;}
.result-card .sentiment {font-size:35px;font-weight:800;margin:8px 0;}
.result-card .confidence {color:#50635D;font-size:14px;}
.overview {display:flex;height:18px;border-radius:99px;overflow:hidden;background:#E5EBE5;margin:18px 0 12px;}
.legend {display:flex;gap:22px;flex-wrap:wrap;margin:0 0 20px;color:#42564F;font-size:13px;}
.legend b {font-weight:700;}
.dot {display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;}
.footer {background:#102B32;border-radius:22px;text-align:center;padding:32px 16px;margin-top:26px;}
.footer .logo {font-family:'Hind Siliguri',sans-serif;color:white;font-size:27px;font-weight:700;}
.footer .tag {color:#C7DED4;font-size:13px;margin:7px 0 19px;}
.footer .credit {color:#F0C58A;font-size:14px;font-weight:700;}
.footer .copy {color:#91AFA4;font-size:11px;margin-top:10px;}
@media(max-width:700px){.hero{padding:2rem 1.5rem}.hero:after{font-size:170px;top:-25px}.block-container{width:100% !important;padding:1.2rem 1rem 2rem !important}.legend{gap:10px}}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="eyebrow">Bangla Sentiment Intelligence</div><h1>বাংলা<span>AI</span></h1><div class="tagline">Rooted in Bangla. Built for the World.</div><div class="description">Understand the sentiment behind Bangla text. From a single opinion to hundreds of customer reviews, turn feedback into clearer insights.</div><div class="mini-pill">✦ TF-IDF &nbsp; · &nbsp; Logistic Regression &nbsp; · &nbsp; Bangla NLP</div></div>""", unsafe_allow_html=True)

tab_single, tab_batch, tab_about = st.tabs(["✍️ Single Analysis", "📁 Batch Analysis", "ℹ️ About"])

with tab_single:
    st.header("Sentiment Studio")
    st.markdown('<div class="section-lead">Explore the tone of a Bangla review, comment, or opinion.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.subheader("Your text")
        user_text = st.text_area("Enter Bangla text", placeholder="এই পণ্যটির মান খুব ভালো। আমি সন্তুষ্ট।", height=180)
        analyze = st.button("✦ Analyze Sentiment", type="primary", use_container_width=True)
    with right:
        st.subheader("Analysis result")
        if analyze and user_text.strip():
            prediction = int(model.predict([user_text.strip()])[0])
            probabilities = model.predict_proba([user_text.strip()])[0]
            sentiment = LABELS[prediction]
            confidence = float(max(probabilities)) * 100
            st.markdown(f'<div class="result-card"><div class="eyebrow">Predicted sentiment</div><div class="sentiment" style="color:{COLORS[sentiment]}">{escape(sentiment)}</div><div class="confidence">Model confidence · <strong>{confidence:.1f}%</strong></div></div>', unsafe_allow_html=True)
            st.markdown("**Class probabilities**")
            for label, probability in zip(model.classes_, probabilities):
                name = LABELS[int(label)]
                st.write(f"{name} · {probability * 100:.1f}%")
                st.progress(float(probability))
            st.caption("Model confidence is an estimate, not a guarantee of correctness.")
        elif analyze:
            st.warning("Please enter Bangla text first.")
        else:
            st.info("Enter Bangla text and select Analyze Sentiment to see the result.")

with tab_batch:
    st.header("Batch Review Analysis")
    st.markdown('<div class="section-lead">Upload a CSV or Excel file, choose the review column, and analyze multiple texts together.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            else:
                df = pd.read_excel(uploaded_file)
            if df.empty or len(df.columns) == 0:
                st.warning("The uploaded file has no review rows or columns.")
            else:
                st.markdown("#### Dataset preview")
                st.dataframe(df.head(5), use_container_width=True)
                text_column = st.selectbox("Select the column containing Bangla reviews", df.columns.tolist())
                if st.button("✦ Analyze All Reviews", type="primary", use_container_width=True):
                    results = df.dropna(subset=[text_column]).copy()
                    results[text_column] = results[text_column].astype(str).str.strip()
                    results = results.loc[results[text_column] != ""].copy()
                    if results.empty:
                        st.warning("No valid reviews found in the selected column.")
                    else:
                        predictions = model.predict(results[text_column].tolist())
                        results["Predicted_Sentiment"] = [LABELS[int(p)] for p in predictions]
                        counts = results["Predicted_Sentiment"].value_counts()
                        total = len(results)
                        st.success(f"Successfully analyzed {total:,} reviews!")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🟢 Positive", int(counts.get("Positive", 0)))
                        c2.metric("🔴 Negative", int(counts.get("Negative", 0)))
                        c3.metric("🟡 Neutral", int(counts.get("Neutral", 0)))
                        st.markdown("#### Sentiment overview")
                        percentages = {name: float(counts.get(name, 0)) / total * 100 for name in ["Positive", "Negative", "Neutral"]}
                        segments = "".join(f'<div style="width:{percentages[name]:.5f}%;background:{COLORS[name]}"></div>' for name in ["Positive", "Negative", "Neutral"])
                        st.markdown(f'<div class="overview">{segments}</div>', unsafe_allow_html=True)
                        legend = "".join(f'<span><i class="dot" style="background:{COLORS[name]}"></i>{name} <b>{percentages[name]:.1f}%</b></span>' for name in ["Positive", "Negative", "Neutral"])
                        st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
                        st.markdown("#### Analysis results")
                        st.dataframe(results, use_container_width=True)
                        csv_data = results.to_csv(index=False).encode("utf-8-sig")
                        st.download_button("⬇ Download Results (CSV)", data=csv_data, file_name="banglaai_results.csv", mime="text/csv", use_container_width=True)
                        st.caption("These are automated predictions, not verified sentiment labels.")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Unable to read this file: {error}. For CSV files, use UTF-8 and put reviews containing commas in double quotes.")
        except Exception as error:
            st.error(f"Unable to process the uploaded file: {error}")

with tab_about:
    st.header("About BanglaAI")
    st.write("BanglaAI is a machine learning application for classifying Bangla text as Positive, Negative, or Neutral. It supports individual feedback and batch review analysis.")
    st.markdown("#### Technology")
    st.write("**Python** · **Streamlit** · **TF-IDF** · **Logistic Regression**")
    st.markdown("#### Model evaluation")
    st.caption("Scores are from the project's held-out BLUGE test set and separate external evaluations.")
    st.dataframe(pd.DataFrame({"Dataset": ["BLUGE test", "BECS external", "Mendeley external"], "Accuracy": ["76.19%", "54.80%", "62.45%"], "Macro F1": ["76.21%", "53.52%", "52.96%"]}), hide_index=True, use_container_width=True)
    st.info("Predictions can be incorrect, especially for ambiguous or neutral reviews. External dataset performance is lower than the BLUGE test result. Treat output as a screening aid, not a definitive judgment.")

st.markdown('<div class="footer"><div class="logo">বাংলা<span style="color:#F0C58A">AI</span></div><div class="tag">Rooted in Bangla. Built for the World.</div><div class="credit">Designed &amp; Developed by ASAD MD</div><div class="copy">© 2026 BanglaAI</div></div>', unsafe_allow_html=True)
