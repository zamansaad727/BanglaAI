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
:root { --ink:#142F34; --emerald:#087F65; --gold:#E8B86D; }
html,body,[data-testid="stApp"] {font-family:'DM Sans','Hind Siliguri',sans-serif;color:var(--ink);}
[data-testid="stAppViewContainer"] {
 background:
 radial-gradient(ellipse 42% 42% at 0% 0%,rgba(144,190,164,.27),transparent 90%),
 radial-gradient(ellipse 35% 42% at 100% 8%,rgba(232,192,139,.23),transparent 90%),
 radial-gradient(ellipse 45% 45% at 85% 95%,rgba(159,204,190,.16),transparent 95%),
 linear-gradient(132deg,#FBFAF6 0%,#F2F7F3 50%,#FAF5ED 100%);
 background-attachment:fixed;
}
[data-testid="stHeader"] {background:transparent;}
/* Override Streamlit's default narrow content constraint. */
[data-testid="stMainBlockContainer"], .block-container {
 width: min(94vw, 1550px) !important;
 max-width:1550px !important;
 padding:2.4rem 1.5rem 3rem !important;
 margin-inline:auto !important;
}
[data-testid="stAppViewBlockContainer"] {max-width:none !important;}
h1,h2,h3 {color:#17343A;letter-spacing:-.025em;}
[data-testid="stMarkdownContainer"] p, .stCaption {line-height:1.65;}
.hero {
 position:relative;overflow:hidden;
 background:linear-gradient(116deg,#102D33 0%,#15544E 62%,#197464 100%);
 border:1px solid rgba(255,255,255,.13);border-radius:28px;
 padding:clamp(2.5rem,5vw,4.8rem) clamp(2rem,5vw,5.3rem);
 color:white;box-shadow:0 22px 55px rgba(19,71,59,.15);margin-bottom:1.8rem;
}
.hero:before {content:'';position:absolute;right:-8%;bottom:-120px;width:460px;height:460px;border-radius:50%;border:1px solid rgba(255,255,255,.10);box-shadow:0 0 0 65px rgba(255,255,255,.025),0 0 0 130px rgba(255,255,255,.018);pointer-events:none;}
.hero:after {content:'অ';position:absolute;right:7%;top:-80px;font-family:'Hind Siliguri',sans-serif;font-size:clamp(190px,24vw,350px);font-weight:700;color:rgba(255,255,255,.055);pointer-events:none;line-height:1.5;}
.hero > * {position:relative;z-index:1;}
.hero .eyebrow {letter-spacing:.19em;text-transform:uppercase;color:#B8DFD1;font-size:13px;font-weight:800;}
.hero h1 {font-family:'Hind Siliguri','DM Sans',sans-serif;font-size:clamp(54px,6vw,82px);font-weight:700;color:white;line-height:1.16;margin:.8rem 0 .7rem;}
.hero h1 span {color:#F3C889;}
.hero .tagline {font-size:clamp(18px,2vw,23px);color:#F7E8D1;font-weight:700;}
.hero .description {color:#E1EFE9;font-size:clamp(15px,1.25vw,18px);max-width:760px;line-height:1.8;margin-top:1.1rem;}
.hero .mini-pill {display:inline-block;border:1px solid rgba(255,255,255,.28);background:rgba(255,255,255,.09);padding:9px 16px;border-radius:50px;font-size:13px;color:#EDF8F1;margin-top:14px;}
[data-baseweb="tab-list"] {gap:14px;border-bottom:1px solid #D9E6DE;}
button[data-baseweb="tab"] {font-weight:700;font-size:15px;border-radius:12px 12px 0 0;padding:14px 20px;}
[data-testid="stTextArea"] textarea {font-family:'Hind Siliguri','DM Sans',sans-serif;font-size:17px !important;line-height:1.8 !important;border-radius:15px !important;background:#FFFFFF !important;border:1px solid #D8E5DE !important;}
[data-testid="stFileUploader"] {border-radius:16px;}
[data-testid="stMetric"] {background:rgba(255,255,255,.88);border:1px solid #DFE9E2;border-radius:18px;padding:22px;box-shadow:0 9px 30px rgba(23,70,59,.045);}
[data-testid="stMetricLabel"] {font-weight:700;}
.stButton button[kind="primary"], .stDownloadButton button {border-radius:12px;font-weight:700;min-height:51px;font-size:15px;box-shadow:0 6px 16px rgba(8,127,101,.11);}
.section-lead {color:#64786F;font-size:16px;margin-bottom:1.6rem;}
.result-card {background:rgba(255,255,255,.94);border:1px solid #DDE8E1;border-radius:20px;padding:30px 32px;box-shadow:0 14px 35px rgba(23,70,59,.07);margin-bottom:22px;}
.result-card .eyebrow {color:#61776D;font-size:13px;text-transform:uppercase;letter-spacing:.11em;font-weight:700;}
.result-card .sentiment {font-size:clamp(36px,3vw,48px);font-weight:800;margin:10px 0;}
.result-card .confidence {color:#4D655C;font-size:16px;}
.overview {display:flex;height:20px;border-radius:99px;overflow:hidden;background:#E5EBE5;margin:20px 0 14px;}
.legend {display:flex;gap:24px;flex-wrap:wrap;margin:0 0 24px;color:#42564F;font-size:15px;}
.legend b {font-weight:700;}
.dot {display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;}
.footer {background:linear-gradient(120deg,#102B32,#164C48);border:1px solid rgba(255,255,255,.07);border-radius:24px;text-align:center;padding:42px 18px;margin-top:34px;box-shadow:0 14px 35px rgba(13,48,46,.10);}
.footer .logo {font-family:'Hind Siliguri',sans-serif;color:white;font-size:33px;font-weight:700;}
.footer .tag {color:#C7DED4;font-size:15px;margin:9px 0 22px;}
.footer .credit {color:#F0C58A;font-size:16px;font-weight:700;}
.footer .copy {color:#A4C0B5;font-size:12px;margin-top:13px;}
@media(max-width:800px){
 [data-testid="stMainBlockContainer"],.block-container{width:100% !important;padding:1.2rem 1rem 2rem !important;}
 .hero{padding:2.4rem 1.5rem;border-radius:21px;}
 .hero:after{font-size:170px;right:-1%;top:-30px;}
 .hero h1{font-size:53px;}
 .legend{gap:12px;}
 button[data-baseweb="tab"]{padding:10px 8px;font-size:12px;}
 .footer{padding:30px 14px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="eyebrow">Bangla Sentiment Intelligence</div><h1 style="color:#FFFFFF;">বাংলা<span style="color:#FFFFFF;">AI</span></h1><div class="tagline">Rooted in Bangla. Built for the World.</div><div class="description">Understand the sentiment behind Bangla text. From a single opinion to hundreds of customer reviews, turn feedback into clearer insights.</div><div class="mini-pill">✦ TF-IDF &nbsp; · &nbsp; Logistic Regression &nbsp; · &nbsp; Bangla NLP</div></div>""", unsafe_allow_html=True)

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

st.markdown('<div class="footer"><div class="logo">বাংলা<span style="color:#F0C58A">AI</span></div><div class="credit">Designed &amp; Developed by ASAD MD</div><div class="copy">© 2026 BanglaAI</div></div>', unsafe_allow_html=True)
