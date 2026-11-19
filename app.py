
import streamlit as st
import joblib

# Load the trained sentiment classification model
model = joblib.load("bangla_sentiment_final.joblib")

# Page configuration
st.set_page_config(
    page_title="BanglaAI",
    page_icon="🇧🇩",
    layout="centered"
)

# Main heading
st.title("🇧🇩 BanglaAI")
st.subheader("Bangla Text Sentiment Classification System")

st.write(
    "Enter a Bangla sentence below to predict "
    "whether its sentiment is Positive, Negative, or Neutral."
)

# Text input
user_text = st.text_area(
    "Enter Bangla Text:",
    placeholder="উদাহরণ: আজকের দিনটি খুব সুন্দর।",
    height=130
)

# Prediction button
if st.button("Predict Sentiment", type="primary"):

    if user_text.strip():
    # Predict sentiment
    prediction = model.predict([user_text])[0]

    # Convert numeric prediction into sentiment label
    sentiment_labels = {
        0: "Neutral",
        1: "Positive",
        2: "Negative"
    }

    sentiment = sentiment_labels[int(prediction)]


        # Get prediction probabilities
        probabilities = model.predict_proba([user_text])[0]

        # Class labels
        class_names = model.classes_

        # Find confidence
        confidence = max(probabilities) * 100

        # Display result
        st.success(f"Predicted Sentiment: {sentiment}")

        st.write(f"Prediction Confidence: {confidence:.2f}%")

        # Show probability for each class
        st.subheader("Class Probabilities")

        for label, probability in zip(
            class_names, probabilities
        ):
            st.write(f"{label}: {probability * 100:.2f}%")

    else:
        st.warning("Please enter some Bangla text first.")

# Footer
st.divider()
st.caption("BanglaAI | A Machine Learning Project")
