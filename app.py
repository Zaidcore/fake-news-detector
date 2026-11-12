
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

MODEL_PATH = "fake_news_pipeline.pkl"

st.title("📰 Fake News Detection System")
st.write("Machine Learning Based News Classification")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

if st.button("Analyze News", type="primary"):

    if len(news.strip()) < 30:
        st.warning("Enter at least 30 characters.")

    else:
        prediction = model.predict([news])[0]
        probabilities = model.predict_proba([news])[0]

        prob = dict(zip(model.classes_, probabilities))

        if prediction == 0:
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.metric(
            "Prediction Confidence",
            f"{prob[prediction]*100:.2f}%"
        )

        st.write("Fake probability:",
                 f"{prob[0]*100:.2f}%")

        st.write("Real probability:",
                 f"{prob[1]*100:.2f}%")
