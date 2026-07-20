import streamlit as st
import joblib


st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="✉️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("✉️ Email Spam Classifier")

st.write("---")

user_input = st.text_area(
    "Paste an email or SMS message below:",
    height=150,
    placeholder="e.g., Congratulations! You've won a $1,000 Walmart gift card. Click here to claim..."
)

if st.button("Analyze Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message to analyze.")
    else:
        transformed_text = vectorizer.transform([user_input])
        
        prediction = model.predict(transformed_text)[0]

        st.write("### Analysis Result:")
        
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **HAM (SAFE) MESSAGE**")
