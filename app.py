import streamlit as st
import joblib

# Load trained model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 Spam Message Classifier")

st.write(
    "Enter an SMS message below and the machine learning model "
    "will classify it as Spam or Not Spam."
)

st.divider()

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

# Classification button
if st.button("🔍 Classify Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Convert text into TF-IDF features
        message_vector = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_vector)[0]

        # Probability
        probabilities = model.predict_proba(message_vector)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")

            confidence = probabilities[1] * 100
            st.metric("Spam Probability", f"{confidence:.2f}%")

        else:
            st.success("✅ NOT SPAM")

            confidence = probabilities[0] * 100
            st.metric("Ham Probability", f"{confidence:.2f}%")

st.divider()

st.caption("Machine Learning Model: Multinomial Naive Bayes | Features: TF-IDF")
st.caption("Model Accuracy: 97.04%")