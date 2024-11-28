
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import time

# Load the pre-trained Random Forest model
with open("Models/random_forest.pkl", "rb") as model_file:
    model_rf = pickle.load(model_file)

# Load the CountVectorizer used during training
with open("Models/countVectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to predict sentiment
def predict_sentiment(text):
    # Transform the input text using the loaded CountVectorizer
    text_vectorized = vectorizer.transform([text])
    # Make prediction using the Random Forest model
    prediction = model_rf.predict(text_vectorized)
    return prediction[0]


# Set up the Streamlit app
st.set_page_config(page_title="Sentiment Analysis App", page_icon=":smiley:", layout="wide")
st.title("Sentiment Analysis App")
st.markdown("<h3 style='text-align: center;'>Analyze your reviews to see if they are positive or negative!</h3>", unsafe_allow_html=True)

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.write("1. Enter a review text in the text area below.")
st.sidebar.write("2. Click the 'Analyze' button to get the sentiment.")
st.sidebar.write("3. Enjoy the insights!")

# Text input for user review
user_input = st.text_area("Review Text", height=150, placeholder="Type your review here...")

# Button to submit the input
if st.button("Analyze"):
    with st.spinner("Analyzing..."):  # Show a spinner while processing
        time.sleep(1)  # Simulate processing time (optional)
        if user_input:
            sentiment = predict_sentiment(user_input)
            # Display the result
            if sentiment == 1:
                st.success("Sentiment: **Positive**", icon="✅")
            else:
                st.error("Sentiment: **Negative**", icon="❌")
        else:
            st.warning("Please enter a review text to analyze.")

# Add footer
st.markdown("<footer style='text-align: center;'>Developed with ❤️ by NAVEEN</footer>", unsafe_allow_html=True)