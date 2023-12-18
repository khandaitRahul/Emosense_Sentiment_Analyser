import streamlit as st

# import joblib
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# import nltk
# from nltk.corpus import stopwords

# Save the TF-IDF Vectorizer as plain text
# vectorizer = joblib.load('tfidf_vectorizer.joblib')
vectorizer = pickle.load(open("tf-idf.sav", "rb"))

# Save the Naive Bayes model as plain text
# mnb_os = joblib.load('naive_bayes_model.joblib')
mnb_os = pickle.load(open("mnb_os.sav", "rb"))


# Preprocessing function with stop words removal
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub("\[.*?\]", "", text)
#     text = re.sub("https?://*", "", text)
#     text = re.sub("\www.*", "", text)
#     text = re.sub("\.com*", "", text)
#     text = re.sub("<.*?>+", "", text)
#     text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
#     text = re.sub("\n", "", text)
#     text = re.sub("\w*\d\w*", "", text)

#     # Remove stop words (customize this list based on your needs)
#     stop_words = set(stopwords.words("english"))
#     words = nltk.word_tokenize(text)
#     text = " ".join([word for word in words if word.lower() not in stop_words])

#     return text


# Streamlit app code
st.title("Sentiment Analysis App")

# User input text area
user_input = st.text_area("Enter your hotel review:")

if st.button("Predict your sentiments"):
    # Check if the user has entered a review
    if not user_input:
        st.warning("Please enter a review before predicting the sentiment.")
    else:
        # Preprocess the input text
        # preprocessed_input = preprocess_text(user_input)

        # Vectorize the input text using the loaded TF-IDF Vectorizer
        text_vectorized = vectorizer.transform([user_input])

        # Make predictions using the loaded Naive Bayes model
        sentiment = mnb_os.predict(text_vectorized)

        # Display sentiment along with emoticons
        if sentiment[0] == "positive":
            st.write("It's a positive comment!! 😃")
        elif sentiment[0] == "neutral":
            st.write("It's a neutral comment 😐")
        else:
            st.write("It's a negative comment 😔")
