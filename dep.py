import streamlit as st

# import subprocess
# subprocess.call(["pip", "install", "wordcloud"])
# from wordcloud import WordCloud
# import nltk
# from nltk.corpus import stopwords

import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Save the TF-IDF Vectorizer as plain text
vectorizer = pickle.load(open("tf-idf.sav", "rb"))

# Save the Naive Bayes model as plain text
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

page_bg_img = """
<style>
[data-testid="stReportViewContainer"]{
background-image: linear-gradient( 2deg,  rgba(0,0,0,1) 9.2%, rgba(127,16,16,1) 103.9% );
}

[id="sentiment-analysis-app"]{
color:#faf3dd;
}

[id="unveiling-experiences-at-xyz-hotel"]{
color:#faf3dd;
font-weight: 900;
font-family: cursive;
}
</style>
"""


# Streamlit app code
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Sentiment Analysis App")
st.subheader("✨ Unveiling Experiences at XYZ-Hotel 🧧")
# st.set_option("deprecation.showPyplotGlobalUse", False)  # for wordcloud


# # wordcloud function
# def generate_wordcloud(user_input):
#     wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_input)
#     st.image(wordcloud.to_image(), caption="Word Cloud", use_column_width=True)


# User input text area
user_input = st.text_area("write a review to us :")

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

# if user_input:
#     generate_wordcloud(user_input)
