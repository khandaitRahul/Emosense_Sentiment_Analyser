import streamlit as st
from streamlit_option_menu import option_menu

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# import nltk
# from nltk.corpus import stopwords

import pickle
import pandas as pd
import numpy as np
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
[data-testid="stApp"]{
background-image: linear-gradient( 58.2deg,  #C33764, rgba(171,53,163,0.45), #1D267197 );
}

[data-testid="StyledLinkIconContainer"]{
color:#faf3dd;
font-weight: 900;
font-family: cursive;
}

[data-testid="textInputRootElement"]{
box-sizing: border-box;
border-radius: 50px;
background-color: #f8f8f8;
font-size: 20px;
padding: 0px 20px;
}
</style>
"""


# Function to generate Word Cloud
def generate_wordcloud(user_input):
    if user_input and any(
        c.isalpha() for c in user_input
    ):  # Check if there are any alphabetical characters in the input
        wordcloud = WordCloud(width=800, height=200, background_color="white").generate(
            user_input
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("No words to generate a word cloud.")


def main():
    # __full screen__
    st.set_page_config(
        page_title=None,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("EmoSense App🎭")  # App name

    # __Navigation Bar__
    menu = option_menu(
        menu_title=None,
        options=["About us", "Review section"],
        icons=["house", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    # __Review-section-page__
    if menu == "Review section":
        st.divider()
        st.header("Navigating through sentiments with emotions🎎")

        # dividing into two columns using st.columns()
        col1, col2 = st.columns(2, gap="small")
        # __Left column__
        with col1:
            st.subheader("unveil your experience with XYZ-Hotel🧧")
            # User input text area
            user_input = st.text_area("write a review to us :")

            if st.button("Predict your sentiments"):
                # Check if the user has entered a review
                if not user_input:
                    st.warning(
                        "Please enter a review before predicting the sentiment.",
                        icon="⚠️",
                    )
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
                        st.balloons()
                    elif sentiment[0] == "neutral":
                        st.write("It's a neutral comment 😐")
                        st.balloons()

                    else:
                        st.write("It's a negative comment 😔")
                        st.snow()

        # __Right column__
        with col2:
            # Check if the review is not empty before generating Word Cloud
            if user_input.strip():
                generate_wordcloud(user_input)
            else:
                st.warning("No words to generate a word cloud.", icon="⚠️")
        st.divider()

    # __About-us-page__
    else:
        # __header__
        st.divider()
        st.header("Feel the Pulse : Decoding Emotions with EmoSense💌")

        # __content-area__
        col1, col2, col3 = st.columns([0.45, 0.20, 0.35], gap="small")
        # __left-col__
        with col1:
            para1 = """
                        EmoSense, an innovative sentiment analysis app, transcends the boundaries of conventional emotional understanding. 
                        Harnessing cutting-edge algorithms, the app sifts through textual content with precision, unraveling the complex spectrum of 
                        sentiments expressed. Its user-friendly interface ensures accessibility for all, making the exploration of emotional landscapes 
                        intuitive and insightful. Businesses can optimize strategies 
                        by tapping into customer sentiments, while individuals can enhance interpersonal connections.
                    """
            link1 = " <span style='font-size:25px;'>Ready for a memorable stay? [Book your hotel now](https://www.google.com/)</span>"
            st.markdown(para1)
            st.markdown(link1, unsafe_allow_html=True)

        # __middle-col__
        # logo
        with col2:
            logo = "https://png.pngtree.com/png-clipart/20230914/original/pngtree-cartoon-building-in-the-background-with-stars-around-it-vector-png-image_11089578.png"
            st.image(logo, width=250)

        # __Right-col__
        with col3:
            link2 = "<span style='font-size:25px;'>[Explore the features & benefits of this app.](https://medium.com/@pavansahu0809/predictive-modelling-for-hotel-ratings-a-data-science-approach-1d4af6431bb5 )</span>"
            para2 = """
                            EmoSense is more than an app: it's a transformative tool that empowers users to navigate the web of human emotions, 
                            fostering empathy and effective communication. With EmoSense stays ahead of linguistic shifts, 
                            ensuring that users receive accurate and relevant insights. In an era defined by digital connectivity, EmoSense emerges as a valuable 
                            companion, decoding emotions and bridging the gap in our increasingly nuanced communication landscape.
                        """
            st.markdown(link2, unsafe_allow_html=True)
            st.markdown(para2)
        st.divider()

        # __footer__
        # dividing into two columns using st.columns()
        col1, col2 = st.columns([3, 1], gap="small")
        # __Left column__
        with col1:
            # Feedback input box
            with st.form(key="feedback_form"):
                feedback_input = st.text_input("your feedback is valuable to us :")
                submit_button = st.form_submit_button(label="submit")

        # __Right column__
        with col2:
            # reply after submission
            if submit_button and feedback_input:
                st.success(
                    """thanxs for the feedbaack,
                        please visit our review section once.""",
                    icon="✅",
                )


if __name__ == "__main__":
    # st.set_option("deprecation.showPyplotGlobalUse", False)  # for wordcloud
    main()
