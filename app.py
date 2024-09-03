import streamlit as st
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import string

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Load the model and frequency dictionary
with open("logistic_regression_model.pkl", "rb") as file:
    theta = pickle.load(file)

with open("freqs.pkl", "rb") as file:
    freqs = pickle.load(file)

# Define the preprocessing function
def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1.0), 0)
        x[0, 2] += freqs.get((word, 0.0), 0)
    return x

def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def pre(sentence):
    yhat = predict_tweet(sentence, freqs, theta)
    if yhat > 0.5:
        return 'Positive sentiment'
    else:
        return 'Negative sentiment'

# Streamlit app
st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment:")

user_input = st.text_area("Tweet Input", "Type your tweet here...")

if st.button("Predict Sentiment"):
    if user_input.strip():  # Check if input is not empty
        result = pre(user_input)
        st.write(f"The sentiment of the tweet is: **{result}**")
    else:
        st.write("Please enter a valid tweet.")
