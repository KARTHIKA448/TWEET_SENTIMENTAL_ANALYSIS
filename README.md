# Logistic Regression-Based Sentiment Analysis on Tweets
### Project Overview
This project involves building a logistic regression model from scratch to perform sentiment analysis on tweets. The goal is to classify tweets as having either a positive or negative sentiment. The project is implemented in a Jupyter Notebook, and the model is then deployed as a web app using Streamlit.

# Analysis in Jupyter Notebook
### Data Collection
We utilize the twitter_samples dataset from the NLTK library, which contains a collection of positive and negative tweets. This dataset is split into training and testing sets for model development and evaluation.

### Data Preprocessing
The data preprocessing involves cleaning the tweets by:
- Removing unwanted characters like URLs, Twitter handles, and special symbols.
- Tokenizing the tweets to break them down into individual words.
- Removing stopwords (common words like "the", "is", etc.) and punctuation.
- Stemming the words to reduce them to their root forms (e.g., "running" becomes "run").
- 
### Feature Engineering
The main feature used in this model is the frequency of each word in the dataset. A frequency dictionary is built that maps each word to its occurrence in positive and negative tweets. These frequencies are then used to extract features from new tweets for the model.

### Model Building
The logistic regression model is built from scratch, with the following key steps:
 - Sigmoid Function: This function is used to output probabilities for binary classification.
 - Cost Function and Gradient Descent: These are used to optimize the model by adjusting the weights to minimize the error between the predicted and actual labels.

### Model Training and Evaluation
The model is trained on a dataset of positive and negative tweets. After training, the model is evaluated on a test set, achieving an accuracy of 99.5%.

### Saving the Model
The trained model and the frequency dictionary are saved using Python's pickle module, allowing them to be reused for predictions without retraining.

# Streamlit Web App
The model is deployed as a web app using Streamlit. The app allows users to input a tweet and receive a prediction on whether the tweet is positive or negative.

### How the App Works
- Input: Users can type a tweet into the input box.
- Prediction: The app processes the tweet using the same preprocessing steps as in the notebook. It then extracts features from the tweet and uses the logistic regression model to make a prediction.
- Output: The app displays whether the tweet has a positive or negative sentiment.

# Running the App
To run the app locally:
- Clone the repository.
- Install the required packages using pip install -r requirements.txt.
- Run the app using streamlit run app.py.
  
The app will open in a web browser, ready for you to input tweets and see predictions.

# Conclusion
This project demonstrates how to build a logistic regression model from scratch for sentiment analysis on tweets and deploy it as an interactive web app using Streamlit.
