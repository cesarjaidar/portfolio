#!/usr/bin/env python
# coding: utf-8

# # DSC550 Week 3 - Sentiment Analysis
Author: Cesar Jaidar
# # Part 1: Using the TextBlob Sentiment Analyzer

# ## Import the movie review data as a data frame and ensure that the data is loaded properly.

# In[39]:


import pandas as pd
from textblob import TextBlob

# Load the TSV file
data = pd.read_csv('labeledTrainData.tsv', delimiter='\t', encoding='utf-8')

# Display the first few rows of the DataFrame to ensure it's loaded properly
print(data.head())


# ## How many of each positive and negative reviews are there?

# In[40]:


# Count the number of positive and negative reviews
sentiment_counts = data['sentiment'].value_counts()

# Print the counts
print(sentiment_counts)


# ## Use TextBlob to classify each movie review as positive or negative. Assume that a polarity score greater than or equal to zero is a positive sentiment and less than 0 is a negative sentiment.

# In[41]:


# Function to classify sentiment based on polarity
def classify_review(review):
    analysis = TextBlob(review)
    # Return 1 if polarity >= 0 (positive), else 0 (negative)
    return 1 if analysis.sentiment.polarity >= 0 else 0

# Apply the function to each review in the DataFrame
data['predicted_sentiment'] = data['review'].apply(classify_review)

# Display the first few rows to verify the classifications
print(data[['review', 'predicted_sentiment']].head())


# ## Check the accuracy of this model. Is this model better than random guessing?

# In[42]:


# Calculate the accuracy
accuracy = (data['predicted_sentiment'] == data['sentiment']).mean()

# Check if the model is better than random guessing
random_guessing_accuracy = 0.50
is_better_than_random = accuracy > random_guessing_accuracy

# Print the results
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Is the model better than random guessing? {'Yes' if is_better_than_random else 'No'}")


# ## For up to five points extra credit, use another prebuilt text sentiment analyzer, e.g., VADER, and repeat steps (3) and (4).

# In[43]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the TSV file
data = pd.read_csv('labeledTrainData.tsv', delimiter='\t', encoding='utf-8')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment based on polarity
def classify_review(review):
    scores = sia.polarity_scores(review)
    # Return 1 if compound score >= 0 (positive), else 0 (negative)
    return 1 if scores['compound'] >= 0 else 0

# Apply the function to each review in the DataFrame
data['predicted_sentiment'] = data['review'].apply(classify_review)

# Calculate the accuracy
accuracy = (data['predicted_sentiment'] == data['sentiment']).mean()

# Check if the model is better than random guessing
random_guessing_accuracy = 0.50
is_better_than_random = accuracy > random_guessing_accuracy

# Print the results
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Is the model better than random guessing? {'Yes' if is_better_than_random else 'No'}")


# # Part 2: Prepping Text for a Custom Model

# ## Convert all text to lowercase letters.

# In[44]:


import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the TSV file
data = pd.read_csv('labeledTrainData.tsv', delimiter='\t', encoding='utf-8')

# Convert all text to lowercase
data['review'] = data['review'].str.lower()


# ## Remove punctuation and special characters from the text.

# In[45]:


# Remove punctuation and special characters
data['review'] = data['review'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))


# ## Remove stop words.

# In[46]:


# Define stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# ## Apply NLTK’s PorterStemmer.

# In[47]:


# Initialize PorterStemmer and apply stemming
stemmer = PorterStemmer()
data['review'] = data['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Display the first few rows to verify the preprocessing
print(data[['review']].head())


# ## Create a bag-of-words matrix from your stemmed text (output from (4)), where each row is a word-count vector for a single movie review (see sections 5.3 & 6.8 in the Machine Learning with Python Cookbook). Display the dimensions of your bag-of-words matrix. The number of rows in this matrix should be the same as the number of rows in your original data frame.

# In[48]:


from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the model and transform the data to a bag-of-words
bow_matrix = vectorizer.fit_transform(data['review'])

# Display the dimensions of the bag-of-words matrix
print("Shape of the bag-of-words matrix:", bow_matrix.shape)


# ## Create a term frequency-inverse document frequency (tf-idf) matrix from your stemmed text, for your movie reviews (see section 6.9 in the Machine Learning with Python Cookbook). Display the dimensions of your tf-idf matrix. These dimensions should be the same as your bag-of-words matrix.

# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the model and transform the data to a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data['review'])

# Display the dimensions of the TF-IDF matrix
print("Shape of the TF-IDF matrix:", tfidf_matrix.shape)

