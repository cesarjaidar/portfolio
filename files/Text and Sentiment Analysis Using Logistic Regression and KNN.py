#!/usr/bin/env python
# coding: utf-8

# # DSC550 Week 5 - Text/Sentiment Analysis, Categorical Data, and Dates/Times
Author: Cesar Jaidar
# # Prepping Text for a Custom Model

# ## Convert all text to lowercase letters.

# In[1]:


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

# In[2]:


# Remove punctuation and special characters
data['review'] = data['review'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))


# ## Remove stop words.

# In[3]:


# Define stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# ## Apply NLTK’s PorterStemmer.

# In[4]:


# Initialize PorterStemmer and apply stemming
stemmer = PorterStemmer()
data['review'] = data['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Display the first few rows to verify the preprocessing
print(data[['review']].head())


# ## Create a bag-of-words matrix from your stemmed text (output from (4)), where each row is a word-count vector for a single movie review (see sections 5.3 & 6.8 in the Machine Learning with Python Cookbook). Display the dimensions of your bag-of-words matrix. The number of rows in this matrix should be the same as the number of rows in your original data frame.

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the model and transform the data to a bag-of-words
bow_matrix = vectorizer.fit_transform(data['review'])

# Display the dimensions of the bag-of-words matrix
print("Shape of the bag-of-words matrix:", bow_matrix.shape)


# ## Create a term frequency-inverse document frequency (tf-idf) matrix from your stemmed text, for your movie reviews (see section 6.9 in the Machine Learning with Python Cookbook). Display the dimensions of your tf-idf matrix. These dimensions should be the same as your bag-of-words matrix.

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the model and transform the data to a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data['review'])

# Display the dimensions of the TF-IDF matrix
print("Shape of the TF-IDF matrix:", tfidf_matrix.shape)


# # 1. Split this into a training and test set.

# In[7]:


from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)


# # 2. Fit and apply the tf-idf vectorization to the training set.

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the model and transform the training data to a TF-IDF matrix
tfidf_train_matrix = tfidf_vectorizer.fit_transform(X_train)


# # 3. Apply but DO NOT FIT the tf-idf vectorization to the test set (Why?).

# In[9]:


# Transform the test data to a TF-IDF matrix using the already fitted vectorizer
tfidf_test_matrix = tfidf_vectorizer.transform(X_test)

Fitting the tf-idf vectorizer only on the training set and then using it to transform the test set ensures that the test set remains a true representation of new, unseen data. This practice prevents the inadvertent leakage of information from the test set during the model training process. It also maintains a consistent feature space across both training and testing phases. By applying the same transformation across both datasets, you ensure that the model is evaluated based on its ability to generalize from the training data to new scenarios, reflecting a realistic performance that could be expected in real-world applications. This method mirrors the challenges a model faces when deployed, as it will only encounter data that was not present during its development.
# # 4. Train a logistic regression using the training data.

# In[10]:


from sklearn.linear_model import LogisticRegression

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(tfidf_train_matrix, y_train)


# # 5. Find the model accuracy on the test set.

# In[11]:


# Calculate the accuracy of the model on the test set
accuracy = model.score(tfidf_test_matrix, y_test)
print("Accuracy:", accuracy)


# # 6. Create a confusion matrix for the test set predictions.

# In[12]:


from sklearn.metrics import confusion_matrix

# Predict the labels for the test set
y_pred = model.predict(tfidf_test_matrix)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# # 7. Get the precision, recall, and F1-score for the test set predictions.

# In[13]:


from sklearn.metrics import classification_report

# Generate a report on precision, recall, and F1-score
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# # 8. Create a ROC curve for the test set.

# In[14]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, model.decision_function(tfidf_test_matrix))
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# # 9. Pick another classification model you learned about this week and repeat steps.

# ## Train a K-Nearest Neighbors Classifier using the training data

# In[15]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model with a specific number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model using the TF-IDF training matrix
knn_model.fit(tfidf_train_matrix, y_train)


# ## Find the model accuracy on the test set.

# In[16]:


# Calculate the accuracy of the KNN model on the test set
knn_accuracy = knn_model.score(tfidf_test_matrix, y_test)
print("Accuracy of KNN model:", knn_accuracy)


# ## Create a confusion matrix for the test set predictions.

# In[17]:


# Predict the labels for the test set using the KNN model
knn_y_pred = knn_model.predict(tfidf_test_matrix)

# Generate the confusion matrix for the KNN model
knn_conf_matrix = confusion_matrix(y_test, knn_y_pred)
print("Confusion Matrix for KNN model:\n", knn_conf_matrix)


# ## Get the precision, recall, and F1-score for the test set predictions.

# In[18]:


# Generate a report on precision, recall, and F1-score for the KNN model
knn_report = classification_report(y_test, knn_y_pred)
print("Classification Report for KNN model:\n", knn_report)


# ## Create a ROC curve for the test set.

# In[19]:


# Calculate the probabilities of predictions
knn_probs = knn_model.predict_proba(tfidf_test_matrix)[:, 1]

# Compute ROC curve and ROC area for the KNN model
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
knn_roc_auc = auc(knn_fpr, knn_tpr)

# Plot the ROC curve for the KNN model
plt.figure()
plt.plot(knn_fpr, knn_tpr, color='darkorange', lw=2, label='KNN ROC curve (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for KNN')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




