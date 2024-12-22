#!/usr/bin/env python
# coding: utf-8
Author: Cesar Jaidar
# # Hyperparameter Tuning

# ## 1. Import the dataset and ensure that it loaded properly.

# In[6]:


import pandas as pd

# Load the dataset
data = pd.read_csv('Loan_Train.csv')

# Display the first few rows of the dataframe to confirm it's loaded correctly
data.head()


# ## 2. Prepare the data for modeling by performing the following steps:  Drop the column “Load_ID.”, Drop any rows with missing data, Convert the categorical features into dummy variables.

# In[7]:


# Drop the 'Loan_ID' column and any rows with missing data
data = data.drop('Loan_ID', axis=1)
data = data.dropna()

# Convert categorical features into dummy variables
data = pd.get_dummies(data, drop_first=True)

# Display the first few rows to verify the transformations
data.head()


# ## 3. Split the data into a training and test set, where the “Loan_Status” column is the target.

# In[8]:


from sklearn.model_selection import train_test_split

# Split the data into features and target variable
X = data.drop('Loan_Status_Y', axis=1)
y = data['Loan_Status_Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm the split
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## 4. Create a pipeline with a min-max scaler and a KNN classifier (see section 15.3 in the Machine Learning with Python Cookbook).

# In[9]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create a pipeline with MinMaxScaler and KNN classifier
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
])


# ## 5. Fit a default KNN classifier to the data with this pipeline. Report the model accuracy on the test set. Note: Fitting a pipeline model works just like fitting a regular model.

# In[10]:


# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Evaluate the model accuracy on the test set
accuracy = pipeline.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')


# ## 6. Create a search space for your KNN classifier where your “n_neighbors” parameter varies from 1 to 10. (see section 15.3 in the Machine Learning with Python Cookbook).

# In[11]:


from sklearn.model_selection import GridSearchCV

# Define the search space for n_neighbors parameter
param_grid = {'knn__n_neighbors': list(range(1, 11))}


# ## 7. Fit a grid search with your pipeline, search space, and 5-fold cross-validation to find the best value for the “n_neighbors” parameter.

# In[12]:


# Fit a grid search with the pipeline and the search space
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Find the best parameter
print('Best n_neighbors:', grid_search.best_params_)


# ## 8. Find the accuracy of the grid search best model on the test set. Note: It is possible that this will not be an improvement over the default model, but likely it will be.

# In[13]:


# Evaluate the best model from grid search
best_model_accuracy = grid_search.score(X_test, y_test)
print(f'Accuracy of the best model: {best_model_accuracy}')


# ## 9. Now, repeat steps 6 and 7 with the same pipeline, but expand your search space to include logistic regression and random forest models with the hyperparameter values in section 12.3 of the Machine Learning with Python Cookbook. 

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Define a new pipeline that can switch between models
pipeline_switch = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', KNeighborsClassifier())
])

# Define the search space for different classifiers
param_grid_switch = [
    {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors': list(range(1, 11))},
    {'classifier': [LogisticRegression(max_iter=1000)], 'classifier__C': [0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier()], 'classifier__n_estimators': [10, 100, 1000], 'classifier__max_features': ['sqrt', 'log2']}
]

# Fit a grid search with the new pipeline and search space
grid_search_switch = GridSearchCV(pipeline_switch, param_grid_switch, cv=5, verbose=3)
grid_search_switch.fit(X_train, y_train)


# ## 10. What are the best model and hyperparameters found in the grid search? Find the accuracy of this model on the test set.

# In[16]:


# Find the best model and parameters
best_params_switch = grid_search_switch.best_params_
best_model_switch_accuracy = grid_search_switch.score(X_test, y_test)
print('Best Parameters:', best_params_switch)
print('Best Model Accuracy:', best_model_switch_accuracy)


# ## 11. Summarize your results.
In this analysis, three predictive models were evaluated using a dataset to determine loan approval outcomes. The initial model, a K-Nearest Neighbors (KNN) classifier with default settings, demonstrated a reasonable accuracy of 78.13%. Subsequent optimization of the KNN model, specifically by tuning the n_neighbors parameter to 3, marginally improved the accuracy to 79.17%.

Further exploration involved expanding the modeling approach to include Logistic Regression and Random Forest classifiers, with an extensive hyperparameter tuning conducted via grid search. Among the models considered, the Logistic Regression with a regularization strength C of 10 emerged as the most effective, achieving an accuracy of 82.29% on the test set.
