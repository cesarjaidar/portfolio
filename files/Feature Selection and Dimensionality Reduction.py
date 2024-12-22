#!/usr/bin/env python
# coding: utf-8
Author: Cesar Jaidar
# # Feature Selection and Dimensionality Reduction

# # Part 1: PCA and Variance Threshold in a Linear Regression

# ## 1. Import the housing data as a data frame and ensure that the data is loaded properly.

# In[126]:


import pandas as pd

# Step 1: Import the housing data
housing_data = pd.read_csv('train.csv')
housing_data.head()


# ## 2. Drop the "Id" column and any features that are missing more than 40% of their values.

# In[127]:


# Drop the "Id" column and any features that are missing more than 40% of their values.
housing_data = housing_data.drop('Id', axis=1)

# Calculate the threshold for missing values (more than 40% missing)
threshold = len(housing_data) * 0.4

# Drop columns with missing values above the threshold
housing_data = housing_data.dropna(thresh=threshold, axis=1)

housing_data.head()


# ## 3. For numerical columns, fill in any missing data with the median value.

# In[128]:


# Identify numerical columns
numerical_cols = housing_data.select_dtypes(include=['number']).columns

# Fill missing values in numerical columns with their median
housing_data[numerical_cols] = housing_data[numerical_cols].fillna(housing_data[numerical_cols].median())


# ## 4. For categorical columns, fill in any missing data with the most common value (mode).

# In[129]:


# Identify categorical columns
categorical_cols = housing_data.select_dtypes(include=['object']).columns

# Fill missing values in categorical columns with their mode
for col in categorical_cols:
    mode_value = housing_data[col].mode()[0]
    housing_data[col] = housing_data[col].fillna(mode_value)

housing_data.head()


# ## 5. Convert the categorical columns to dummy variables.

# In[130]:


# Convert the categorical columns to dummy variables.
housing_data = pd.get_dummies(housing_data, columns=categorical_cols, drop_first=True)


# ## 6. Split the data into a training and test set, where the SalePrice column is the target.

# In[131]:


from sklearn.model_selection import train_test_split

# Define features and target
X = housing_data.drop('SalePrice', axis=1)
y = housing_data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()


# ## 7. Run a linear regression and report the R2-value and RMSE on the test set.

# In[132]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.decomposition import PCA

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Fit the model
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Calculate R2 and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

(r2, rmse)


# ## 8. Fit and transform the training features with a PCA so that 90% of the variance is retained 

# In[133]:


from sklearn.preprocessing import StandardScaler

# Scale features before PCA 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data explicitly to numpy arrays to avoid any DataFrame issues
X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)

# Initialize PCA to retain 90% of the variance
pca = PCA(n_components=0.9)
X_train_pca = pca.fit_transform(X_train_scaled)  # PCA on scaled data
X_test_pca = pca.transform(X_test_scaled)

# Initialize the Linear Regression model
lr_model_pca = LinearRegression()
lr_model_pca.fit(X_train_pca, y_train)

# Predict on the PCA-transformed test set
y_pred_pca = lr_model_pca.predict(X_test_pca)

# Calculate R2 and RMSE for PCA-transformed data
r2_pca = r2_score(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))

print("R-squared (PCA):", r2_pca)
print("RMSE (PCA):", rmse_pca)

# Show the shape of the transformed data
X_train_pca.shape


# ## 9. How many features are in the PCA-transformed matrix?

# In[134]:


# Number of features in the PCA-transformed matrix
num_features_pca = X_train_pca.shape[1]
num_features_pca


# ## 10. Transform but DO NOT fit the test features with the same PCA.

# In[135]:


# Apply StandardScaler to test data
X_test_scaled = scaler.transform(X_test)

# Explicitly convert X_test_scaled to a NumPy array if it's not already
X_test_scaled = np.array(X_test_scaled)

# Apply PCA to the scaled test data
X_test_pca = pca.transform(X_test_scaled)


# ## 11. Repeat step 7 with your PCA transformed data.

# In[136]:


# Fit the linear regression model on the PCA transformed training data
lr_model_pca = LinearRegression()
lr_model_pca.fit(X_train_pca, y_train)

# Predict on the PCA transformed test set
y_pred_pca = lr_model_pca.predict(X_test_pca)

# Calculate R2 and RMSE for PCA transformed data
r2_pca = r2_score(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))

(r2_pca, rmse_pca)


# ## 12. Take your original training features (from step 6) and apply a min-max scaler to them

# In[137]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)


# ## 13. Find the min-max scaled features in your training set that have a variance above 0.1

# In[138]:


# Find the min-max scaled features
variances = X_train_scaled.var(axis=0)
high_variance_features = variances > 0.1
X_train_high_var = X_train_scaled[:, high_variance_features]

# Show the shape of high variance data
X_train_high_var.shape


# ## 14. Transform but DO NOT fit the test features with the same steps applied in steps 11 and 12.

# In[139]:


# Transform but DO NOT fit the test features
X_test_scaled = scaler.transform(X_test)
X_test_high_var = X_test_scaled[:, high_variance_features]


# ## 15. Repeat step 7 with the high variance data.

# In[140]:


# Fit the linear regression model on the high variance training data
lr_model_high_var = LinearRegression()
lr_model_high_var.fit(X_train_high_var, y_train)

# Predict on the high variance test set
y_pred_high_var = lr_model_high_var.predict(X_test_high_var)

# Calculate R2 and RMSE for high variance data
r2_high_var = r2_score(y_test, y_pred_high_var)
rmse_high_var = np.sqrt(mean_squared_error(y_test, y_pred_high_var))

(r2_high_var, rmse_high_var)


# ## Summarize your findings.
PCA

Principal Component Analysis (PCA) was applied to reduce the dimensionality of the dataset, aiming to simplify the model and potentially enhance performance by reducing overfitting. The regression model using this PCA-transformed data performed robustly, with an R2 value of 0.842. This high R2 value indicates that the model with PCA-transformed data can explain about 84.2% of the variability in the target variable (SalePrice), and the RMSE was 34,828.09. 

High Variance

Instead of dimensionality reduction through PCA, another approach was taken where features were scaled using Min-Max scaling, and then only those features with a variance greater than 0.1 were retained. This method aims to keep only the features that show significant variability, thereby presumably retaining more useful information. 44 features were identified as having high variance.
The linear regression model built with these high variance features showed an R2 of 0.655. This suggests that about 65.5% of the variance in the target variable is explained by the model and the RMSE was 51,393.43, which was higher than PCA. 
# # Part 2: Categorical Feature Selection

# ## 1. Import the data as a data frame and ensure it is loaded correctly.

# In[141]:


import pandas as pd

# Load the data
data = pd.read_csv('mushrooms.csv')
data.head()


# ## 2. Convert the categorical features (all of them) to dummy variables.

# In[142]:


# Convert categorical variables to dummy variables
data_dummies = pd.get_dummies(data)
data_dummies.head()


# ## 3. Split the data into a training and test set.

# In[143]:


from sklearn.model_selection import train_test_split

# Define X and y
X = data_dummies.drop(['class_e', 'class_p'], axis=1)  # Features
y = data_dummies['class_p']  # Target (class 'p')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## 4. Fit a decision tree classifier on the training set.

# In[144]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
dt_classifier.fit(X_train, y_train)


# ## 5. Report the accuracy and create a confusion matrix for the model prediction on the test set.

# In[145]:


from sklearn.metrics import accuracy_score, confusion_matrix

# Predict on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

accuracy, conf_matrix


# ## 6. Create a visualization of the decision tree.

# In[146]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, max_depth=3, fontsize=10)
plt.title('Decision Tree - 3 levels')
plt.show()


# ## 7. Use a χ2-statistic selector to pick the five best features for this data

# In[147]:


from sklearn.feature_selection import SelectKBest, chi2

# Initialize the SelectKBest with chi-square statistic
chi_selector = SelectKBest(chi2, k=5)

# Fit the selector to the training data
chi_selector.fit(X_train, y_train)


# ## 8. Which five features were selected in step 7?

# In[148]:


# Get the selected features
selected_features = X.columns[chi_selector.get_support()]

selected_features


# ## 9. Repeat steps 4 and 5 with the five best features selected in step 7

# In[149]:


# Subset the training and test data to the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Initialize a new Decision Tree Classifier
dt_classifier_selected = DecisionTreeClassifier(random_state=42)

# Fit the classifier on the training set using only selected features
dt_classifier_selected.fit(X_train_selected, y_train)

# Predict on the test set with selected features
y_pred_selected = dt_classifier_selected.predict(X_test_selected)

# Calculate accuracy with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)

# Generate confusion matrix with selected features
conf_matrix_selected = confusion_matrix(y_test, y_pred_selected)

accuracy_selected, conf_matrix_selected


# ## 10. Summarize your findings.
The decision tree model utilizing all available features demonstrated performance with a 100% accuracy rate on the test dataset. In contrast, when the model was simplified to include only the top five features identified by the X2-statistic selector, accuracy dipped to approximately 92.49%. This indicates that although the selected features are highly informative, they do not capture all the nuances necessary for perfect classification, underscoring the value of the additional attributes present in the dataset.
