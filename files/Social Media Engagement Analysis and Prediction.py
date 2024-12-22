#!/usr/bin/env python
# coding: utf-8

# #### Appendix 1 - DSC630-T301
# #### Mallory Holland, Cesar Jaidar, Ryan Krenke

# In[2]:


# Load the data XLSX file, specifying sheet name
file_path = '/Users/Malloryh5/Downloads/archive-2/social_media_engagement_data.xlsx'
social_data = pd.read_excel(file_path, sheet_name='Working File')


# Our Team has concluded that the Post ID column is a unique identifier value for each social media post which will not be any use in our analysis. However, because it is a unique value, we recognize that we can use it to check for duplicated data.

# In[3]:


#creating duplicateRows df calling the duplicated function on the Post ID column in the dataset
duplicateRows = social_data[social_data.duplicated(['Post ID'])]


# In[4]:


duplicateRows.shape#checking the size of the duplicated df


# This duplicated dataframe shows zero(0) duplicated data. Our Team has agreed to then outright remove this Post ID column and  three others columns in our dataset which also have no value in our analysis.

# In[5]:


#using drop to remove identified columns
social_data2 = social_data.drop(['Post ID', 'Post Timestamp', 'Audience Interests', 'Campaign ID'], axis=1)


# We have identified five categorical data columns to be converted to dummy variables. 

# In[6]:


#creating new df using pandas get_dummies on previous df 
#listing all the column prefixes and names of desired dummies
social_data3 = pd.get_dummies(social_data2,prefix=['Platform', 'Post Type', 'Age Group', 'Audience Gender', 'Audience Continent'], columns=['Platform', 'Post Type', 'Age Group', 'Audience Gender', 'Audience Continent'], drop_first=True)


# The next feature to engineer is the Time Period. It appears to be a categorical value based on the time of day. Our Team has determined to encode this column as an ordinal value due to its ordinal classification. 

# In[7]:


#creating oe variable to run OrdinalEncoder function
oe = OrdinalEncoder(categories=[['Morning', 'Afternoon', 'Evening', 'Night']])


# In[8]:


#creating array to fit/transform the ordinal encoder function on the Time Period,#reshape column
time_encoded_data = oe.fit_transform(social_data3['Time Periods'].array.reshape(-1, 1)) 


# In[9]:


#adding new column to df using list
social_data3['Time Period Encoded'] = [x[0] for x in time_encoded_data]


# In[10]:


#using drop to remove original Time Periods column
social_data4 = social_data3.drop(['Time Periods'], axis=1)


# Our Team would like to create new columns to specify the day of the week and quarter of the year. 
# 
# Using dt.weekday results in values of 0=Monday 1=Tuesday 2=Wednesday 3=Thursday 4=Friday 5=Saturday 6=Sunday.
# 
# Using dt.quarter results in values of Jan-Mar=1 Apr-June=2 Jul-Sep=3 Oct-Dec=4

# In[11]:


#creating new column Day_of_Week using dt.weekday function on Date column
social_data4['Day_of_Week'] = social_data4['Date'].dt.weekday


# In[12]:


#creating new column Quarter using dt.quarter function on Date column
social_data4['Quarter'] = social_data4['Date'].dt.quarter


# From the numeric attributes like "Comments" and "Likes", our Team would like to create some additional binned range and ratio features for further analysis. 
# 
# First we need to understand the natural quartiles of these attributes to be able to split them in thirds. 

# In[13]:


#using describe() to summarize comments column
comments_summary = social_data4['Comments'].describe()
comments_summary
#187 313


# In[14]:


#using describe() to summarize likes column
likes_summary = social_data4['Likes'].describe()
likes_summary
#375 625


# In[15]:


#using describe() to summarize shares column
shares_summary = social_data4['Shares'].describe()
shares_summary
#75 125


# In[16]:


#assigning bin sizes and labels for each parameter #used describe() to find the bins
comments_bins = [0, 187, 313, 500]
comments_labels = ['Low','Medium','High']
likes_bins = [0, 375, 625, 1000]
likes_labels = ['Low','Medium','High']
shares_bins = [0, 75, 125, 200]
shares_labels = ['Low','Medium','High']


# In[17]:


#adding range column using pd.cut() on specified column with designated bins and labels
social_data4['Comments_Range'] = pd.cut(social_data4['Comments'], bins=comments_bins, labels=comments_labels)


# In[18]:


#adding range column using pd.cut() on specified column with designated bins and labels
social_data4['Likes_Range'] = pd.cut(social_data4['Likes'], bins=likes_bins, labels=likes_labels)


# In[19]:


#adding range column using pd.cut() on specified column with designated bins and labels
social_data4['Shares_Range'] = pd.cut(social_data4['Shares'], bins=shares_bins, labels=shares_labels)


# In[20]:


#creating copy of current df
social_data5 = social_data4.copy()


# In[21]:


#creating new columns in df by dividing one column by another
social_data5['Likes/Shares'] = social_data5['Likes'] / social_data5['Shares']
social_data5['Shares/Comments'] = social_data5['Shares'] / social_data5['Comments']
social_data5['Comments/Reach'] = social_data5['Comments'] / social_data5['Reach']


# Audience Age

# In[22]:


#using describe() to summarize shares column
age_summary = social_data5['Audience Age'].describe()
age_summary


# Our Team originally planned to group Audience Age into categories of teenagers, adults, and seniors. However, after realizing the ages only range from 18-65, this does not seem necessary. We can use the Age Group parameter already defined. 

# For future ease and consistency, we would like to update the column names to all lowercase and replace the spaces with underscores. 

# In[23]:


#creating copy of current df
social_data6 = social_data5.copy()


# In[24]:


#using map function on df columns to assign str.lower on columns to all lowercase results
social_data6.columns = map(str.lower, social_data6.columns)


# In[25]:


#using replace function on df columns to replace string blank spaces with underscores
social_data6.columns = social_data6.columns.str.replace(' ', '_')


# In[26]:


#creating copy of current df
social_data7 = social_data6.copy()


# In[27]:


# Function to set value to 0 if the influencer_id is null or blank, otherwise 1
social_data7['influencer_id_processed'] = social_data7['influencer_id'].apply(lambda x: 0 if pd.isnull(x) or str(x).strip() == '' else 1)


# In[28]:


# Make a month columns
social_data7['month'] = social_data7['date'].dt.month


# ### Visualizations

# In[29]:


# Number of Posts per Platform
plt.figure(figsize=(5,3))# Figure size
count_platform = sns.countplot(x='Platform', data=social_data2) # Count plot
count_platform.bar_label(count_platform.containers[0])# Place the count above the column
plt.title('Number of Posts per Platform')# Title
plt.xlabel('Platform')# X-axis label
plt.ylabel('Post Count')# Y-axis label
plt.xticks(rotation=45)# rotate x-axis labels
plt.show()# Print plot


# In[30]:


# How Many Posts per Type
plt.figure(figsize=(5,3))# Figure Size
count_type = sns.countplot(x='Post Type', data=social_data2)# Count Plot (Post Type)
count_type.bar_label(count_type.containers[0])# Place count above the post type
plt.title('Number of Posts per Type')# Title
plt.xlabel('Post Type')# X-axis Label
plt.ylabel('Post Count')# Y-axis Label
plt.xticks(rotation=45)# Rotate x-axis
plt.show()# Print graph


# In[31]:


#  Boxplot Engagement by Platform
plt.figure(figsize=(5,3))# Figure size
sns.boxplot(x='Platform', y='Engagement Rate', data=social_data2)# boxplot of engagement rate by platform
plt.title('Engagement Rate by Platform')# Title
plt.xlabel('Platform')# X-axis label
plt.ylabel('Engagement Rate')# Y-axis label
plt.xticks(rotation=45)# Rotate x-labels
plt.show()# Print graph


# In[32]:


# Boxplot Engagement per Post Type
plt.figure(figsize=(5,3))# Figure size
sns.boxplot(x='Post Type', y='Engagement Rate', data=social_data2)# boxplot of engagement rate by platform
plt.title('Engagement Rate by Post Type')# Title
plt.xlabel('Platform')# X-axis label
plt.ylabel('Engagement Rate')# Y-axis label
plt.xticks(rotation=45)# Rotate x-labels
plt.show()# Print graph


# In[33]:


# Bar graph Likes per Quarter Period
plt.figure(figsize=(5,3))# Figure size
sns.barplot(x='quarter', y='likes', data=social_data7, estimator=sum)# Likes per quarter
plt.title('Total Likes per Quarter')# Title
plt.xlabel('Quarter')# X-axis label
plt.ylabel('Total Likes')# Y-axis label
plt.show()# Show graph


# In[34]:


# Bar Graph Likes per Month Period
plt.figure(figsize=(5,3)) # Plot size
sns.barplot(x='month', y='likes', data=social_data7, estimator=sum)# Bar plot of likes per month
plt.title('Total Likes per Month')# Title
plt.xlabel('Month')# X-axis label
plt.ylabel('Total Likes')# Y-axis label
plt.show()# Show graph


# In[35]:


# Bar Graph Likes per Day of the week Period
plt.figure(figsize=(5,3))
sns.barplot(x='day_of_week', y='likes', data=social_data7, estimator=sum)# Bar plot of likes per day of week
plt.title('Total Likes per Day of the Week')# Title
plt.xlabel('Day of the Week')# X-axis label
plt.ylabel('Total Likes')# Y-axis label
plt.show()# Show graph


# In[36]:


# Likes per time period (Morning, Afternoon, etc.)
plt.figure(figsize=(5,3))# Figure size
sns.barplot(x='Time Period Encoded', y='Likes', data=social_data3, estimator=sum)# Boxplot of time period
plt.title('Total Likes per Time Period')# Title
plt.xlabel('Time Period')# X-axis label
plt.ylabel('Total Likes')# Y-axis label
plt.show()#Show graph


# In[37]:


# Count plot gender vs. platform
plt.figure(figsize=(5,3))# Figure size
sns.countplot(x='Platform', hue='Audience Gender', data=social_data2)#Count plot
plt.title('Audience Gender per Platform')#Title
plt.xlabel('Platform')# X-axis label
plt.ylabel('Count')# Y-axis label
plt.legend(title='Gender')# Key
plt.xticks(rotation=45)# Rotate X-axis
plt.show()# Show results


# In[38]:


# Count plot gender vs. platform
plt.figure(figsize=(5,3))# Figure size
sns.countplot(x='Platform', hue='Age Group', data=social_data2)#Count plot
plt.title('Audience Gender per Platform')#Title
plt.xlabel('Platform')# X-axis label
plt.ylabel('Count')# Y-axis label
plt.legend(title='Gender')# Key
plt.xticks(rotation=45)# Rotate X-axis
plt.show()# Show results


# In[39]:


# Sentiment Count Plot
plt.figure(figsize=(5,3))#Figure size
sns.countplot(x='sentiment', data=social_data7)# Count plot
plt.title('Sentiment Distribution')# Title
plt.xlabel('Sentiment')# X-axis label
plt.ylabel('Count')# Y-axis label
plt.xticks(rotation=45)# Rotate label
plt.show()# Show graph


# In[40]:


# Sentiment vs Engagement Rate
plt.figure(figsize=(5,3))#Figure size
sns.boxplot(x='sentiment', y='engagement_rate', data=social_data7)# Boxplot: sentiment vs engagement rate
plt.title('Sentiment vs Engagement Rate')# Title
plt.xlabel('Sentiment')# X-axis label
plt.ylabel('Engagement Rate')# Y-axis label
plt.show()# Show graph


# In[41]:


# Find the mean engagement rate for each country
engagement_by_country = social_data7.groupby('audience_location')['engagement_rate'].mean().reset_index()


# In[42]:


# Remove warning request to download map to computer. So everyone does not have to download the map data
warnings.filterwarnings('ignore', category=FutureWarning)


# In[43]:


# Load world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# In[44]:


# Combine engagement data with world map based on country names
world = world.merge(engagement_by_country, how='left', left_on='name', right_on='audience_location')


# In[45]:


# Plot the world heatmap based on engagement rate
fig, ax = plt.subplots(1, 1, figsize=(9, 6))# Figure size and allow map outline
world.boundary.plot(ax=ax)# Print the outline
# Plot engagement rate
world.plot(column='engagement_rate', ax=ax, legend=True, cmap='YlOrRd', legend_kwds={'label': "Engagement Rate by Country",'orientation': "horizontal"})
plt.title('World Heatmap of Engagement Rate by Country')# Title
plt.show()# Show graph


# ### Run models

# In[46]:


social_data7['year'] = social_data7['time'].dt.year# And year to allow Datetime


# In[47]:


X_engagement = social_data7.drop(columns=['post_content', 'date', 'platform_instagram', 'platform_linkedin', 'platform_twitter', 'influencer_id', 'likes/shares', 'shares/comments', 'comments/reach', 'quarter','post_type_link', 'audience_gender_male', 'audience_gender_other', 'audience_location','post_type_video', 'audience_age', 'age_group_mature_adults', 'age_group_senior_adults', 'time', 'engagement_rate'])# X split for engagement rate prediction
Y_engagement = social_data7['engagement_rate']# Y split for engagement rate prediction


# In[48]:


X_sentiment = social_data7.drop(columns=['post_content', 'date', 'influencer_id', 'likes/shares', 'shares/comments', 'comments/reach', 'audience_gender_male', 'audience_gender_other', 'audience_location', 'time','sentiment','influencer_id_processed', 'year', 'engagement_rate', 'likes', 'comments', 'shares', 'reach', 'impressions', 'comments_range', 'likes_range', 'shares_range', 'audience_continent_antarctica','audience_continent_asia', 'audience_continent_europe','audience_continent_northamerica', 'audience_continent_oceania', 'audience_continent_southamerica', 'audience_age'])# X split for sentiment prediction
Y_sentiment = social_data7['sentiment']# Y split for sentiment prediction


# In[49]:


preprocessor = ColumnTransformer(#Transform Data# preprocessor transform columns
    transformers=[# Apply StandardScaler to numeric columns
        ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])), 
        ('cat', OneHotEncoder(handle_unknown='ignore'), # Apply OneHotEncoder to categorical columns
         make_column_selector(dtype_include=['object', 'category']))],remainder='passthrough')# Pass anything else


# In[50]:


# Logistic Regression and Random Forest Models
models = {# Model
    'logistic_regression': {# Logisitic regression
        'model': Pipeline([
            ('preprocess', preprocessor),#Apply processor
            ('smote', SMOTE(random_state=42)),# Balance data
            ('model', LogisticRegression(max_iter=500))]),# Logistic regression
        'params': {# params for logistic regression
            'model__C': [0.01, 0.1, 1.0],'model__penalty': ['l2'],'model__solver': ['liblinear', 'saga'],
            'model__tol': [1e-3]}},
    'random_forest': {# Random forest
        'model': Pipeline([# Model Pipeline
            ('preprocess', preprocessor),# Apply processor 
            ('smote', SMOTE(random_state=42)),# Balance data
            ('model', RandomForestClassifier(random_state=42))]),# Model random forest classifier
        'params': {# params for classifier
            'model__n_estimators': [100, 200],'model__max_depth': [5, 10, 15],
            'model__min_samples_split': [5, 10],'model__min_samples_leaf': [5, 10],}}}


# In[51]:


# Test Train Split # Sentiment Train-Test Split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sentiment, Y_sentiment, test_size=0.2, 
                                                            random_state=42)


# In[52]:


# Try both models for sentiment rate
for name, info in models.items():# For loop to run all models
    print(f'Running Grid Search for {name}')
    random_search_m1 = RandomizedSearchCV(info['model'], param_distributions=info['params'], n_iter=6, cv=5, verbose=0, n_jobs=-1)# Run random search
    random_search_m1.fit(X_train_s, y_train_s)# Fit data


# In[54]:


y_pred_s = random_search_m1.predict(X_test_s)# Predict y for sentiment


# In[55]:


report_test = classification_report(y_test_s, y_pred_s, target_names=['Neutral', 'Positive', 'Negative', 'Mixed'])# Accuracy test (metrics)
print(report_test)# Print results


# In[57]:


y_train_pred_s = random_search_m1.predict(X_train_s)# Check results for y pred using train data


# In[58]:


report_train = classification_report(y_train_s, y_train_pred_s)# Make classification report for train data
print(report_train)# Print results


# In[64]:


preprocessor = ColumnTransformer(# use preprocessor to transform columns
    transformers=[#Transform Data# replace na with mean
        ('imputer', SimpleImputer(strategy='mean'), make_column_selector(dtype_include=['int64', 'float64'])),
        ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])), # Scale num col
        ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(# Scale cat col
            dtype_include=['object', 'category', 'bool']))],
    remainder='passthrough')# Pass everything else


# In[76]:


models_3 = {
    'random_forest': {#Random Forest Models
        'model': Pipeline([# Pipeline for random forest
            # Use preprocessor first
            ('preprocess', preprocessor),
            # Model regressor
            ('model', RandomForestRegressor(random_state=42))]),
        'params': {# Params for random forest
            'model__n_estimators': [100, 200],'model__max_depth': [5, 10, 15],
            'model__min_samples_split': [5, 10],'model__min_samples_leaf': [5, 10]}},
    'xgboost': {  # XGBoost
        'model': Pipeline([  # Pipeline for xgboost
            ('preprocess', preprocessor),  # Use preprocessor first
            ('model', XGBRegressor(objective='reg:squarederror', random_state=42))]),# XGBoost
        'params': {  # Params for xgboost
            'model__n_estimators': [10, 25, 50, 100], 'model__max_depth': [3, 6, 10],
            'model__learning_rate': [0.01, 0.1, 0.2],'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0]}}}


# In[77]:


social_data9 = social_data7# Copy data


# In[78]:


social_data9 = social_data9.dropna(subset=['shares/comments','likes/shares','comments/reach'])# Drop Nans


# In[79]:


# Drop likes, shares, reach and comments because they are used to make engagement rate and ID
social_data9 = social_data9.drop(['likes', 'shares', 'reach', 'comments', 'influencer_id'], axis=1)


# In[80]:


social_data9_3 = social_data9[social_data9['audience_location'] == 'Italy']# Filter out all data, but Italy


# In[84]:


# For loop to turn all columns into numbers with only Italy data
for col in social_data9_3.select_dtypes(include=['object', 'category', 'datetime64[ns]', 'bool']).columns:
    social_data9_3.loc[:, col] = pd.factorize(social_data9_3[col])[0]


# In[85]:


correlation_m3_italy = round(social_data9_3.corr().loc[:,['engagement_rate', 'sentiment']], 2)# Correlation Mat


# In[86]:


plt.figure(figsize=(3, 5))# Plot Size# Heatmap for Italy data
sns.heatmap(correlation_m3_italy, annot=True, fmt='g', cmap='cividis')# Heatmap of data by location
plt.title('Italy Correlation Matrix of Engagement Rate and Sentiment')# Title


# In[87]:


highest_engagement_rate = social_data9.groupby('audience_location')['engagement_rate'].mean()#engagement mean
highest_engagement_rate.sort_values(ascending=False).head(10)# Sort values and show top 10


# In[88]:


social_data9_6 = social_data9[social_data9['audience_location'] == 'Italy']# Filter out all data, but Italy


# In[103]:


# X split for engagement rate prediction Italy
X_engagement_italy = social_data9_6[['platform_instagram','impressions', 'day_of_week','platform_linkedin', 'platform_twitter', 'post_type_link', 'post_type_video', 'age_group_mature_adults', 'age_group_senior_adults','audience_gender_male', 'audience_gender_other', 'quarter', 'comments_range', 'likes_range', 'shares_range', 'month']].dropna()
# Y split for engagement rate prediction Italy
Y_engagement_italy = social_data9_6['engagement_rate'].dropna()


# In[109]:


# Fix error
if np.isinf(social_data9_3.select_dtypes(include=[np.number])).values.sum() > 0:
    social_data9_3 = social_data9_3.replace([np.inf, -np.inf], mean())


# In[110]:


X_train_e_italy, X_test_e_italy, y_train_e_italy, y_test_e_italy = train_test_split(X_engagement_italy, Y_engagement_italy, test_size=0.2, random_state=42)# Test Train Split Italy


# In[111]:


for name, info in models_3.items():# for loop to find the best model for italy
    print(f'Running Grid Search for {name} (Italy)')# Print each time model is ran
    grid_search_italy = GridSearchCV(info['model'], param_grid=info['params'], cv=5, verbose=0, n_jobs=-1)# search for the best model and save
    grid_search_italy.fit(X_train_e_italy, y_train_e_italy)# Fit the model


# In[112]:


best_model_italy = grid_search_italy.best_estimator_# Save the best model


# In[113]:


y_pred_e_italy = best_model_italy.predict(X_test_e_italy)# Predict y for engagement for italy


# In[114]:


mse_e_italy = mean_squared_error(y_test_e_italy, y_pred_e_italy)# Italy MSE
print(f"Mean Squared Error (MSE) Italy: {mse_e_italy:.3f}")
mas_e_italy = mean_absolute_error(y_test_e_italy, y_pred_e_italy)# Italy MAS
print(f"Mean Absolute Error (MAE) Italy: {mas_e_italy:.3f}")
rmse_e_italy = np.sqrt(mse_e_italy)# Italy RMSE
print(f"Root Mean Squared Error (RMSE) Italy: {rmse_e_italy:.3f}")
r2_e_italy = r2_score(y_test_e_italy, y_pred_e_italy)# Italy R^2
print(f"R^2 Score Italy: {r2_e_italy:.3f}")


# In[115]:


cv_e_italy = cross_validate(best_model_italy, X_engagement_italy, Y_engagement_italy, return_train_score=True, cv=5, n_jobs=-1)# Check for overfitting or underfitting for Italy


# In[116]:


train_scores_e_italy = np.round(cv_e_italy['train_score'],3)#scores for Italy(train)
test_scores_e_italy = np.round(cv_e_italy['test_score'], 3)# scores for Italy# Test score


# In[117]:


train_mean_e_italy = np.mean(train_scores_e_italy)# Cal. mean Italy train
train_std_e_italy = np.std(train_scores_e_italy)#Cal. std Italy train
test_mean_e_italy = np.mean(test_scores_e_italy)#Cal. mean Italy test
test_std_e_italy = np.std(test_scores_e_italy)#Cal. std Italy test


# In[118]:


print(f"Train Scores: {train_scores_e_italy}")#train data Italy
print(f"Test Scores: {test_scores_e_italy}")#test data Italy
print(f"Mean Train Score: {train_mean_e_italy:.3f}")#mean train Italy
print(f"Mean Test Score: {test_mean_e_italy:.3f}")#mean test Italy
print(f"Standard Deviation of Train Scores: {train_std_e_italy:.3f}")#std train Italy
print(f"Standard Deviation of Test Scores: {test_std_e_italy:.3f}")#std test Italy


# In[119]:


social_data9_5 = social_data9[social_data9['audience_location'] == 'Croatia']# Filter out all data, but Croatia


# In[120]:


# For loop to turn all columns into numbers with only Croatia data
for col in social_data9_5.select_dtypes(include=['object', 'category', 'datetime64[ns]', 'bool']).columns:
    social_data9_5.loc[:,col] = pd.factorize(social_data9_5[col])[0]# Turn cols into num


# In[121]:


correlation_m3_croatia = round(social_data9_5.corr().loc[:,['engagement_rate', 'sentiment']],2)#Correlation max


# In[122]:


# Heatmap for Croatia data
plt.figure(figsize=(3, 5))# Plot Size
sns.heatmap(correlation_m3_croatia, annot=True, fmt='g', cmap='PiYG')# Heatmap of data by location
plt.title('Croatia Correlation Matrix of Engagement Rate and Sentiment')# Title


# In[123]:


# Filter everything, but Croatia.
social_data9_5 = social_data9[social_data9['audience_location'] == 'Croatia']


# In[124]:


social_data9_5 = social_data9_5.dropna()# Drop NA


# In[125]:


if np.isinf(social_data9_5.select_dtypes(include=[np.number])).values.sum() > 0:
    social_data9_5 = social_data9_5.replace([np.inf, -np.inf], mean())# Fix error


# In[126]:


X_engagement_croatia = social_data9_5[['platform_instagram','impressions', 'day_of_week','platform_linkedin', 'platform_twitter','post_type_link', 'post_type_video', 'age_group_mature_adults', 'age_group_senior_adults','audience_gender_male', 'audience_gender_other', 'quarter', 'comments_range','likes_range', 'shares_range', 'month']].dropna()# X split for engagement
Y_engagement_croatia = social_data9_5['engagement_rate'].dropna()# Y split


# In[127]:


X_train_e_croatia, X_test_e_croatia, y_train_e_croatia, y_test_e_croatia = train_test_split(#Test-Train Croatia
    X_engagement_croatia, Y_engagement_croatia, test_size=0.2, random_state=42)


# In[128]:


for name, info in models_3.items():# For loop to find the best model
    print(f'Running Grid Search for {name} (Croatia)')#ran the model
    grid_search_croatia = GridSearchCV(info['model'], param_grid=info['params'], cv=5, verbose=0, n_jobs=-1)# Grid Search to find best model and save
    grid_search_croatia.fit(X_train_e_croatia, y_train_e_croatia)# Fit model


# In[129]:


best_model_croatia = grid_search_croatia.best_estimator_# Save best model


# In[130]:


y_pred_e_croatia=best_model_croatia.predict(X_test_e_croatia)#Predict croatia


# In[131]:


mse_e_croatia = mean_squared_error(y_test_e_croatia, y_pred_e_croatia)#Croatia MSE
mas_e_croatia = mean_absolute_error(y_test_e_croatia, y_pred_e_croatia)#Croatia MAE
rmse_e_croatia = np.sqrt(mse_e_croatia)#Croatia RMSE
r2_e_croatia = r2_score(y_test_e_croatia, y_pred_e_croatia)#Croatia R^2


# In[132]:


print(f"Mean Absolute Error Croatia (MAE): {mas_e_croatia:.3f}")# Print Croatia MAE
print(f"Mean Squared Error Croatia (MSE): {mse_e_croatia:.3f}")# Print Croatia MSE
print(f"Root Mean Squared Error (RMSE): {rmse_e_croatia:.3f}")# Print Croatia RMSE
print(f"R^2 Score Croatia: {r2_e_croatia:.3f}")# Print Croatia R^2


# In[133]:


cv_e_croatia = cross_validate(best_model_croatia, X_engagement_croatia,Y_engagement_croatia, return_train_score=True, cv=5, n_jobs=-1)#Check for overfitting or underfitting


# In[134]:


train_scores_e_croatia = np.round(cv_e_croatia['train_score'],3)#round scores Train
test_scores_e_croatia = np.round(cv_e_croatia['test_score'],3)#round scores Test


# In[135]:


train_mean_e_croatia = np.mean(train_scores_e_croatia)# Cal. mean Croatia train
train_std_e_croatia = np.std(train_scores_e_croatia)# Cal. std Croatia train
test_mean_e_croatia = np.mean(test_scores_e_croatia)# Cal. mean Croatia Test
test_std_e_croatia = np.std(test_scores_e_croatia)# Cal. std Croatia Test


# In[136]:


print('***Croatia***')
print(f"Train Scores: {train_scores_e_croatia}")#train scores
print(f"Test Scores: {test_scores_e_croatia}")#test scores
print(f"Mean Train Score: {train_mean_e_croatia:.3f}")#mean train
print(f"Mean Test Score: {test_mean_e_croatia:.3f}")#mean test
print(f"Standard Deviation of Train Scores: {train_std_e_croatia:.3f}")#std train
print(f"Standard Deviation of Test Scores: {test_std_e_croatia:.3f}")#std test

