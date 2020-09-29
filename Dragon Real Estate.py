#!/usr/bin/env python
# coding: utf-8

# # Dragon Real Estate-Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing["CHAS"].value_counts()


# In[6]:


housing.describe()


# In[7]:

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#For plotting histogram
import matplotlib as plt
housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[9]:


# For larning purpose
   
import numpy as np

def split_train_test(data, test_ratio):
   np.random.seed(42)
   shuffled = np.random.permutation(len(data))
   print(shuffled)    
   test_set_size = int(len(data) * test_ratio)
   test_indices = shuffled[:test_set_size]
   train_indices = shuffled[test_set_size:]
   return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


#train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


#print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n" )


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n" )


# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["CHAS"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[15]:


strat_test_set.describe()


# In[16]:


strat_test_set["CHAS"].value_counts()


# In[17]:


strat_train_set["CHAS"].value_counts()


# In[18]:


#95/7                                          


# In[19]:


#376/28


# In[20]:


housing = strat_train_set.copy()       


# ## Looking for Correlation

# In[21]:


corr_matrix = housing.corr()


# In[22]:


corr_matrix["MEDV"].sort_values(ascending=False)


# In[23]:


from pandas.plotting import scatter_matrix


# In[24]:


attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[25]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying out Attribute Combinations

# In[26]:


housing["TAXRM"]=housing["TAX"]/housing["RM"]


# In[27]:


housing.head(5)


# In[28]:


housing["TAXRM"]


# In[29]:


corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# In[30]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[31]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels= strat_train_set["MEDV"].copy()


# ## Missing Attributes

# #### To take care of missing attributes, you have three option:
# 1. Get rid of the missing data points.
# 2. Get rid of the whole attribute.
# 3. Set the value to some value(0, mean or median)

# In[32]:


a = housing.dropna(subset=["RM"])          #Option 1  
a.shape


# In[33]:


housing.drop("RM", axis=1).shape                 #option 2  
#Note that there is no RM column and also note that original housing dataframe will remain unchanged


# In[34]:


median = housing["RM"].median()


# In[35]:


housing["RM"].fillna(median)     #Option 3
# Note that original housing dataframe will remain unchanged         


# In[36]:


housing.shape


# In[37]:


housing.describe()        #Before we started filling missing attribute    #See RM counts before imputing


# In[38]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[39]:


imputer.statistics_


# In[40]:


X = imputer.transform(housing)


# In[41]:


housing_tr=pd.DataFrame(X, columns=housing.columns)


# In[42]:


housing_tr.describe()                           #Now see RM counts after imputing


# ## Scikit-learn Design

# Primarily, three types of objects
# 1. Estimators - It estimate some parameter based on a dataset. Eg. Imputer.
# It has a fit method and transform method.
# Fit method - Fits the dataset and calculates internal parameters
# 
# 2. Transformers - This method takes input and return output based on learning from fit(). It also has a convinience function called fit_transform() which fits and then transform.
# 
# 3. Predicters - Linear Regresion model is a example of Predicter. fit() and predict() are two common functions.It also gives score() function which will evaluate the prediction.

# ## Feature Scaling
# 1. Min-max scaling (Normalization)
#    (value-min)/(max-min)
#     Sklearn provide a class called MinMaxScaler for this
#     
# 2. Standardization
#     (value - mean)/std
#     sklearn provide a class called StandarScaler for this

# ## Creating a Pipeline

# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    #   .... add as many as you want in your pipeline
    ("std_scaler", StandardScaler()),
    
])


# In[44]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[45]:


housing_num_tr.shape


# ## Selecting the desired model for Dragon Real Estate

# In[46]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor() 
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[47]:


some_data = housing.iloc[:5]


# In[48]:


some_labels = housing_labels.iloc[:5]


# In[49]:


prepared_data = my_pipeline.transform(some_data)


# In[50]:


model.predict(prepared_data)


# In[51]:


list(some_labels)


# ## Evaluating the model

# In[52]:


from sklearn.metrics import mean_squared_error
housing_predictions= model.predict(housing_num_tr)
mse= mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[53]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[54]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model , housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[55]:


rmse_scores


# In[56]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[57]:


print_scores(rmse_scores)


# ## Saving the model

# In[59]:


from joblib import dump, load
dump(model, "Dragon.joblib")


# ## Testing the model on test data

# In[66]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse= np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[67]:


final_rmse


# In[70]:


prepared_data[0]

