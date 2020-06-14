
# coding: utf-8

# Importing Libraries

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


# Importing Dataset

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                header = None, sep = "\s+")


# In[4]:


df.head()


# In[5]:


# Adding Column Headers

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", 
             "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


# Attribute Information:
# 
#     1. CRIM      per capita crime rate by town
#     2. ZN        proportion of residential land zoned for lots over 
#                  25,000 sq.ft.
#     3. INDUS     proportion of non-retail business acres per town
#     4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#                  river; 0 otherwise)
#     5. NOX       nitric oxides concentration (parts per 10 million)
#     6. RM        average number of rooms per dwelling
#     7. AGE       proportion of owner-occupied units built prior to 1940
#     8. DIS       weighted distances to five Boston employment centres
#     9. RAD       index of accessibility to radial highways
#     10. TAX      full-value property-tax rate per $10,000
#     11. PTRATIO  pupil-teacher ratio by town
#     12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#                  by town
#     13. LSTAT    % lower status of the population
#     14. MEDV     Median value of owner-occupied homes in $1000's

# In[6]:


df.head()


# In[7]:


# Visualizing Imp characteristics of dataset

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = "whitegrid", context = "notebook")



# In[78]:


# Plots

cols = ["CRIM", "LSTAT", "INDUS", "NOX", "RM", "MEDV"]

sns.pairplot(df[cols], size = 2.5)
plt.show()


# In[79]:


# Plotting Correlations

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale = 1.5)

hm = sns.heatmap(cm, 
                cbar = True,
                annot = True, 
                square = True,
                fmt = ".2f",
                annot_kws = {"size" : 15},
                xticklabels = cols,
                yticklabels = cols)
plt.show()




# In[80]:


# Seperating DF

x = df.iloc[ : , :-1].values

y = df.iloc[ : , 13].values


# In[81]:


# Splitting Dataset

from sklearn.cross_validation import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)


# In[82]:


# Fitting MLR model with sklearn

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)



# In[83]:


regressor.coef_


# In[84]:


regressor.intercept_


# In[85]:


# Predictions

y_pred = regressor.predict(x_test)


# In[86]:


# RMSE


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))


# In[87]:


rmse


# #######################################################################################################################
# #######################################################################################################################

# In[88]:


# Now we will make MLR model with statsmodel library

# Add extra variable with 1's

import statsmodels.api as sm

xs = sm.add_constant(x)


# In[89]:


# Splitting Dataset

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(xs,y, test_size = 0.3, random_state = 0)


# In[90]:


# Fit model

import statsmodels.formula.api as smf

regressor_ols = smf.OLS(endog = y_train_s, exog = x_train_s).fit()


# In[91]:


# Summary

regressor_ols.summary()


# In[92]:


# Predictions

y_pred1 = regressor_ols.predict(x_test_s)


# In[93]:


# RMSE


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse1 = sqrt(mean_squared_error(y_test_s, y_pred1))


# In[94]:


# RMSE

rmse1


# In[95]:


# RMSE from sklearn

rmse


# ###########################################################################################################################
# ###########################################################################################################################

# In[96]:


# Making optimal model using backward elimination

x_opt = x_train_s[ :, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]


# In[97]:


# Step 1

regressor_ols_b = smf.OLS(endog = y_train_s, exog = x_opt).fit()


# In[98]:


regressor_ols_b.summary()


# In[99]:


# Removing x3 or Indus since it has highest P value


x_opt = x_train_s[ :, [0,1,2,4,5,6,7,8,9,10,11,12,13]]

# Step 2

regressor_ols_b = smf.OLS(endog = y_train_s, exog = x_opt).fit()

regressor_ols_b.summary()


# In[100]:


# Removing x6 or Age since it has highest P value


x_opt = x_train_s[ :, [0,1,2,4,5,6,8,9,10,11,12,13]]

# Step 2

regressor_ols_b = smf.OLS(endog = y_train_s, exog = x_opt).fit()

regressor_ols_b.summary()


# # Since all P values are less than 0.05, no need to go futher.
# # Stop here
# 
# 
# 

# In[101]:


# Final Model

regressor_ols_b = smf.OLS(endog = y_train_s, exog = x_opt).fit()


# In[102]:


y_pred2 = regressor_ols_b.predict(x_test_s[ :, [0,1,2,4,5,6,8,9,10,11,12,13]])


# In[103]:


rmse2 = sqrt(mean_squared_error(y_test, y_pred2))


# In[104]:


rmse2


# #####################################################################################################################
# ####################################################################################################################

# In[105]:


# Multicollinearity

from patsy import dmatrices

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[106]:


# Subset

df2 = df[["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", 
             "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]]


# In[107]:


# Get x and y df based on this regression


y1, x1 = dmatrices("MEDV ~  CRIM + ZN +  INDUS +  CHAS +  NOX + RM + AGE +  DIS  + RAD + TAX+ PTRATIO + B + LSTAT",
                  data = df2, return_type = "dataframe")


# In[108]:


# VIF


vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])]

vif["Features"] = x1.columns


# In[109]:


vif


# In[112]:


# Plots

cols1 = ["CRIM", "LSTAT", "INDUS", "RAD", "TAX", "MEDV"]



# In[116]:


# Plotting Correlations

cm = np.corrcoef(df[cols1].values.T)

sns.set(font_scale = 1.5)

hm = sns.heatmap(cm, 
                cbar = True,
                annot = True, 
                square = True,
                fmt = ".2f",
                annot_kws = {"size" : 15},
                xticklabels = cols1,
                yticklabels = cols1)
plt.show()

