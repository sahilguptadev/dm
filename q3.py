#!/usr/bin/env python
# coding: utf-8
Load the data from wine dataset. Check whether all attributes are standardized or not (mean is 0 and standard deviation is 1). If not, standardize the attributes. Do the same with Iris dataset.
# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Wine dataset
wine_df = pd.read_csv("Wine dataset.csv")
wine_df.head(5)

# Check if attributes are standardized
if ((iris_df.mean(axis=0) == 0).all() and (iris_df.std(axis=0) == 1).all()):
    print("All attributes in the iris dataset are already standardized")
else:
    print("All attributes in the iris dataset are not standardized")


# In[ ]:


# Standardize the attributes
scaler = StandardScaler()
iris_df_standardized = scaler.fit_transform(iris_df)
iris_df = pd.DataFrame(iris_df_standardized, columns=iris_df.columns)
print("Now, all attributes are standardized")


# In[ ]:


# Load the Iris dataset
iris_df = pd.read_csv("iris_dirty.csv")
iris_df = iris_df.drop(columns=['Species']).dropna()
iris_df.head(7)


# In[ ]:


# Check if attributes are standardized
if ((iris_df.mean(axis=0) == 0).all() and (iris_df.std(axis=0) == 1).all()):
    print("All attributes in the iris dataset are already standardized")
else:
    print("All attributes in the iris dataset are not standardized")


# In[ ]:


# Standardize the attributes
scaler = StandardScaler()
iris_df_standardized = scaler.fit_transform(iris_df)
iris_df = pd.DataFrame(iris_df_standardized, columns=iris_df.columns)
print("Now, all attributes are standardized")

