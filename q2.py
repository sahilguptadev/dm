#!/usr/bin/env python
# coding: utf-8
Q2. Perform the following preprocessing tasks on the dirty_iris dataset.
i.Calculate the number and percentage of observations that are complete. 
ii. Replace all the special values in data with NA.
iii. Define these rules in a separate text file and read them. (Use editfile function in R (package editrules). Use similar function in Python). Print the resulting constraint object.
–Species should be one of the following values: setosa, versicolor or virginica.
–All measured numerical properties of an iris should be positive.
–The petal length of an iris is at least 2 times its petal width. 
–The sepal length of an iris cannot exceed 30 cm.
–The sepals of an iris are longer than its petals.
iv.Determine how often each rule is broken (violatedEdits). Also summarize and plot the result.
Find outliers in sepal length using boxplot and boxplot.stats
# In[ ]:


#create the req files before running
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris_dirty.csv')
df.info()

df['Sepal.Width'].value_counts()

# [i] Number and percentage of complete obeservations

df_completeobs = df.dropna()

total_obs = len(df) 
notna_obs = len(df_completeobs)
perc_notna_obs = notna_obs/total_obs*100 #BODMAS WORKS 

print('Total Observations: ', total_obs)
print('Complete Observations: ', notna_obs)
print('Percentage of Complete Observations: ', perc_notna_obs,' %', sep='')


# In[ ]:


#Replace all special values with NA

cols_to_check = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']

df[df[cols_to_check] == 'inf'] = pd.NA


# In[ ]:


#[iii] Ruleset definition

def DirtyIrisRuleset(row):
   errorlist = []

   #rule 1
   if row['Species'] not in ['setosa','versicolor','virginica']:
       errorlist.append("species must be 'setosa' or 'versicolor' or 'virginica'")
   
   #rule 2
   if (row['Sepal.Length'] <= 0) or (row['Sepal.Width'] <= 0):
       errorlist.append('numerical properties cannot be zero')
   elif (row['Petal.Length'] <= 0) or (row['Petal.Length'] <= 0):
       errorlist.append('numerical properties cannot be zero')
   else:
       pass

   #rule 3
   if row['Sepal.Length'] < row['Petal.Width']:
       errorlist.append('sepal length must be atleast 2x petal width')
   
   #rule 4
   if row['Sepal.Length'] > 30:
       errorlist.append("sepal length must not be > 30")

   #rule 5
   if row['Sepal.Length'] <= row['Petal.Length']:
       errorlist.append("sepal length must be longer than petal length")

   return errorlist


# In[ ]:


#Importing Ruleset
from ruleset_for_dmp2 import DirtyIrisRuleset as E

# [iv] Applying Rulset and visualizing results
df['ERRORS'] = df.apply(E, axis=1)

# data summarization 
vio = df[df['ERRORS'].apply(lambda x: len(x) > 0)]
df = df.drop(columns='ERRORS')

# Flatten the list of errors and count occurrences
error_counts = vio['ERRORS'].explode().value_counts()
print(error_counts)

# Plotting
error_counts.plot(kind='bar')
plt.title('Visualization of Rule Violations')
plt.xlabel('Rules')
plt.ylabel('Number of Violations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# plotting boxplot
plt.figure()
sns.boxplot(x=df['Sepal.Length'])
plt.title('Boxplot of Sepal Length')
plt.show()

# indentifying outliers using interquartile ranges
Q1 = df['Sepal.Length'].quantile(0.25)
Q3 = df['Sepal.Length'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# detecting and printing outliers
outliers = df[(df['Sepal.Length'] < lower_bound) | (df['Sepal.Length'] > upper_bound)]
print("Detected Outliers:")
print(outliers)

