#!/usr/bin/env python
# coding: utf-8
Section 1: Preprocessing 
Q1. Create a file “people.txt” with the following data:
i.Read the data from the file “people.txt”.
ii.Create a ruleset E that contain rules to check for the following conditions: 
1. The age should be in the range 0-150.
2. The age should be greater than yearsmarried. 
3. The status should be married or single or widowed.
4. If age is less than 18 the agegroup should be child, if age is between 18 and 65 the agegroup should be adult, if age is more than 65 the agegroup should be elderly.
iii.Check whether ruleset E is violated by the data in the file people.txt. 
iv.Summarize the results obtained in part (iii)
 Visualize the results obtained in part (iii)
# In[ ]:


#Before implementing thi code create the file with given data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# [i] reading data from file

path = 'people.txt'
df = pd.read_table(path, sep=',', header=0)

df
df.info()


# In[ ]:


# [ii] Ruleset definition

def E(row):
    errorlist = []

    #rule 1
    if not (0 <= row['age'] <= 150):
        errorlist.append('age should be in range 0-150')

    #rule 2
    if not(row['age'] > row['yearsmarried']):
        errorlist.append('age should be greater than years married')

    #rule 3
    if row['status'] not in ['single','married','widowed']:
        errorlist.append("status must be 'single' or 'married' or 'widowed'")

    #rule 4
    #expected_agegroup = ''
    if row['age'] < 18:
        expected_agegroup = 'child'
    elif 18 <= row['age'] < 65:
        expected_agegroup = 'adult'
    else:
        expected_agegroup = 'elderly'

    if row['agegroup'] != expected_agegroup:
        errorlist.append(f"expected age group:'{expected_agegroup}', received: '{row['agegroup']}'")

    return errorlist


# In[ ]:


# [iii] Applying Rulset to check for rule violations
df['ERRORS'] = df.apply(E, axis=1)
df


# In[ ]:


# [iv] data summarization

vio = df[df['ERRORS'].apply(lambda x: len(x) > 0)]
vio


# In[ ]:


# [v] visualizing the summarizations

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

