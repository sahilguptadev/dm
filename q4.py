#!/usr/bin/env python
# coding: utf-8
Section 2: Data Mining Techniques 
Run following algorithms on 2 real datasets and use appropriate evaluation measures to compute correctness of obtained patterns:

Q4. Run Apriori algorithm to find frequent itemsets and association rules 
i)Use minimum support as 50% and minimum confidence as 75%
ii)Use minimum support as 60% and minimum confidence as 60%
# In[ ]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_excel('Online Retail.xlsx')
df

#Load the data for your first dataset
data1 = df[df.Country=="France"]

# Load the data (replace with your own dataset)
data = data1.copy()  # Make a copy of the original dataframe

# Clean the data
data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]

# Split data by region (e.g., France, UK, etc.)
basket = (data
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))


def hot_encode(x):
    if isinstance(x, float):
        return 0 if x <= 0 else 1
    else:
        return x.map(hot_encode)

basket_encoded = basket.apply(hot_encode)
basket_encoded = basket_encoded.astype(bool)

# Build the model
frq_items = apriori(basket_encoded, min_support=0.05, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head())


assc_rules = rules.loc[:,["antecedents","consequents","support","confidence"]]
assc_rules

#4.1
#Selecting all the rules which have a minimun supprt of 50% and a minimum confidence of 75%
temp = assc_rules[assc_rules.support>0.05]
rules1 = temp[assc_rules.confidence>0.75]
rules1

# Selecting all the rules which have a minimum support of 60% and a minimum confidence of 60%
temp = assc_rules[assc_rules.support > 0.06]
rules2 = temp[temp.confidence > 0.6]
rules2

#Load data for the second dataset
data2 = df[df.Country=="Spain"]

# Load the data (replace with your own dataset)
data = data2.copy()

# Clean the data
data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]

# Split data by region (e.g., France, UK, etc.)
basket = (data
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def hot_encode(x):
    return x.apply(lambda val: 0 if val <= 0 else 1)


basket_encoded = basket.apply(hot_encode, axis=1)

# Convert DataFrame to boolean type
basket_encoded = basket_encoded.astype(bool)

# Build the model
frq_items = apriori(basket_encoded, min_support=0.05, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head())


assc_rules = rules.loc[:,["antecedents","consequents","support","confidence"]]
assc_rules

#Selecting all the rules which have a minimun supprt of 50% and a minimum confidence of 75%
temp = assc_rules[assc_rules.support>0.05]
rules2 = temp[assc_rules.confidence>0.75]
rules2

temp = assc_rules[assc_rules.support>0.06]
rules4 = temp[assc_rules.confidence>0.60]
rules4


