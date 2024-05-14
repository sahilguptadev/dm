#!/usr/bin/env python
# coding: utf-8
Q5. Use Naive bayes, K-nearest, and Decision tree classification algorithms and build classifiers. Divide the data set into training and test set. Compare the accuracy of the different classifiers under the following situations:
 5.1 a) Training set = 75% Test set = 25%
 b) Training set = 66.6% (2/3rd of total), Test set = 33.3% 

5.2 Training set is chosen by
 i) hold out method ii) Random subsampling iii) Cross-Validation. Compare the accuracy of the classifiers obtained.

5.3 Data is scaled to standard format.
# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score,cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

df1 = pd.read_csv('diabetes.csv')
df2 = pd.read_csv('Thyroid_Diff.csv')

df1.info()

df2.info()

list_for_onehot = ['Thyroid Function','Physical Examination','Adenopathy','Pathology',
                   'Pathology','Risk','T','N','Stage','Response']

# binary categorical --> to --> binary numerical

df2['Gender'] = (df2['Gender']=='M').astype(int)
df2['Smoking'] = (df2['Smoking']=='Yes').astype(int)
df2['Hx Smoking'] = (df2['Hx Smoking']=='Yes').astype(int)
df2['Hx Radiothreapy'] = (df2['Hx Radiothreapy']=='Yes').astype(int)
df2['Focality'] = (df2['Focality']=='Uni-Focal').astype(int)
df2['M'] = (df2['M']=='M1').astype(int)
df2['Recurred'] = (df2['Recurred']=='Yes').astype(int)

df2 = pd.get_dummies(df2, columns = list_for_onehot)
df2.info()

tf_map = {False:0, True:1}

cols_to_encode = [x for x in range(8,54)]

for col_idx in cols_to_encode:
    df2.iloc[:, col_idx] = df2.iloc[:, col_idx].map(tf_map)

df2.info()

X1 = df1.loc[:,df1.columns!='Outcome']
y1 = df1.loc[:,'Outcome']

X2 = df2.loc[:,df2.columns!='Recurred']
y2 = df2.loc[:,'Recurred']

# a) 75%-25%
X1_train_A, X1_test_A, y1_train_A, y1_test_A = train_test_split(X1, y1, test_size=0.25, random_state=42)
X2_train_A, X2_test_A, y2_train_A, y2_test_A = train_test_split(X2, y2, test_size=0.25, random_state=42)

# b) 66.6%-33.3%
X1_train_B, X1_test_B, y1_train_B, y1_test_B = train_test_split(X1, y1, test_size=0.33, random_state=42)
X2_train_B, X2_test_B, y2_train_B, y2_test_B = train_test_split(X2, y2, test_size=0.33, random_state=42)


# In[ ]:


#5.1 Evaluation on Train-Test Split as 75-25 and 66.6-33.3
classifiers = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}

print('For DF1: 75% Train - 25% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X1_train_A, y1_train_A)

    # Evaluate the classifier
    y_pred = clf.predict(X1_test_A)
    accuracy = accuracy_score(y1_test_A, y_pred)
    print('  ', f"{name} Accuracy: {accuracy:.2f}")

print('\nFor DF1: 66.6% Train - 33.3% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X1_train_B, y1_train_B)

    # Evaluate the classifier
    y_pred = clf.predict(X1_test_B)
    accuracy = accuracy_score(y1_test_B, y_pred)
    print('  ', f"{name} Accuracy: {accuracy:.2f}")

print('\nFor DF2: 75% Train - 25% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X2_train_A, y2_train_A)

    # Evaluate the classifier
    y_pred = clf.predict(X2_test_A)
    accuracy = accuracy_score(y2_test_A, y_pred)
    print('  ', f"{name} Accuracy: {accuracy:.2f}")

print('\nFor DF2: 66.6% Train - 33.3% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X2_train_B, y2_train_B)

    # Evaluate the classifier
    y_pred = clf.predict(X2_test_B)
    accuracy = accuracy_score(y2_test_B, y_pred)
    print('  ', f"{name} Accuracy: {accuracy:.2f}")


# In[ ]:


#5.2 Evaluation using Holdout, Random Subsampling and 5-Fold CV

# a) holdout (70%-30%)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, stratify=y1, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, stratify=y2, random_state=42)

print('Holdout Method for DF1: 70% Train - 30% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X1_train, y1_train)

    # Evaluate the classifier
    y_pred = clf.predict(X1_test)
    accuracy = accuracy_score(y1_test, y_pred)
    print('  ',f"{name} Accuracy: {accuracy:.2f}")

print('\nHoldout Method for DF2: 70% Train - 30% Test')
for name, clf in classifiers.items():


    # Train the classifier
    clf.fit(X2_train, y2_train)


    # Evaluate the classifier
    y_pred = clf.predict(X2_test)
    accuracy = accuracy_score(y2_test, y_pred)
    print('  ',f"{name} Accuracy: {accuracy:.2f}")
    

# b) random subsample 

X1_train_1, X1_test_1, y1_train_1, y1_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=42)  # 80-20
X1_train_2, X1_test_2, y1_train_2, y1_test_2 = train_test_split(X1, y1, test_size=0.33, random_state=42)  # 66.6-33.3
X1_train_3, X1_test_3, y1_train_3, y1_test_3 = train_test_split(X1, y1, test_size=0.3, random_state=42)  # 70-30

X2_train_1, X2_test_1, y2_train_1, y2_test_1 = train_test_split(X2, y2, test_size=0.2, random_state=42)  # 80-20
X2_train_2, X2_test_2, y2_train_2, y2_test_2 = train_test_split(X2, y2, test_size=0.33, random_state=42)  # 66.6-33.3
X2_train_3, X2_test_3, y2_train_3, y2_test_3 = train_test_split(X2, y2, test_size=0.3, random_state=42)  # 70-30

print('Random Subsample for DF1:')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X1_train_1, y1_train_1)
    y_pred_1 = clf.predict(X1_test_1)
    acc_1 = accuracy_score(y1_test_1, y_pred_1)

    clf.fit(X1_train_2, y1_train_2)
    y_pred_2 = clf.predict(X1_test_2)
    acc_2 = accuracy_score(y1_test_2, y_pred_2)

    clf.fit(X1_train_3, y1_train_3)
    y_pred_3 = clf.predict(X1_test_3)
    acc_3 = accuracy_score(y1_test_3, y_pred_3)

    accuracy = (acc_1 + acc_2 + acc_3) / 3

    print('  ', f"{name} Accuracy: {accuracy:.2f}")

print('\nRandom Subsample for DF2:')

for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X2_train_1, y2_train_1)
    y_pred_1 = clf.predict(X2_test_1)
    acc_1 = accuracy_score(y2_test_1, y_pred_1)

    clf.fit(X2_train_2, y2_train_2)
    y_pred_2 = clf.predict(X2_test_2)
    acc_2 = accuracy_score(y2_test_2, y_pred_2)

    clf.fit(X2_train_3, y2_train_3)
    y_pred_3 = clf.predict(X2_test_3)
    acc_3 = accuracy_score(y2_test_3, y_pred_3)

    accuracy = (acc_1 + acc_2 + acc_3) / 3

    print('  ', f"{name} Accuracy: {accuracy:.2f}")


# In[ ]:


# c) 5-Fold Cross-Validation 

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scoring = ['accuracy']

models = [KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier()]

print('5-Fold CV for DF1:')
for model in models:
    result = list()
    scores = cross_validate(model, X1, y1, cv=kf, scoring=scoring)

    for value in scores:
        v = str(value)
        mean_score = scores[v].mean()
        std_score = scores[v].std()
        if (v == "fit_time" or v == "score_time"):
            pass
        else:
            print(f"{model} --> {mean_score:.2f} ± {std_score:.2f}")

print('\n5-Fold CV for DF2:')
for model in models:
    result = list()
    scores = cross_validate(model, X2, y2, cv=kf, scoring=scoring)

    for value in scores:
        v = str(value)
        mean_score = scores[v].mean()
        std_score = scores[v].std()
        if (v == "fit_time" or v == "score_time"):
            pass
        else:
            print(f"{model} --> {mean_score:.2f} ± {std_score:.2f}")


# In[ ]:


#5.3 Results after scaling the values

scaler = StandardScaler()

X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, stratify=y1, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, stratify=y2, random_state=42)

print('After Scaling values for DF1: 80% Train - 20% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X1_train, y1_train)

    # Evaluate the classifier
    y_pred = clf.predict(X1_test)
    accuracy = accuracy_score(y1_test, y_pred)
    print('  ',f"{name} Accuracy: {accuracy:.2f}")

print('\nAfter Scaling values for DF2: 80% Train - 20% Test')
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X2_train, y2_train)

    # Evaluate the classifier
    y_pred = clf.predict(X2_test)
    accuracy = accuracy_score(y2_test, y_pred)
    print('  ',f"{name} Accuracy: {accuracy:.2f}")

