# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
gender = pd.read_csv('gender_submission.csv')

train['Fare'].fillna(train['Fare'].mean(), inplace=True)
train['Age'].fillna(int(train['Age'].mean()), inplace=True)

simple = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], 1)

#####################################################################################
#===============   GRAPH 1 : PROPORTIONS OF SURVIVORS BY AGE AND FARE (meaningless)
#####################################################################################

survivors = train.groupby('Survived')
survived = survivors.get_group(1)
deceased = survivors.get_group(0)

plt.figure(1)
sur = plt.scatter(survived['Fare'], survived['Age'], c='red', marker='o')
dec = plt.scatter(deceased['Fare'], deceased['Age'], c='black', marker='x')

plt.xlabel('Fare')
plt.ylabel('Age')
plt.title('Proportion of survivors by age and fare')
plt.legend((sur, dec), ('Survived', 'Deceased'))

plt.savefig('graph1.png')
#plt.savefig('graph11.png', bbox_inches='tight') ## To avoid whitespaces around the image

#############################################################################
#================         SOME VARIABLES       ============================ 
#############################################################################

sexes = train.groupby('Sex')
females = sexes.get_group('female')
males = sexes.get_group('male')
m_survived = males['Survived'].sum()
m_deceased = males['Survived'].shape[0]
f_survived = females['Survived'].sum()
f_deceased = females['Survived'].shape[0]

passengers = [m_survived, m_deceased, f_survived, f_deceased]

#############################################################################
#================   GRAPH 2 : SURVIVORS AND DECEASED CLASSIFIED BY SEX
#############################################################################

plt.figure(2)

i = np.arange(2)
width = 0.2

rect1 = plt.bar(i, [m_survived, m_deceased], width, color='blue')
rect2 = plt.bar(i+width, [f_survived, f_deceased], width, color='red')

plt.title('Proportion of survivors by sex')
plt.ylabel('Number of people')
plt.xticks(i+(width/2), ('Survived', 'Deceased'))
plt.legend((rect1, rect2), ('Men', 'Women'))
plt.grid(linestyle='--')

def autolabel(rects, ratio):    
    for rect in rects:
        h = rect.get_height()
        plt.text(rect.get_x()+(width/ratio), h+5, '%d'%int(h))
        
autolabel(rect1, 3)
autolabel(rect2, 3)

plt.savefig('graph2.png')

#############################################################################
#================   GRAPH 3 : PEOPLE COUNT BY AGE BIN
#############################################################################

plt.figure(3)
plt.hist(train['Age'], bins=16)
plt.grid(linestyle='--')
#or...
#plt.figure(4)
#train['Age'].hist(bins=16)

plt.savefig('graph3.png')

###################################################################################
#================   GRAPH 4 : CHANCES OF SURVIVAL IN REGARDS TO SibSp AND PARCH
###################################################################################
sib = train['SibSp'].max() 
parch = train['Parch'].max()
n = parch if parch > sib else sib
idx = np.arange(n+1)
width = 0.45

sib_group = train.groupby('SibSp')
parch_group = train.groupby('Parch')

sib_sur = []
parch_sur = []

for i in idx:
    try:
        sib_sur.append(sib_group.get_group(i)['Survived'].sum())
    except:
        sib_sur.append(0)
    try:
        parch_sur.append(parch_group.get_group(i)['Survived'].sum())
    except:
        parch_sur.append(0)    
    

plt.figure(4)
sib = plt.bar(idx, sib_sur, width) 
parch = plt.bar(idx+width, parch_sur, width)

plt.xlabel('ParCh/SibSp')
plt.ylabel('Number of survivors')
plt.title('Number of survivors by number of relatives')
plt.legend((sib, parch), ('Siblings/Spouse', 'Parents/Children'))
plt.xticks(idx+width/2, idx)

autolabel(sib, 150)
autolabel(parch, 150)

plt.savefig('graph4.png')

###################################################################################
#================   GRAPH 5 : CHANCES OF SURVIVAL BY PASSENGER CLASS
###################################################################################

pclass = train.groupby('Pclass')
idx = np.arange(1, train['Pclass'].max()+1)
classes = []

for i in idx:
    try:
        classes.append(pclass.get_group(i)['Survived'].sum())
    except:
        classes.append(0)

plt.figure(5)
classes_bar = plt.bar(idx, classes, width)

plt.title('Chances of survival by passenger class')
plt.xlabel('Passenger class')
plt.ylabel('Number of survivors')
plt.xticks(idx, idx)
autolabel(classes_bar, 3)

plt.savefig('graph5.png')

###################################################################################
#================   GRAPH 6 : DIMENSION REDUCTION USING PCA
###################################################################################

#~~~~~~~~Step 1: Preprocessing~~~~~~~~~~~~~~~~~~~~~

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

simple = pd.get_dummies(data=simple, columns=['Sex'])
features = list(simple.columns)
features.remove('Survived')

x = simple.loc[:, features].values
y = simple.loc[:, ['Survived']].values

x = StandardScaler().fit_transform(x)

X = PCA(n_components=2).fit_transform(x)

principal = pd.DataFrame(data=X, columns=['PC 1', 'PC 2'])
final = pd.concat([principal, simple['Survived']], axis=1)

#~~~~~~~~Step 2: Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure(6)
colors = ['red', 'black']
markers = ['o', 'x']
statuses = [0, 1]

final_groups = final.groupby('Survived')

for color, marker, status in zip(colors, markers, statuses):
    group = final_groups.get_group(status)
    plt.scatter(x=group['PC 1'], y=group['PC 2'], color=color, marker=marker)
    
plt.title('2D Principal Component Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(('Survived', 'Deceased'))

plt.savefig('graph6.png')

#######################TRAINING######################"""#
import training
training.train(principal, simple['Survived'])

