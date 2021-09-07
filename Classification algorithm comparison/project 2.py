import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
header = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated',
          'ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems',
          'Browser','Region','TrafficType','VisitorType','Weekend','Revenue'
]

train = pandas.read_csv("dataset.csv", header=None , names= header , index_col=False , skipinitialspace=True)
# test = pandas.read_csv("adult.test.csv", header=None , names = header , index_col=False , na_values="?" , skipinitialspace=True)
# dataset = pandas.concat([train,test])
dataset = pandas.concat([train])
# dataset.replace('?', np.NaN)
dataset.info()

# print("data is null:",dataset.isnull().sum().sum())

for col in header:
    print ("values of %s"%(col))
    print (dataset[col].value_counts())
    print (dataset[col].count())
    print ('////////////////////////////\n')
# print(dataset.loc)
# print (dataset.isnull().sum())

plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(2, 2, 1)
sns.countplot(dataset['Revenue'], palette = 'pastel')
plt.title('Buy or Not', fontsize = 10,color='red')
plt.xlabel('Revenue or not', fontsize = 10)
plt.ylabel('count', fontsize = 10)


# checking the Distribution of customers on Weekend
plt.subplot(2, 2, 2)
sns.countplot(sorted(dataset['Weekend']), palette = 'inferno')
plt.title('Purchase on Weekends', fontsize = 10,color='red')
plt.xlabel('Weekend or not', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 3)
sns.countplot(dataset['TrafficType'], palette = 'husl')
plt.title('TrafficTypes', fontsize = 10,color='red')
plt.xlabel('TrafficType', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 4)
sns.countplot(dataset['VisitorType'], palette = 'Paired')
plt.title('VisitorTypes', fontsize = 10,color='red')
plt.xlabel('VisitorType', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.show()
# //////////////////////////////////////////////////////////////////////////////////
plt.subplot(2, 2, 1)
sns.countplot(dataset['TrafficType'], palette = 'pastel')
plt.title('Traffic Type', fontsize = 10,color='red')
plt.xlabel('Traffic Types', fontsize = 10)
plt.ylabel('count', fontsize = 10)


# checking the Distribution of customers on Weekend
plt.subplot(2, 2, 2)
sns.countplot(dataset['Region'], palette = 'inferno')
plt.title('Regions', fontsize = 10,color='red')
plt.xlabel('Region', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 3)
sns.countplot(dataset['Browser'], palette = 'husl')
plt.title('Browser', fontsize = 10,color='red')
plt.xlabel('Browser', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 4)
sns.countplot(dataset['OperatingSystems'], palette = 'Paired')
plt.title('OperatingSystems', fontsize = 10,color='red')
plt.xlabel('OperatingSystem', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.show()
# ////////////////////////////////////////////////////////

plt.rcParams['figure.figsize'] = (18, 15)

plt.subplot(2, 2, 1)
sns.boxenplot(dataset['Revenue'], dataset['Administrative'], palette = 'rainbow')
plt.title('Administrative', fontsize = 10,color='red')
plt.ylabel('Administrative', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# product related duration vs revenue

plt.subplot(2, 2, 2)
sns.boxenplot(dataset['Revenue'], dataset['Administrative_Duration'], palette = 'pastel')
plt.title('Administrative_Duration', fontsize = 10,color='red')
plt.ylabel('Administrative_Duration', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# product related duration vs revenue

plt.subplot(2, 2, 3)
sns.boxenplot(dataset['Revenue'], dataset['Informational'], palette = 'dark')
plt.title('Informational', fontsize = 10,color='red')
plt.ylabel('Informational', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# exit rate vs revenue

plt.subplot(2, 2, 4)
sns.boxenplot(dataset['Revenue'], dataset['Informational_Duration'], palette = 'spring')
plt.title('Informational_Duration', fontsize = 10,color='red')
plt.ylabel('Informational_Duration', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)



plt.show()
# ////////////////////////////////////////////////////////////

plt.rcParams['figure.figsize'] = (18, 15)

plt.subplot(2, 2, 1)
sns.boxenplot(dataset['Revenue'], dataset['ProductRelated'], palette = 'rainbow')
plt.title('ProductRelated', fontsize = 10,color='red')
plt.ylabel('ProductRelated', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# product related duration vs revenue

plt.subplot(2, 2, 2)
sns.boxenplot(dataset['Revenue'], dataset['ProductRelated_Duration'], palette = 'pastel')
plt.title('ProductRelated_Duration', fontsize = 10,color='red')
plt.ylabel('ProductRelated_Duration', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# product related duration vs revenue

plt.subplot(2, 2, 3)
sns.boxenplot(dataset['Revenue'], dataset['BounceRates'], palette = 'dark')
plt.title('BounceRates', fontsize = 10,color='red')
plt.ylabel('BounceRates', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# exit rate vs revenue

plt.subplot(2, 2, 4)
sns.boxenplot(dataset['Revenue'], dataset['ExitRates'], palette = 'spring')
plt.title('ExitRates', fontsize = 10,color='red')
plt.ylabel('ExitRates', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)



plt.show()
# /////////////////////////////////////////////////////////
plt.subplot(1, 2, 1)
sns.boxenplot(dataset['Revenue'], dataset['PageValues'], palette = 'dark')
plt.title('PageValues', fontsize = 10,color='red')
plt.ylabel('PageValues', fontsize = 10)
plt.xlabel('Revenue', fontsize = 10)

# exit rate vs revenue

size = [11079, 351, 325, 243, 178,154]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen','blue']
labels = "0", "0.6","0.8","0.4","0.2",'1'
explode = [0, 0, 0, 0, 0,0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Special days', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()


plt.show()


dataset.loc[dataset['SpecialDay']!=0.0,'SpecialDay'] = 1
dataset.loc[dataset['PageValues']!=0,'PageValues'] = 1
dataset.loc[dataset['Informational_Duration']!=0,'Informational_Duration'] = 1
dataset.loc[dataset['Informational']!=0,'Informational'] = 1
dataset['Revenue'] = dataset['Revenue'].map({ False:0, True:1})
dataset['Weekend'] = dataset['Weekend'].map({ False:0, True:1})
dataset['VisitorType'] = dataset['VisitorType'].map({ 'Returning_Visitor':0, 'New_Visitor':1,'Other':2})
dataset['Month'] = dataset['Month'].map({'May':1, 'Nov':2, 'Mar':3,
                                                 'Dec':4, 'Oct':5, 'Sep':6,
                                                 'Aug':7, 'Jul':8, 'June':9, 'Feb':10})
for col in header:
    print ("values of %s"%(col))
    print (dataset[col].value_counts())
    print (dataset[col].count())
    print ('////////////////////////////\n')

corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
#
dataset.drop(labels=["Administrative_Duration","Informational_Duration","ProductRelated_Duration",
                     "ExitRates"], axis = 1, inplace = True)
test = dataset['Revenue']
train = dataset.drop('Revenue', axis=1)

validation_size = 0.20
num_folds = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(train,test,
    test_size=validation_size)

models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=20)))
models.append(('Decision Tree Gini', DecisionTreeClassifier(criterion='gini')))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Random Forest', RandomForestClassifier()))
results = []
names = []
for name, model in models:
    kfold = KFold(3)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    print (" for stage 3 %s: %f" % (name, cv_results.mean()))
    kfold = KFold(num_folds)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f" % (name, cv_results.mean())
    print(msg)
scaler=StandardScaler()
models.append(('mlp', MLPClassifier()))
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_validation)
cv_results=cross_val_score(MLPClassifier(),X_train_scaled,Y_train)
results.append(cv_results)
names.append("MLP")
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()