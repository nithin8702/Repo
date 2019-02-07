# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:47:31 2019

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from subprocess import call

df_train = pd.read_csv('training.csv')
df_Test = pd.read_csv('testing.csv')

df_train['class'].unique()

df_train.columns

X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values

X_test = df_Test.iloc[:, 1:].values
y_test = df_Test.iloc[:, 0].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = classifier.predict(X_test)
#
#
#cm = confusion_matrix(y_test, y_pred)
#
#accuracy_score(y_test, y_pred)
#

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 0)
#visualize_classifier(classifier, X_train, y_train);
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)



#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
#accuracy_score(y_test, y_pred)
#
#
#
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
#accuracy_score(y_test, y_pred)
#
#
#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#classifier.fit(X_train, y_train)
#y_pred = classifier.predict(X_test)
#accuracy_score(y_test, y_pred)
#

# Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Classifier (Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()

# Visualising the Test set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Classifier (Test set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
#

#def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
#    ax = ax or plt.gca()
#    
#    # Plot the training points
#    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
#               clim=(y.min(), y.max()), zorder=3)
#    ax.axis('tight')
#    ax.axis('off')
#    xlim = ax.get_xlim()
#    ylim = ax.get_ylim()
#    
#    # fit the estimator
#    model.fit(X, y)
#    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
#                         np.linspace(*ylim, num=200))
#    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#
#    # Create a color plot with the results
#    n_classes = len(np.unique(y))
#    contours = ax.contourf(xx, yy, Z, alpha=0.3,
#                           levels=np.arange(n_classes + 1) - 0.5,
#                           cmap=cmap, clim=(y.min(), y.max()),
#                           zorder=1)
#
#    ax.set(xlim=xlim, ylim=ylim)
#    
#visualize_classifier(classifier, X_train, y_train)

#from sklearn.tree import export_graphviz
#export_graphviz(classifier.estimators_[5], 
#                out_file='tree.dot', 
#                feature_names = df_train.columns[1:],
##                class_names = iris.target_names,
#                rounded = True, proportion = False, 
#                precision = 2, filled = True)
#
#from subprocess import call
#call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
#from IPython.display import Image
#Image(filename = 'tree.png')






#modelname.feature_importance_
#y = classifier.feature_importances_
##plot
#fig, ax = plt.subplots() 
#width = 0.4 # the width of the bars 
#ind = np.arange(len(y)) # the x locations for the groups
#ax.barh(ind, y, width, color='green')
#ax.set_yticks(ind+width/10)
#ax.set_yticklabels(col, minor=False)
#
#plt.title('Feature importance in RandomForest Classifier')
#plt.xlabel('Relative importance')
#plt.ylabel('feature') 
#plt.figure(figsize=(5,5))
#fig.set_size_inches(6.5, 4.5, forward=True)
#
#


import os
from sklearn.tree import export_graphviz
import six
import pydot
from sklearn import tree
import matplotlib.image as mpimg

col = df_train.columns[1:]
dotfile = six.StringIO()
i_tree = 0
for tree_in_forest in classifier.estimators_:
    export_graphviz(tree_in_forest,out_file='tree.dot',
    feature_names=col,
    filled=True,
    rounded=True)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    name = 'tree' + str(i_tree)
    graph.write_png(name+  '.png')
    os.system('dot -Tpng tree.dot -o tree.png')
    i_tree +=1
    
f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
    
img=mpimg.imread('tree0.png')
imgplot = plt.imshow(img)


img1 = mpimg.imread('tree0.png')
img2 = mpimg.imread('tree1.png')

fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
axs[0].plot(img1)
axs[0].plot(img2)
plt.show()

plt.figure(1)
plt.subplot(211, figsize=(9, 3),)
plt.imshow(img1)

plt.subplot(212, figsize=(9, 3),)
plt.imshow(img2)
plt.show()
 
 
    
    
#_grid = np.arange(min(X_test), max(X_test), 0.01)
#X_grid = X_test.reshape((len(X_test), 1))
#plt.scatter(X, y, color = ‘red’)
#plt.plot(X_grid, regressor.predict(X_grid), color = ‘blue’)
#plt.title(‘Random Forest Regression Model’)
#plt.xlabel(‘Years’)
#plt.ylabel(‘Account Balance’)
#plt.show()
#
#
feature_imp = pd.Series(classifier.feature_importances_, index= df_train.columns[1:]).sort_values(ascending=False)
sns.barplot(x=feature_imp, y =feature_imp.index)
plt.xlabel('Feature importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.legend(handles=[])
plt.show()
#

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes = df_train['class'].unique(),title = 'Confusion Matrix')
#df_train.columns[0]



#export_graphviz(classifier, 'tree_real_data.dot', rounded = True, 
#                feature_names = df_train.columns[1:], max_depth = 6,
#                class_names = df_train['class'].unique(), filled = True)
#
## Convert to png
#call(['dot', '-Tpng', 'tree_real_data.dot', '-o', 'tree_real_data.png', '-Gdpi=200'])
#
## Visualize
#Image(filename='tree_real_data.png')

#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='rainbow');


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    
visualize_classifier(classifier, X_test, y_test)


sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');



w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, 10):
    img = mpimg.imread('tree' + str(i) + '.png')    
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()