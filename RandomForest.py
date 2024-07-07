#------RandomForest---------
import pandas as pd
df = pd.read_csv("C:/Users/lgino/OneDrive/Bureau/bigdata/dc_modelisation.csv", sep=';')
clean_df = df.loc[df["NB"] != 0 , :  ]
features = clean_df.drop(columns=['CMD'])
import numpy as np
labels = np.array(features['RETOUR'])
features = features.drop('RETOUR',axis = 1)
feature_list = list(features.columns)
features = np.array(features)
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'retour.')
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#-----Arbre png------

from sklearn.tree import export_graphviz
import pydot

tree = rf.estimators_[5]

export_graphviz(tree, out_file = 'C:/Users/lgino/OneDrive/Bureau/bigdata/tree.dot',feature_names = feature_list, rounded = True, precision = 1)

(graph,) = pydot.graph_from_dot_file('C:/Users/lgino/OneDrive/Bureau/bigdata/tree.dot')

graph.write_png('C:/Users/lgino/OneDrive/Bureau/bigdata/tree.png')


#------Importance des éléments-------

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#------Importance élements tableau ------

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

%matplotlib inline

# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');