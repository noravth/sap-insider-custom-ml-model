import os
#
# Variables
DATA_PATH = '/app/data/train.csv'
MODEL_PATH = '/app/model/model.pkl'
#
# Load Datasets
import pandas as pd
df = pd.read_csv(DATA_PATH)
print(df.size)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['airline'] = le.fit_transform(df['airline'])
df['flight'] = le.fit_transform(df['flight'])
df['source_city'] = le.fit_transform(df['source_city'])
df['departure_time'] = le.fit_transform(df['departure_time'])
df['stops'] = le.fit_transform(df['stops'])
df['arrival_time'] = le.fit_transform(df['arrival_time'])
df['destination_city'] = le.fit_transform(df['destination_city'])
df['class'] = le.fit_transform(df['class'])
X = df.drop('price', axis=1)
y = df['price']
#
# Partition into Train and test dataset
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
#
# Init model
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_depth=10)
#
# Train model
clf.fit(train_x, train_y)
#
# Test model
test_r2_score = clf.score(test_x, test_y)
# Output will be available in logs of SAP AI Core.
# Not the ideal way of storing /reporting metrics in SAP AI Core, but that is not the focus this tutorial
print(f"Test Data Score {test_r2_score}")
#
# Save model
import pickle
pickle.dump(clf, open(MODEL_PATH, 'wb'))
