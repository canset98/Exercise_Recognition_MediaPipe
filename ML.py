import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import os

"""# Read And Merge"""

labels = pd.read_csv('labels.csv')
distances = pd.read_csv('3d_distances.csv')
angles = pd.read_csv('angles.csv')

features = pd.merge(distances, angles, on="pose_id")
all_data = pd.merge(labels, features, on="pose_id")
"""# List of columns to remove
columns_to_remove = ['left_hip_avg_left_wrist_left_ankle','right_hip_avg_right_wrist_right_ankle']
# Use pop in a loop to remove each column
for column in columns_to_remove:
    all_data.pop(column)"""
# Exporting the DataFrame to a CSV file
all_data.to_csv(r'C:\Users\Ranim\Desktop\Bahar dönemi 2024\BİTİRME\Makaleler\Dataset\archive\all_data.csv',index=False)
#inputs as numpy array
X = all_data.drop(['pose','pose_id'],axis=1).values

y = all_data['pose'].values


# Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)



# Scalling

#scaler = MinMaxScaler(feature_range = (-1,1))
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Training

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediction

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))