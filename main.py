import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('train.csv')
print(passengers.head(5))
print(passengers.columns)
print(passengers.dtypes)

# Update sex column to numerical
passengers['Sex'] = \
    passengers['Sex'].map({'female': 1, 'male': 0})
#print(passengers)

# Fill the nan values in the age column
#print(passengers['Age'].values)
passengers['Age'].fillna(value=passengers['Age'].mean(), inplace=True)
#print(passengers['Age'].values)


# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
#print(passengers)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
#print(features)
#print(survival)

# Perform train, test, split
train_features, test_features, train_labels, test_labels = \
    train_test_split(features,survival, test_size= 0.7 )

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create and train the model
regression = LogisticRegression()
regression.fit(train_features, train_labels)


# Score the model on the train data
print('Score of training data is ' + str(regression.score(train_features, train_labels)))

# Score the model on the test data
print('Score of test data is ' + str(regression.score(test_features, test_labels)))

# Analyze the coefficients
print(list(zip(['Sex','Age','FirstClass','SecondClass'],regression.coef_[0])))
# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Ryan = np.array([0.0,25,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array(([Jack, Rose, Ryan ]))

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
print(regression.predict(sample_passengers))
print(regression.predict_proba(sample_passengers))
