
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.info()
train_data.describe()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

train_data['Embarked'].describe()

data = [train_data, test_data]

for dataset in data:
    mean = train_data["Age"].mean()
    std = train_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)

common_value = 'S'
data = [train_data, test_data]
gender = {"male": 0, "female": 1}
port = {"S": 0, "C": 1, "Q": 2}


for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Sex'] = dataset['Sex'].map(gender)
    dataset['Embarked'] = dataset['Embarked'].map(port)
    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66) , 'Age'] = 6
    dataset.loc[(dataset['Age'] > 66) , 'Age'] = 6
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)



data = [train_data, test_data]
for dataset in data:
    dataset['FamSize'] = dataset['SibSp'] + dataset['Parch'] + 1
    print(dataset['FamSize'].unique())



data = [train_data, test_data]
for dataset in data:
    dataset['FamType'] = pd.cut(dataset.FamSize,[0,1,4,7,11],labels=[1,2,3,4])
    dataset['FamType'] = dataset['FamType'].astype(int)
    print(dataset['FamType'].unique())


data = [train_data, test_data]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)

    
train_data = train_data.drop(['Name'], axis= 1)
test_data = test_data.drop(['Name'], axis= 1)

train_data.info()
train_data.describe()



from sklearn.model_selection import train_test_split

cols_to_use = ['Title','Pclass','Sex','Age','Fare','Embarked','FamType']
x = train_data[cols_to_use]
y = train_data.Survived

x_train, x_valid, y_train, y_valid = train_test_split(x, y)

x_test = test_data[cols_to_use]



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

params = {
    'colsample_bytree': 0.6,
    'gamma': 0,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_child_weight': 1,
    'n_estimators': 100,
    'subsample': 0.6
}

model = XGBClassifier(**params)

model.fit(x_train, y_train)

preds = model.predict(x_valid)

accuracy = accuracy_score(y_valid, preds)
print('Accuracy:', accuracy)


preds_test = model.predict(x_test) 
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': (preds_test >= 0.5).astype(int)})
output.to_csv('submission1.csv', index=False)
output.head()


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')

print("Accuracy scores:", scores)