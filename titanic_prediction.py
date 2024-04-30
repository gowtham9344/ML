
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
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

from sklearn.model_selection import train_test_split

cols_to_use = ['Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Cabin','Embarked']
x = train_data[cols_to_use]
y = train_data.Survived

x_train, x_valid, y_train, y_valid = train_test_split(x, y)

cols_to_use.append('PassengerId')
x_test = test_data[cols_to_use]

numerical_cols = [cname for cname in x_train.columns if x_train[cname].dtype in ['int64', 'float64']]
print(numerical_cols)

categorical_cols = [cname for cname in x_train.columns if x_train[cname].nunique() < 10 and
                        x_train[cname].dtype == "object"]
print(categorical_cols)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = GaussianNB()

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(x_train, y_train)


preds = my_pipeline.predict(x_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

preds_test = my_pipeline.predict(x_test)
output = pd.DataFrame({'PassengerId': x_test.PassengerId,
                       'Survived': (preds_test >= 0.5).astype(int)})
output.to_csv('submission1.csv', index=False)
output.head()

