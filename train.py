import pandas as pd
import joblib
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
DATA_PATH = "iris.data"
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
COLUMN_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'iris_model.joblib')
print("Model trained and saved as 'iris_model.joblib")