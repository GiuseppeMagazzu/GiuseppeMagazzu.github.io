import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# set random state to obtain reproducible results
random_state = 1
# dataset creation
X, y = make_classification(n_samples=400, n_features=20, n_informative=14, n_redundant=2, n_classes=2, shuffle=True, random_state=random_state)
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=random_state)
