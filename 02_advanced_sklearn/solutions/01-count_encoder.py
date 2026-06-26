
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        n_features = X.shape[1]
        counters = []
        for k in range(n_features):
            counters.append(Counter(X[:, k]))
        self.counters_ = counters
        return self
    
    def transform(self, X):
        X_t = X.copy()
        for x, counter in zip(X_t.T, self.counters_):
            # Uses numpy broadcasting
            idx = np.nonzero(list(counter.keys()) == x[:, None])[1]
            x[:] = np.asarray(list(counter.values()))[idx]
        return X_t

X = np.array([
    [0, 2],
    [1, 3],
    [1, 1],
    [1, 1],
])
ce = CountEncoder()
print(ce.fit_transform(X))

# Let's put this now in a Pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
    ("count_encoder", CountEncoder())
])

categorical_preprocessing = ColumnTransformer([
    ("categorical_preproc", cat_pipeline, cat_col)
])

model = Pipeline([
    ("categorical_preproc", categorical_preprocessing),
    ("classifier", RandomForestClassifier(n_estimators=100))
])
model.fit(X_train, y_train)
model.score(X_test, y_test)
