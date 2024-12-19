from skrub import tabular_learner
from sklearn.linear_model import Ridge

model = tabular_learner(Ridge(alpha=10))
model.fit(X_train, y_train).score(X_test, y_test)
