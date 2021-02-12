from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from scipy.stats import randint, reciprocal, uniform


model = Pipeline([
    ('preprocessing', preprocessing),
    ('clf', HistGradientBoostingClassifier(n_jobs=-1, random_state=42))
])

param_distributions = {
    'clf__learning_rate': reciprocal(1e-3, 0.5),
    'clf__l2_regularization': uniform(0, 0.5),
    'clf__max_leaf_nodes': randint(5, 30),
    'clf__min_samples_leaf': randint(5, 30),
}
search = RandomizedSearchCV(
    model, param_distributions=param_distributions,
    n_iter=20, n_jobs=-1, cv=5, random_state=42
)
