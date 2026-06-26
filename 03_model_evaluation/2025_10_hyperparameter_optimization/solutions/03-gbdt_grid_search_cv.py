
from sklearn.ensemble import HistGradientBoostingRegressor

learning_rate = [0.01, 0.05, 0.1]
max_iter = [10, 50, 100, 200]
param_grid = {
    'learning_rate': learning_rate,
    'max_iter': max_iter
}

est = HistGradientBoostingRegressor(random_state=42)
grid = GridSearchCV(est, param_grid=param_grid, cv=3)

grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))

pd.DataFrame(grid.cv_results_)[
    ['param_learning_rate', 'param_max_iter', 'mean_test_score']
].sort_values(by='mean_test_score', ascending=False)
