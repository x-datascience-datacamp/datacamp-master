
from ipywidgets import interact

@interact(max_depth=(1, 10))
def plot_tree(max_depth=1):
    reg = DecisionTreeRegressor(max_depth=max_depth)
    reg.fit(X, y)
    X_test = np.linspace(-3, 3, 1000).reshape((-1, 1))
    y_test = reg.predict(X_test)

    plt.figure()
    plt.plot(X_test.ravel(), y_test, color='tab:blue', label="prediction")
    plt.plot(X.ravel(), y, 'C7.', label="training data")
    _ = plt.legend(loc="best")
