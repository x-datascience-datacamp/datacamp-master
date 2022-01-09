
def information_gain_mse(y, y_left, y_right):
    orig_variance = np.var(y)
    left_variance = 0
    right_variance = 0
    if y_left.size > 0:
        left_variance = np.var(y_left)
    if y_right.size > 0:
        right_variance = np.var(y_right)
    return orig_variance - (left_variance + right_variance)

all_information_gain = [
    information_gain_mse(y, y[:idx], y[idx:]) for idx in range(len(X))
]

best_idx = np.argmax(all_information_gain)

y_left = y[:best_idx]
y_right = y[best_idx:]

y_pred = y.copy()
y_pred[:best_idx] = np.mean(y_left)
y_pred[best_idx:] = np.mean(y_right)

plt.figure()
plt.plot(X.ravel(), all_information_gain, color="red", label="Information gain")
plt.axvline(X[best_idx, 0], color='r', linestyle='--')
plt.plot(X.ravel(), y_pred, color='tab:blue', label="prediction")
plt.plot(X.ravel(), y, 'C7.', label="training data")
_ = plt.legend(loc="best")
