
X_test_num_imputed = X_test_num.fillna(X_train_num.mean())
model.score(X_test_num_imputed, y_test)
