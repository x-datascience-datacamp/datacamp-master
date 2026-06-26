
clf = make_pipeline(
    make_column_transformer(
        ('passthrough', ['magnitude_b', 'magnitude_r', 'sigma_flux_b', 'sigma_flux_r', 'log_p_not_variable']),
        (FunctionTransformer(lambda X: X.apply(resample, axis=1)), ['time_points_b', 'light_points_b', 'period'])
    ),
    RandomForestClassifier()
).fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
print("Balanced accuracy:", balanced_accuracy_score(y_test, clf.predict(X_test)))
