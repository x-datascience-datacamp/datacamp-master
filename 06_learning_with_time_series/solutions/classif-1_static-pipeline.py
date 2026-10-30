from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier

clf_static = make_pipeline(
    make_column_transformer(
        ('passthrough', ['magnitude_b', 'magnitude_r', 'sigma_flux_b', 'sigma_flux_r', 'log_p_not_variable'])
    ),
    RandomForestClassifier()
).fit(X_train, y_train)
y_pred_static = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_static))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred_static))
