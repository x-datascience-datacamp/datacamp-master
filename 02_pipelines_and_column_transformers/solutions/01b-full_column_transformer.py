
from sklearn.preprocessing import OrdinalEncoder

cat_cols = ['sex', 'embarked', 'pclass']
num_cols = ['pclass', 'age', 'parch', 'fare']

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
    ("ordinal_encoder", OrdinalEncoder())
])

preprocessor = ColumnTransformer([
    ("categorical_preproc", cat_pipeline, cat_cols),
    ("numerical_preproc", SimpleImputer(), num_cols)

])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(max_depth=10, n_estimators=500))
])
model.fit(X_train, y_train)
model.score(X_test, y_test)
