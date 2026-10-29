titanic_models = {
    "Mean imputation + LR": titanic_model(
        make_pipeline(SimpleImputer(), StandardScaler()), LogisticRegression()
    ),
    "Mean imputation + mask + LR": titanic_model(
        make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()),
        LogisticRegression(),
    ),
    "Iterative imputation + LR": titanic_model(
        make_pipeline(IterativeImputer(), StandardScaler()), LogisticRegression()
    ),
    "HGB (native)": titanic_model("passthrough", HistGradientBoostingClassifier()),
}
pd.Series({
    name: cross_val_score(model, X_titanic, y_titanic, cv=cv, scoring="roc_auc").mean()
    for name, model in titanic_models.items()
}).round(3)

# All strategies are within about 0.01 of each other: on real data, the choice of
# imputation often matters little. Missing ages are mostly in third class, and
# pclass is always observed: given the class, age is close to MAR, so the mask
# adds little. The informative missingness of the cabin is already kept by the
# "missing" category of the encoder.
