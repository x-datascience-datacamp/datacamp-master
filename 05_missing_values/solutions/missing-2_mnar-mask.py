X_mnar = add_missing_values(X_large, "MNAR")
models_mnar = {
    **models,
    "Mean imputation + mask + Ridge": make_pipeline(
        SimpleImputer(add_indicator=True), RidgeCV()
    ),
    "Iterative imputation + mask + Ridge": make_pipeline(
        IterativeImputer(add_indicator=True), RidgeCV()
    ),
}
compare(models_mnar, X_mnar, y_large, "Linear target, MNAR")

# 1. The iterative imputer predicts x1 from x2 as if the data were MAR. Under MNAR
#    the missing values are the large ones, so its imputations are too small, and
#    so are the predictions. HGB learns from the data that a missing x1 means a
#    large x1, and does better.
# 2. The mask tells the Ridge that the value was missing, so it can learn a shift
#    for these samples. It helps both Ridge pipelines, and iterative imputation
#    + mask + Ridge catches up with HGB.
