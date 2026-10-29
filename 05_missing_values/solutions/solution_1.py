X, _ = generate_data(10_000)
rows = []
for mechanism in ["MCAR", "MAR", "MNAR"]:
    X_missing = add_missing_values(X, mechanism)
    rows.append({
        "mechanism": mechanism,
        "complete case": np.nanmean(X_missing[:, 0]),
        "mean imputation": SimpleImputer().fit_transform(X_missing)[:, 0].mean(),
        "iterative imputation": IterativeImputer().fit_transform(X_missing)[:, 0].mean(),
    })
print(f"True mean of x1: {X[:, 0].mean():.3f}")
pd.DataFrame(rows).set_index("mechanism").round(3)

# - MCAR: all three estimates are unbiased.
# - Mean imputation gives exactly the complete-case mean: it adds no information.
# - MAR: complete case is biased, but the iterative imputer uses x2 to correct it.
# - MNAR: all three are biased. The observed data does not say how the missing
#   values differ from the observed ones.
