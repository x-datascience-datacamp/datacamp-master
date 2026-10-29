X, y = df_bike[["hr", "workingday"]].values, df_bike["cnt"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=1000, test_size=2000, random_state=0
)

reg_icl = TabICLRegressor(n_estimators=4)
reg_icl.fit(X_train, y_train)

x_8am = np.array([[8, 0], [8, 1]])  # 8am on a weekend/holiday, 8am on a working day
q = reg_icl.predict(x_8am, output_type="quantiles", alphas=alphas)
for wd, (lo, med, hi) in zip([0, 1], q):
    print(f"workingday={wd}: median={med:.0f}, 90% interval=[{lo:.0f}, {hi:.0f}]")