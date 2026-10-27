sex = X[:, feature_names.index("sex")]
rng_ex = np.random.default_rng(RANDOM_STATE)
B = 9999

# 1. Algorithm of Section 2.1: shuffle y against sex -> null "sex and y are unrelated"
r_obs = abs(np.corrcoef(sex, y)[0, 1])
r_null = np.array([abs(np.corrcoef(sex, rng_ex.permutation(y))[0, 1]) for _ in range(B)])
p_marginal = (1 + np.sum(r_null >= r_obs)) / (1 + B)

# 2. Freedman-Lane: shuffle only what the other nine features leave unexplained
_, _, p_partial = freedman_lane(X, y, "sex", B=B)

print(f"shuffle y      (sex alone):             p = {p_marginal:.4f}")
print(f"Freedman-Lane  (sex given the other 9): p = {p_partial:.4f}")

# 3. Both are right: they test different nulls.
#    - shuffling y:   "sex says nothing about y on its own"            -> not rejected
#    - Freedman-Lane: "sex adds nothing once the other nine are known" -> rejected
#    sex is correlated with several features that predict y (table below). Holding
#    them fixed, sex = 2 lowers progression (coefficient -23). On its own, sex also
#    picks up part of their effect, which goes the other way: the two nearly cancel.
diabetes.data.corr()["sex"].drop("sex").round(2)
