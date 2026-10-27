
X_ = X.drop(columns=['EDUCATION'])

ridge.fit(X_, y)

# Visualize the coefs
plt.figure(figsize=(6, 4))

coefs = pd.Series(ridge.coef_, index=X_.columns)
coefs.plot(kind='barh')
plt.axvline(0., color='k', linestyle='--', alpha=0.7)

plt.title("Coefficients")
plt.tight_layout()
