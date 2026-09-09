import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mall_customers.csv')
X = df[['Age', 'Annual_Income_k', 'Spending_Score', 'Savings_Score']]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA().fit(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
print("Cumulative Explained Variance:")
for i, val in enumerate(cum_var, 1):
    print(f"{i} Component(s): {val:.4f}")