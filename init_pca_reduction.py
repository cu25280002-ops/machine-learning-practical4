import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mall_customers.csv')
X = df[['Age', 'Annual_Income_k', 'Spending_Score', 'Savings_Score']]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca[:5])