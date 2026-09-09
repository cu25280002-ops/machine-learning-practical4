import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mall_customers.csv')
X = df[['Age', 'Annual_Income_k', 'Spending_Score', 'Savings_Score']]
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
print("Original Feature Matrix Shape:", X.shape)
print("PCA Reduced Matrix Shape:", X_pca.shape)