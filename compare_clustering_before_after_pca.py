import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv('mall_customers.csv')
X = df[['Age', 'Annual_Income_k', 'Spending_Score', 'Savings_Score']]
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
labels_orig = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
labels_pca = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca)
print("Silhouette Score (Original Scaled Data):", silhouette_score(X_scaled, labels_orig))
print("Silhouette Score (PCA Reduced Data):     ", silhouette_score(X_pca, labels_pca))