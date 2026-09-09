import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv('mall_customers.csv')
X = df[['Age', 'Annual_Income_k', 'Spending_Score', 'Savings_Score']]
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='magma')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means Clustering on PCA Transformed Data')
plt.show()