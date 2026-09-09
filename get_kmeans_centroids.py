import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
print("Cluster Centroids (Income, Spending Score):")
print(kmeans.cluster_centers_)