import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score', 'Age']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)
cluster_stats = df.groupby('Cluster')[['Annual_Income_k', 'Spending_Score', 'Age']].mean()
print("Cluster Profiles (Mean Values):")
print(cluster_stats)