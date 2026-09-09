import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
print("Cluster assignments for K=2:")
print(labels)