import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
score = silhouette_score(X, labels)
print("Silhouette Score (K=3):", score)