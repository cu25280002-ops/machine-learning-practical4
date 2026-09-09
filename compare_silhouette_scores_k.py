import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K = {k} | Silhouette Score: {score:.4f}")