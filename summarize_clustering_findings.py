import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
agg_labels = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)
kmeans_score = silhouette_score(X, kmeans_labels)
agg_score = silhouette_score(X, agg_labels)
summary = pd.DataFrame({
    'Algorithm': ['K-Means Clustering', 'Hierarchical (Ward)'],
    'Silhouette Score': [kmeans_score, agg_score]
})
print("--- Clustering Performance Summary ---")
print(summary)