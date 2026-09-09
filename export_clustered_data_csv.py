import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Label'] = kmeans.fit_predict(X)
df.to_csv('mall_customers_clustered.csv', index=False)
print("Clustered dataset successfully saved to 'mall_customers_clustered.csv'.")