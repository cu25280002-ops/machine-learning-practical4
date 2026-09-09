import pandas as pd
from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)
print("Agglomerative Cluster Labels:")
print(labels)