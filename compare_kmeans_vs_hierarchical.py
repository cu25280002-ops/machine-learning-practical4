import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
agg_labels = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)
plt.subplot(1, 2, 1)
plt.scatter(X['Annual_Income_k'], X['Spending_Score'], c=kmeans_labels)
plt.title('K-Means Clustering')
plt.subplot(1, 2, 2)
plt.scatter(X['Annual_Income_k'], X['Spending_Score'], c=agg_labels)
plt.title('Hierarchical Clustering')
plt.show()