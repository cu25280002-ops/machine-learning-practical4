import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)
plt.scatter(X['Annual_Income_k'], X['Spending_Score'], c=labels, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Hierarchical Clustering Results')
plt.show()