import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title('Hierarchical Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()