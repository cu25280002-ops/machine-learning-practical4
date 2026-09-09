import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_unscaled = kmeans1.fit_predict(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_scaled = kmeans2.fit_predict(X_scaled)
plt.subplot(1, 2, 1)
plt.scatter(X['Annual_Income_k'], X['Spending_Score'], c=labels_unscaled)
plt.title('Before Scaling')
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled)
plt.title('After Scaling')
plt.show()