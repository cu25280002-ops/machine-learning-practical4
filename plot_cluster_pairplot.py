import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv('mall_customers.csv')
features = ['Annual_Income_k', 'Spending_Score', 'Age']
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[features])
sns.pairplot(df, vars=features, hue='Cluster', palette='Set1')
plt.show()