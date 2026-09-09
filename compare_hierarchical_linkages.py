import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
linkages = ['ward', 'complete', 'average', 'single']
for i, link in enumerate(linkages, 1):
    model = AgglomerativeClustering(n_clusters=3, linkage=link)
    labels = model.fit_predict(X)
    plt.subplot(2, 2, i)
    plt.scatter(X['Annual_Income_k'], X['Spending_Score'], c=labels)
    plt.title(f"Linkage: {link.capitalize()}")
plt.tight_layout()
plt.show()