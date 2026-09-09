import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mall_customers.csv')
X = df[['Annual_Income_k', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])