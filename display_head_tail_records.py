import pandas as pd

df = pd.read_csv('mall_customers.csv')
print("First 5 Records")
print(df.head())
print("\n Last 5 Records")
print(df.tail())