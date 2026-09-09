import pandas as pd

df = pd.read_csv('mall_customers.csv')
print("Dataset Info")
df.info()
print("\nSummary Statistics")
print(df.describe())