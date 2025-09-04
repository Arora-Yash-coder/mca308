import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame(np.random.randint(1,201,(100,30)))

df = df.mask(df.between(10,60))
print("NA count per row:\n", df.isna().sum(axis=1))
print("NA count per column:\n", df.isna().sum())

df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

plt.figure(figsize=(12,8))
sns.heatmap(df, cmap="viridis")
plt.show()

corr_matrix = df.corr()
low_corr_cols = (corr_matrix.abs() <= 0.7).sum().sum()
print("Number of column pairs with correlation <= 0.7:", low_corr_cols)

df_norm = (df - df.min()) / (df.max() - df.min()) * 10

df_bin = df_norm.applymap(lambda x: 0 if x <= 5 else 1)

plt.figure(figsize=(10,6))
df_norm.stack().hist(bins=50)
plt.title("Distribution of Normalized Dataset")
plt.show()
