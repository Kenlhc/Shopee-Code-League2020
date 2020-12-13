import pandas as pd

df1 = pd.read_csv("train.csv")
df1 = df1.drop(df1.columns[0], axis=1)
df2 = pd.read_csv("245_1.csv")

df1 = df1.append(df2)

df1.to_csv('train.csv', index=False)