import pandas as pd

train_df = pd.read_csv("clean_train.csv")

users_df = pd.read_csv("clean_users.csv")

merged_df = train_df.merge(users_df, on='user_id')

merged_df.to_csv('train_and_users.csv', index = False)