import pandas as pd

merged_df = pd.read_csv("train_and_users.csv")

domain_replace = {'@gmail.com' : '0', '@hotmail.com' : '1', '@yahoo.com' : '2', '@icloud.com' : '3', '@outlook.com' : '4',
 '@ymail.com' : '5', '@rocketmail.com' : '6', '@live.com' : '7', '@qq.com' : '8', '@163.com' : '9', 'other' : '10'}

merged_df['domain'] = merged_df['domain'].map(domain_replace)

merged_df.to_csv('train_and_users_date_encoded.csv', index = False)