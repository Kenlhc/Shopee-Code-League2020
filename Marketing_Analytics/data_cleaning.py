import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")

train_df['last_open_day'] = train_df['last_open_day'].replace('Never open', '0')
train_df['last_checkout_day'] = train_df['last_checkout_day'].replace('Never checkout', '0')
train_df['last_login_day'] = train_df['last_login_day'].replace('Never login', '0')

users_df = pd.read_csv("users.csv")

users_df = users_df.replace('', np.nan)
users_df['age'] = users_df['age'].fillna('999')
users_df['attr_1'] = users_df['attr_1'].fillna('2')
users_df['attr_2'] = users_df['attr_2'].fillna('3')
users_df['attr_3'] = users_df['attr_3'].fillna('5')

users_list = users_df.values.tolist() 

users_size = len(users_list)

for i in range(0, users_size):
	users_list[i][4] = str(users_list[i][4])
	if users_list[i][4].find("-") != -1: 
		users_list[i][4] = users_list[i][4].replace("-", "")

clean_df = pd.DataFrame(users_list, columns=['user_id', 'attr_1', 'attr_2', 'attr_3', 'age', 'domain'])
clean_df.to_csv('clean_users.csv', index = False)
train_df.to_csv('clean_train.csv', index = False)