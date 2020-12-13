import pandas as pd 

date_df = pd.read_csv('train_and_users_date_encoded.csv')

dates_list = date_df['grass_date'].values.tolist() 

unique = set() 

for date in dates_list: 
	unique.add(date)

unique_dates = list(unique)

date_with_int = list() 
count = 0 

for date in unique: 
	date_with_int.append(count)
	count += 1

date_dictionary = dict(zip(unique_dates, date_with_int))

date_df['grass_date'] = date_df['grass_date'].map(date_dictionary)

date_df.to_csv('clean_final_train.csv', index = False)