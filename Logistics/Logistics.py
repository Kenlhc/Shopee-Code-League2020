import pandas as pd 
import numpy as np 
import datetime 
import math 
from datetime import timedelta, date

exclude_days = [date(2020, 3, 1), date(2020,3,8), date(2020,3,15), date(2020,3,22), date(2020,3,25), date(2020,3,29), date(2020,3,30), date(2020,3,31)]

deliveries_df = pd.read_csv("delivery_orders_march.csv")

deliveries_df.replace(np.nan,0)

deliveries_list = deliveries_df.values.tolist()
 
final_ouput = [] 

array_length = len(deliveries_list)

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

for rows in range(0, array_length):
	count = 0 
	working_days = 0
	if deliveries_list[rows][4].lower().find("metro manila") != -1 and deliveries_list[rows][5].lower().find("metro manila") != -1:
		working_days = 3 
		deliveries_list[rows][1] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][1])))
		pick = deliveries_list[rows][1].date()
		deliveries_list[rows][2] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][2])))
		attempt1 = deliveries_list[rows][2].date()
		for dt in daterange(pick+timedelta(days=1), attempt1):
			if dt in exclude_days:
				continue
			count += 1
		if count <= working_days and math.isnan(deliveries_list[rows][3]):
			final_ouput.append([deliveries_list[rows][0], 0])
		elif count > working_days:
			final_ouput.append([deliveries_list[rows][0], 1])
		else: 
			count = 0
			deliveries_list[rows][3] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][3])))
			attempt2 = deliveries_list[rows][3].date()
			for dt in daterange(attempt1+timedelta(days=1), attempt2):
				if dt in exclude_days:
					continue
				count += 1
			if count <= 3: 
				final_ouput.append([deliveries_list[rows][0], 0])
			else:
				final_ouput.append([deliveries_list[rows][0], 1])
	elif deliveries_list[rows][4].lower().find("luzon") != -1 and deliveries_list[rows][5].lower().find("luzon") != -1:
		working_days = 5 
		deliveries_list[rows][1] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][1])))
		pick = deliveries_list[rows][1].date()
		deliveries_list[rows][2] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][2])))
		attempt1 = deliveries_list[rows][2].date()
		for dt in daterange(pick+timedelta(days=1), attempt1):
			if dt in exclude_days:
				continue
			count += 1
		if count <= working_days and math.isnan(deliveries_list[rows][3]):
			final_ouput.append([deliveries_list[rows][0], 0])
		elif count > working_days:
			final_ouput.append([deliveries_list[rows][0], 1])
		else: 
			count = 0 
			deliveries_list[rows][3] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][3])))
			attempt2 = deliveries_list[rows][3].date()
			for dt in daterange(attempt1+timedelta(days=1), attempt2):
				if dt in exclude_days:
					continue
				count += 1
			if count <= 3: 
				final_ouput.append([deliveries_list[rows][0], 0])
			else:
				final_ouput.append([deliveries_list[rows][0], 1])
	elif deliveries_list[rows][4].lower().find("luzon") != -1 and deliveries_list[rows][5].lower().find("metro manila") != -1:
		working_days = 5 
		deliveries_list[rows][1] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][1])))
		pick = deliveries_list[rows][1].date()
		deliveries_list[rows][2] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][2])))
		attempt1 = deliveries_list[rows][2].date()
		for dt in daterange(pick+timedelta(days=1), attempt1):
			if dt in exclude_days:
				continue
			count += 1
		if count <= working_days and math.isnan(deliveries_list[rows][3]):
			final_ouput.append([deliveries_list[rows][0], 0])
		elif count > working_days:
			final_ouput.append([deliveries_list[rows][0], 1])
		else: 
			count = 0
			deliveries_list[rows][3] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][3])))
			attempt2 = deliveries_list[rows][3].date()
			for dt in daterange(attempt1+timedelta(days=1), attempt2):
				if dt in exclude_days:
					continue
				count += 1
			if count <= 3: 
				final_ouput.append([deliveries_list[rows][0], 0])
			else:
				final_ouput.append([deliveries_list[rows][0], 1])
	elif deliveries_list[rows][4].lower().find("metro manila") != -1 and deliveries_list[rows][5].lower().find("luzon") != -1:
		working_days = 5 
		deliveries_list[rows][1] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][1])))
		pick = deliveries_list[rows][1].date()
		deliveries_list[rows][2] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][2])))
		attempt1 = deliveries_list[rows][2].date()
		for dt in daterange(pick+timedelta(days=1), attempt1):
			if dt in exclude_days:
				continue
			count += 1
		if count <= working_days and math.isnan(deliveries_list[rows][3]):
			final_ouput.append([deliveries_list[rows][0], 0])
		elif count > working_days:
			final_ouput.append([deliveries_list[rows][0], 1])
		else: 
			count = 0 
			deliveries_list[rows][3] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][3])))
			attempt2 = deliveries_list[rows][3].date()
			for dt in daterange(attempt1+timedelta(days=1), attempt2):
				if dt in exclude_days:
					continue
				count += 1
			if count <= 3: 
				final_ouput.append([deliveries_list[rows][0], 0])
			else:
				final_ouput.append([deliveries_list[rows][0], 1])
	else: 
		working_days = 7  
		deliveries_list[rows][1] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][1])))
		pick = deliveries_list[rows][1].date()
		deliveries_list[rows][2] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][2])))
		attempt1 = deliveries_list[rows][2].date()
		for dt in daterange(pick+timedelta(days=1), attempt1):
			if dt in exclude_days:
				continue
			count += 1
		if count <= working_days and math.isnan(deliveries_list[rows][3]):
			final_ouput.append([deliveries_list[rows][0], 0])
		elif count > working_days:
			final_ouput.append([deliveries_list[rows][0], 1])
		else: 
			count = 0 
			deliveries_list[rows][3] = datetime.datetime.fromtimestamp(int(float(deliveries_list[rows][3])))
			attempt2 = deliveries_list[rows][3].date()
			for dt in daterange(attempt1+timedelta(days=1), attempt2):
				if dt in exclude_days:
					continue
				count += 1
			if count <= 3: 
				final_ouput.append([deliveries_list[rows][0], 0])
			else:
				final_ouput.append([deliveries_list[rows][0], 1])    	 

final_df = pd.DataFrame(final_ouput, columns=['orderid', 'is_late'])
final_df.to_csv('logistics2.csv', index = False)
#Find buyer and seller destination, assign working days 
#Calculate working days from pick up to 1st attempt
#check for exclude days 
#if counter <= working days && 2nd attempt == nan, assign 0 
#if 2nd attempt != NaT, check if within 3 working days