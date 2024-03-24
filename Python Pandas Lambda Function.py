import pandas as pd
import seaborn as sns



sns.get_dataset_names()

df = sns.load_dataset('car_crashes')


df['totalv2'] = df.apply(lambda x: x['total'] * 100 , axis = 1)

df['totalv3'] = df['total'].apply(lambda x: x*100) # x * 100

df['ins_premium_type'] = df['ins_premium'].apply(lambda x: 'High' if x > 1000 else 'Medium' if x > 800 else 'Low') # if else

df['spl'] = df.apply(lambda x: x['speeding'] + x['alcohol'], axis = 1) # new  column

df['abbrev'] = df['abbrev'].apply(lambda x: x.lower()) # upper lower


df['abbrev'] = df['abbrev'].apply(lambda x: x.upper() if x in ['fl','ga','sc','al'] else x) # changing in a same column


df['spl'] = df['spl'].apply(lambda x: x - 3 if x==3.599 else x)

