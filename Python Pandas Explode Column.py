import pandas as pd

df = pd.DataFrame({  'card': ['1971 Nolan Ryan', '1928 Ogden Don Bradman', '1909 T206 Ty Cobb', '1887 Lone Jack Ben Franklin', '2005 Topps Justin Verlander'], 
    'properties': [['Baseball', 'Vintage', 'Pitcher'], ['Cricket', 'Pre War'], ['Baseball', 'Pre War', 'Batter'], ['Non Sports', 'Pre War'], ['Baseball', 'Modern', 'Pitcher']]            
                  })
df = df.explode('properties') # properties exploded in diff columns

df['properties'].value_counts()['Baseball'] # counting

df.query('properties == "Baseball"') #query

df.pivot_table(index = 'properties', values = 'card',aggfunc = 'count').reset_index().rename(columns = {'card':'card_count'}).sort_values('card_count',ascending = False)
