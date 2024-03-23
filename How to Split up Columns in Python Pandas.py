import pandas as pd

df = pd.DataFrame({'country':['1971 Mymensingh Bangladesh','1945 Dillih India','1990 Islamabad Pakistan','1923 Barma Marma']})

df['country'].str.split(' ') # normal split with comas
df['country'].str.split(' ',n=1,expand = True) # split in n colums
df['country'].str.split(' ',n=1,expand = True).rename(columns = {0:'Year', 1:'Place'}) # split in n colums and give name

df[['Year','Place']]=df['country'].str.split(' ',n=1,expand = True) # rename in data frame

df = df.drop(columns =['country']) # Delate previous colums 

df.head()



df2 = pd.DataFrame({'Players':['Mike Trouct','Mookie Bitter','Ted Willieam'],'Slashline':['.301/.412/.523','.123/.345/.567','.234/.456/.567']})

df2[['AVG','OBP','SLG']] = df2['Slashline'].str.split('/',expand = True)
df2 = df2.drop(columns = ['Slashline'])
df2.head()
