import pandas as pd

df =  pd.DataFrame({'num1':['123','534','234','3','34'],'num2':['123','234','345','456','34']})

df.shift(-1,axis = 1,fill_value = 0)
