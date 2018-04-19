##################### Pandas #####################

import numpy as np
import pandas as pd
y = np.arange(10,101,10)
py = pd.Series(y)
py

a = [1,2,3,4,5,6]
b = [2,4,6,8,10,12]
df = pd.DataFrame([a, b])
df

# dictionary and Series
d = {0: 'A', 1: 'B', 2: 'C'}
s = pd.Series(d)
s

# Dictionary and DataFrame
dict = {'Normal': ['A', 'B', 'C'], 'Reverse': ['Z', 'Y', 'X']}
df = pd.DataFrame(dict)
df

##################### csv #####################
# Read CSV file
df = pd.read_csv("data.csv",sep=',')

#Write a DataFrame into a CSV filePermalink
df.to_csv("data-out.csv", index=False)

import time
output_file = 'trip_summary_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
df.to_csv(output_file, index=False)

usage = pd.read_csv("F:\\Python\\usage.csv", index_col=0)
usage
usage['Reading']


# Subset of columns
df1 = df[['a','b']]

# transform column
df['Year'] = df['Year'].map(lambda y: str(y))

# select rows based on column value (subset of rows)
mask = df['Age'] >= 50
df = df[mask]

# plot: show percentage on the y axix
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])


dates = pd.date_range("20160101", periods=6)
data = np.random.random((6,3))
column_names = ['Column1', 'Column2', 'Column3']
df = pd.DataFrame(data, index=dates, columns=column_names)
df
df['Column2'] # Indexing a column: use the column name's string 
df[0:2] # Indexing rows: use the standard indexing technique
df['20160101':'20160102'] # Indexing rows: use the index's strings

#Indexing multiple axes — names
df.loc['20160101':'20160102',['Column1','Column3']]

#Indexing multiple axes — numbers
df.iloc[3:5, 0:2]

df.head(2) # first 2 rows
df.tail(2) # last 2 rows
df.describe() # View summary statistics

# Join
dates1 = pd.date_range("20160101", periods=6)
data1 = np.random.random((6,2))
column_names1 = ['ColumnA', 'ColumnB']
dates2 = pd.date_range("20160101", periods=7)
data2 = np.random.random((7,2))
column_names2 = ['ColumnC', 'ColumnD']
df1 = pd.DataFrame(data1, index=dates1, columns=column_names1)
df2 = pd.DataFrame(data2, index=dates2, columns=column_names2)
df1.join(df2) # joins on the index

#If you want to join on a column other than the index, check out the merge method.
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
 

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

result = pd.merge(left, right, on='key')


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})


right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

##################### merge #####################
result = pd.merge(left, right, on=['key1', 'key2'])
result = pd.merge(left, right, how='left', on=['key1', 'key2'])
result = pd.merge(left, right, how='right', on=['key1', 'key2'])
result = pd.merge(left, right, how='outer', on=['key1', 'key2'])
result = pd.merge(left, right, how='inner', on=['key1', 'key2'])

##################### Group by #####################
df3 = df1.join(df2)
# add a column to df to group on
df3['ProfitLoss'] = pd.Series(['Profit', 'Loss', 'Profit', 'Profit', 'Profit', 'Loss'], index=dates1)
df3.groupby('ProfitLoss').mean()

# groupby in dataframes: count, sum and mean
(df.groupby('Company Name')
 .agg({'Organisation Name':'count', 'Amount': 'sum', 'Rate': 'mean'})
 .reset_index()
 .rename(columns={'Organisation Name':'Organisation Count'})
)

#Access the Index
df3.index

#Access the Values
df3.values

# Access the Columns
df3.columns
