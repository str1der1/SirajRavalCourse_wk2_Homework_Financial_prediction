import pandas as pd
from matplotlib import pyplot as plt 

# print("=========  Ex1   ==============")
# x = [1, 2, 3]
# y = [1, 4, 9]

# plt.plot (x,y)
# plt.show() 

# print("===========  Ex2  ===============")
# sample_data = pd.read_csv('sample_data.csv')
# # print(sample_data)

# #    column_a  column_b  column_c
# # 0         1         1        10
# # 1         2         4         8
# # 2         3         9         6
# # 3         4        16         4
# # 4         5        25         2

# print(type(sample_data))
# # <class 'pandas.core.frame.DataFrame'>

# print(sample_data.column_c)
# # 0    10
# # 1     8
# # 2     6
# # 3     4
# # 4     2
# # Name: column_c, dtype: int64

# print(type(sample_data.column_c))
# # <class 'pandas.core.series.Series'>

# print(sample_data.column_c.iloc[1])
# # 8

# plt.plot(sample_data.column_a, sample_data.column_b, 'o')
# plt.plot(sample_data.column_a, sample_data.column_c)
# plt.legend( )
# plt.show()


print ("===============  Ex3  ================")
cdata = pd.read_csv("countries.csv")
# print (cdata)
# #           country  year  population
# # 0     Afghanistan  1952     8425333
# # 1     Afghanistan  1957     9240934
# # 2     Afghanistan  1962    10267083
# # 3     Afghanistan  1967    11537966
# # 4     Afghanistan  1972    13079460
# # ...           ...   ...         ...
# # 1699     Zimbabwe  1987     9216418
# # 1700     Zimbabwe  1992    10704340
# # 1701     Zimbabwe  1997    11404948
# # 1702     Zimbabwe  2002    11926563
# # 1703     Zimbabwe  2007    12311143

# [1704 rows x 3 columns]

# print (type(cdata))
# # <class 'pandas.core.frame.DataFrame'>

usdata = cdata[cdata.country == 'United States']
# print (usdata)
chdata = cdata[cdata.country == 'China']

# plt.plot(usdata.year, usdata.population)
# Now plot the population in millions
# plt.plot(usdata.year, usdata.population / 10**6)
# plt.plot(chdata.year, chdata.population / 10**6)
# plt.legend(['United States', 'China'])
# plt.xlabel('year')
# plt.ylabel('population')
# plt.show()

# As a percentage growth based on first years as 100%
plt.plot(usdata.year, (usdata.population / usdata.population.iloc[0]) * 100)
plt.plot(chdata.year, (chdata.population / chdata.population.iloc[0]) * 100)
plt.legend(['United States', 'China'])
plt.xlabel('year')
plt.ylabel('population as growth (first year = 100')
plt.show()
