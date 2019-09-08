# https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7

import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
from matplotlib import style

start = datetime.datetime(2010, 1, 1)
# end = datetime.datetime(2019, 1, 11)
end = datetime.datetime(2019, 9, 1)

#AA: Read Apple
print("Reading Core Apple Accounting info")
df = web.DataReader("AAPL", 'yahoo', start, end)
'''
This piece of code will pull 7 years data from January 2010 until January 2017. 
Feel free to tweak the start and end date as you see necessary. For the rest of 
analysis, we will use the Closing Price which remarks the final price in which 
the stocks are traded by the end of the day.
'''
df.tail()

# In this analysis, we analyse stocks using two key measurements: Rolling Mean and Return Rate.

# Rolling Mean (Moving Average) — to determine trend

# Rolling mean/Moving Average (MA) smooths out price data by creating a
# constantly updated average price. This is useful to cut down “noise”
# in our price chart. Furthermore, this Moving Average could act as
# “Resistance” meaning from the downtrend and uptrend of stocks you
# could expect it will follow the trend and less likely to deviate
# outside its resistance point

# Chose the last column
close_px = df['Adj Close']
# Calculate the moving average 
mavg = close_px.rolling(window=100).mean()

# This will calculate the Moving Average for the last 100 windows (100 days) of
# stocks closing price and take the average for each of the window’s moving
# average. As you could see, The Moving Average steadily rises over the window
# and does not follow the jagged line of stocks price chart.

# For better understanding, let’s plot it out with Matplotlib. We will overlay
# the Moving Average with our Stocks Price Chart.

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()
plt.show()

# The Moving Average makes the line smooth and showcase the increasing or
# decreasing trend of stocks price.

# In this chart, the Moving Average showcases increasing trend the upturn or
# downturn of stocks price. Logically, you should buy when the stocks are
# experiencing downturn and sell when the stocks are experiencing upturn.

rets = close_px / close_px.shift(1) - 1
# rets.plot(label='return')
# rets.show()
plt.plot(label='return')
plt.show()

# Analysing your Competitors Stocks

# In this segment, we are going to analyse on how one company performs in
# relative with its competitor. Let’s assume we are interested in technology
# companies and want to compare the big guns: Apple, GE, Google, IBM, and
# Microsoft.

print("Reading Other company Accounting info:   Apple,  Vodafone, Google,  Airbus(Germany),  Ford ")
# dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
dfcomp = web.DataReader(['AAPL', 'VOD', 'GOOG', 'AIR.BE', 'F'],'yahoo',start=start,end=end)['Adj Close']
dfcomp.tail()

# Symbols           AAPL        VOD         GOOG      AIR.BE         F
# Date
# 2010-01-04   26.681330  13.122652   312.204773         NaN  7.339305
# 2010-01-05   26.727465  13.026451   310.829926         NaN  7.824781
# 2010-01-06   26.302330  12.924597   302.994293         NaN  8.117498
# 2010-01-07   26.253704  12.681266   295.940735    7.163323  8.324537
# 2010-01-08   26.428249  12.488871   299.885956    7.150816  8.345957
# ...                ...        ...          ...         ...       ...
# 2019-08-27  204.160004  18.459999  1167.839966  122.660004  8.760000
# 2019-08-28  205.529999  18.680000  1171.020020  121.480003  9.000000
# 2019-08-29  209.009995  18.920000  1192.849976  125.320000  9.120000
# 2019-08-30  208.740005  18.820000  1188.099976  125.620003  9.170000
# 2019-09-02         NaN        NaN          NaN  126.220001       NaN

# Correlation Analysis — Does one competitor affect others?

# We can analyse the competition by running the percentage change and
# correlation function in pandas. Percentage change will find how much the price
# changes compared to the previous day which defines returns. Knowing the
# correlation will help us see whether the returns are affected by other stocks’
# returns

# Let’s plot Apple and GE with ScatterPlot to view their return distributions.

# AA Shows the immediate change fro mthe previous value and the current
print ("Percentage changing the various rows of data points")
retscomp = dfcomp.pct_change()

corr = retscomp.corr()

print ("print a corelation matrix between AAPL and GOOG")

plt.scatter(retscomp.AAPL, retscomp.GOOG)
# plt.xlabel(‘Returns AAPL’)
# plt.ylabel(‘Returns GOOG’)
plt.show()

# We can see here that there are slight positive correlations among GE returns
# and Apple returns. It seems like that the higher the Apple returns, the higher
# GE returns as well for most cases.

# Let us further improve our analysis by plotting the scatter_matrix to
# visualize possible correlations among competing stocks. At the diagonal point,
# we will run Kernel Density Estimate (KDE). KDE is a fundamental data smoothing
# problem where inferences about the population are made, based on a finite data
# sample. It helps generate estimations of the overall distributions.

from pandas.plotting import scatter_matrix
scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));
plt.show()


# From here we could see most of the distributions among stocks which
# approximately positive correlations.

# To prove the positive correlations, we will use heat maps to visualize the
# correlation ranges among the competing stocks. Notice that the lighter the
# color, the more correlated the two stocks are.

plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);
plt.show()

# From the Scatter Matrix and Heatmap, we can find great correlations among the
# competing stocks. However, this might not show causality, and could just show
# the trend in the technology industry rather than show how competing stocks
# affect each other. Stocks Returns Rate and Risk

# Apart from correlation, we also analyse each stock’s risks and returns. In
# this case we are extracting the average of returns (Return Rate) and the
# standard deviation of returns (Risk).

plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()


# Now you could view this neat chart of risk and return comparisons for
# competing stocks. Logically, you would like to minimize the risk and maximize
# returns. Therefore, you would want to draw the line for your risk-return
# tolerance (The red line). You would then create the rules to buy those stocks
# under the red line (MSFT, GE, and IBM) and sell those stocks above the red
# line (AAPL and GOOG). This red line showcases your expected value threshold
# and your baseline for buy/sell decision.

# Predicting Stocks Price Feature Engineering

# We will use these three machine learning models to predict our stocks:
# -   Simple Linear Analysis, 
# -   Quadratic Discriminant Analysis (QDA), and 
# -   K Nearest Neighbor (KNN). 
# But first, let us engineer some features: High Low
# Percentage and Percentage Change.

# AA:  Select All tje rows and these 2 columns
dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

# Pre-processing & Cross Validation

# We will clean up and process the data using the following steps before putting them into the prediction models:

#     Drop missing value
#     Separating the label here, we want to predict the AdjClose
#     Scale the X so that everyone can have the same distribution for linear regression
#     Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
#     Separate label and identify it as y
#     Separation of training and testing of model by cross validation train test split

# Please refer the preparation codes below.

# Drop missing value
# AA:  Fill all NA or NAN to be -99999
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
import math
# AA: len of dfreg is 1769,  1 percent of that is 17.69 and the ceiling of that is 18, therefore 18 rows 
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
# AA: Add a new label, fill with  'Adj Col'  then shift up 18 rows backwards 
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
# AA: Convert the dfreg data to an array of 1 dimension with the first including all dfreg data
import numpy as np
X = np.array(dfreg.drop(["label"], 1))

# Scale the X so that everyone can have the same distribution for linear regression
from sklearn.preprocessing import scale
X = scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]


# ===================================
# Model Generation — Where the prediction fun starts

# But first, let’s insert the following imports for our Scikit-Learn:

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Simple Linear Analysis & Quadratic Discriminant Analysis

# Simple Linear Analysis shows a linear relationship between two or more
# variables. When we draw this relationship within two variables, we get a
# straight line. Quadratic Discriminant Analysis would be similar to Simple
# Linear Analysis, except that the model allowed polynomial (e.g: x squared) and
# would produce curves.

# Linear Regression predicts dependent variables (y) as the outputs given
# independent variables (x) as the inputs. During the plotting, this will give
# us a straight line as shown below:

# We will plug and play the existing Scikit-Learn library and train the model by
# selecting our X and y train sets. The code will be as following.

# AA:  CL = Classifier ?

# 1.  Linear regression
clfreg = LinearRegression(n_jobs=-1)
# clfreg.fit(X_train, y_train)
clfreg.fit(X, y)

# 2. Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
# clfreg.fit(X_train, y_train)
clfpoly2.fit(X, y)

# 3. Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
# clfreg.fit(X_train, y_train)
clfpoly3.fit(X, y)

# K Nearest Neighbor (KNN)

# This KNN uses feature similarity to predict values of data points. This
# ensures that the new point assigned is similar to the points in the data set.
# To find out similarity, we will extract the points to release the minimum
# distance (e.g: Euclidean Distance).

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
# clfknn.fit(X_train, y_train)
clfknn.fit(X, y)

# Evaluation

# A simple quick and dirty way to evaluate is to use the score method in each
# trained model. The score method finds the mean accuracy of self.predict(X)
# with y of the test data set.

# AA:  

confidencereg = clfreg.score(X, y)
confidencepoly2 = clfpoly2.score(X,y)
confidencepoly3 = clfpoly3.score(X,y)
confidenceknn = clfknn.score(X, y)

print(f"Confidence score for LinearRegression : ", confidencereg)
print(f"Confidence score for Quadratic Poly 1: ", confidencepoly2)
print(f"Confidence score for Quadratic Poly 2: ", confidencepoly3)
print(f"Confidence score for KNN : ", confidenceknn)

#Wait
input("press enter to continue")

# AA:  Output 
# >>> print(confidencereg, confidencepoly2, confidencepoly3, confidenceknn)
# 0.9667070401650772 0.96809905731067 0.9691528784934818 0.9809562879810294

# This shows an enormous accuracy score (>0.95) for most of the models. However
# this does not mean we can blindly place our stocks. There are still many
# issues to consider, especially with different companies that have different
# price trajectories over time.

################## 1. PREDICT and PLOT using Linear regression   ######################
print ("PRedict using the test data and Linear regression model")
# For sanity testing, let us print some of the stocks forecast.

forecast_set1 = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan

# #result
# (array([ 115.44941187,  115.20206522,  116.78688393,  116.70244946,
#         116.58503739,  115.98769407,  116.54315699,  117.40012338,
#         117.21473053,  116.57244657,  116.048717  ,  116.26444966,
#         115.78374093,  116.50647805,  117.92064806,  118.75581186,
#         118.82688731,  119.51873699]), 0.96234891774075604, 18)

# Based on the forecast, we will visualize the plot with our existing historical
# data. This will help us visualize how the model fares to predict future stocks
# pricing.

# Plotting the Prediction using 
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

# DEBUG print(f"First Run setup:  last date is {last_date} ,  last_unix is {last_unix} , next_unix is {next_unix}. ")

for i in forecast_set1:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg["Adj Close"].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.title('Predictiing with Linear Regression')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# As we can see the blue color showcased the forecast on the stocks price based
# on regression. The forecast predicted that there would be a downturn for not
# too long, then it will recover. Therefore, we could buy the stocks during
# downturn and sell during upturn.

################## 2. PREDICT and PLOT using Quadratic Regression 2   ######################
print ("PRedict using the test data and Quadratic model 1")

# For sanity testing, let us print some of the stocks forecast.

forecast_set2 = clfpoly2.predict(X_lately)
dfreg['Forecast'] = np.nan


# Based on the forecast, we will visualize the plot with our existing historical
# data. This will help us visualize how the model fares to predict future stocks
# pricing.

# Plotting the Prediction using 

# last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

# DEBUG print(f"Second Run setup:  last date is {last_date} ,  last_unix is {last_unix} , next_unix is {next_unix}. ")

for i in forecast_set2:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg["Adj Close"].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.title('Predictiing with Quadratic Equations 1')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# As we can see the blue color showcased the forecast on the stocks price based
# on regression. The forecast predicted that there would be a downturn for not
# too long, then it will recover. Therefore, we could buy the stocks during
# downturn and sell during upturn.

################## 3. PREDICT and PLOT using Quadratic Regression 3   ######################
print ("PRedict using the test data and Quadratic model 2")

# For sanity testing, let us print some of the stocks forecast.

forecast_set3 = clfpoly3.predict(X_lately)
dfreg['Forecast'] = np.nan

# Based on the forecast, we will visualize the plot with our existing historical
# data. This will help us visualize how the model fares to predict future stocks
# pricing.

# Plotting the Prediction using 

# last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

# DEBUG print(f"Third Run setup:  last date is {last_date} ,  last_unix is {last_unix} , next_unix is {next_unix}. ")

for i in forecast_set3:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg["Adj Close"].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.title('Predictiing with Quadratic Equations 2')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# As we can see the blue color showcased the forecast on the stocks price based
# on regression. The forecast predicted that there would be a downturn for not
# too long, then it will recover. Therefore, we could buy the stocks during
# downturn and sell during upturn.

################## 4. PREDICT and PLOT using KKN Nearest Neighbour   ######################
print ("PRedict using the test data and K NEarest Model ")

# For sanity testing, let us print some of the stocks forecast.

forecast_set4 = clfknn.predict(X_lately)
dfreg['Forecast'] = np.nan

# Based on the forecast, we will visualize the plot with our existing historical
# data. This will help us visualize how the model fares to predict future stocks
# pricing.

# Plotting the Prediction using 

# last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set4:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg["Adj Close"].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.title('Predictiing with K Nearest Neighbour')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# As we can see the blue color showcased the forecast on the stocks price based
# on regression. The forecast predicted that there would be a downturn for not
# too long, then it will recover. Therefore, we could buy the stocks during
# downturn and sell during upturn.


print ("FINISHED")
