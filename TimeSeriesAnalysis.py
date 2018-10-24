# Import required libraries
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
#from pandas.util.testing import assert_frame_equal
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima_model import ARIMA
#import statsmodels as sma
from pyramid.arima import auto_arima
#from kapteyn import kmpfit
#from kmpfitf import confpred_band


## Print XML header
#with open("dblp_20180311.xml") as myfile:
#    head = [next(myfile) for x in xrange(20)]
#print head


## Open XML files
#yearOldList = []
#with open('dblp_20180311.xml') as myfile:
#    for line in myfile:
#        line = line.lower()
#        if "<year>" in line:
#            start = line.find('<year>') + 6
#            end = line.find('</year>', start)
#            yearString = line[start:end]
#            yearOldList.append(yearString)


## Create XML Dataframes
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y')
#yearOldDf = pd.DataFrame(yearOldList)#, parse_dates=['Year'], index_col='Year',date_parser=dateparse)
#yearOldSdf = yearOldDf.apply(pd.value_counts).sort_index(0)
#yearOldSdf.columns = ['Papers']
#yearOldSdf.to_json('yearOldDf.csv')
#yearOldSdf.to_csv('yearOldXLS.csv')


# Read Saved Dataframes
yearOldSdf = pd.read_json('yearOldDf.csv')
yearNewSdf = pd.read_json('yearNewDf.csv')
yearNu2Sdf = pd.read_json('yearNu2Df.csv')


# DBLP Bar Graph
ax = yearNu2Sdf[['Papers']].plot(kind='bar',figsize=(16,9), color='black')
yearNewSdf[['Papers']].plot(ax=ax,kind='bar', color='darkgray')
yearOldSdf[['Papers']].plot(ax=ax,kind='bar', color='lightgray')
ax.set_xlabel("Year")
ax.set_ylabel("Papers Published")
ax.set_ylim([0,350000])
lgray_patch = mpatches.Patch(color='lightgray', label='DBLP 2018-03-11')
dgray_patch = mpatches.Patch(color='darkgray', label='DBLP 2018-04-26')
black_patch = mpatches.Patch(color='black', label='DBLP 2018-05-15')
plt.legend(handles=[lgray_patch,dgray_patch,black_patch],loc=2)
plt.savefig('dblp1BarGraph.png')
plt.show()


# DBLP Plot - Fit Logarithmic Curve
yearNewW = yearNewSdf.loc[2002:2016]
x = yearNewW.index
y = yearNewW['Papers']
def func(x, a, b, c):
    return a * np.log(x) + b * x + c
popt, pcov = curve_fit(func, x, y)
fig, axo = plt.subplots(figsize=(8,5))
tn = yearNu2Sdf.loc[0:2016].index
sn = yearNu2Sdf.loc[0:2016]['Papers']
axo.plot(tn, sn, 'b.')
axo.set_xlabel('Year')
axo.set_ylabel('Papers Published')
axo.set_xlim([1935,2020])
axo.set_ylim([0,350000])
t2a = np.arange(2000,2003,1)
t2b = np.arange(2002,2017,1)
t2c = np.arange(2016,2020,1)
s2a = func(t2a, *popt)
s2b = func(t2b, *popt)
s2c = func(t2c, *popt)
axo.plot(t2a, s2a, 'r--',t2b, s2b, 'r-',t2c, s2c, 'r--')
gray_patch = mpatches.Patch(color='blue', label='Timeseries')
red_patch = mpatches.Patch(color='red', label='Fitted Log Curve')
plt.legend(handles=[gray_patch,red_patch],loc=2)
plt.savefig('dblp2OldNewLog.png')
plt.show()


# Plot TEST predictions
past = yearNewSdf.loc[0:2002]
train = yearNewSdf.loc[2002:2013]
test = yearNewSdf.loc[2014:2016]
y_hat_avg = test.copy()
plt.figure(figsize=(8,5))
plt.plot(past.index,past['Papers'], label='Past Series',color='blue',linestyle=':')
plt.plot(train.index, train['Papers'], label='Train Series',color='blue',linestyle='-')
plt.plot(test.index,test['Papers'], label='Test Series',color='blue',linestyle='--')

t2c = np.arange(2014,2017,1)
s2c = func(t2c, *popt)
y_hat_avg['log_func'] = s2c
plt.plot(y_hat_avg['log_func'], label='Log Function',color='red',linestyle='-')
rmshl = sqrt(mean_squared_error(test.Papers, s2c))
print('Log function',rmshl)

dd= np.asarray(train.Papers)
y_hat_avg['naive'] = dd[len(dd)-1]
plt.plot(y_hat_avg.index,y_hat_avg['naive'], label='Naive Forecast')
rmsnf = sqrt(mean_squared_error(test.Papers, y_hat_avg.naive))
print("Naive Forecast", rmsnf)

y_hat_avg['average'] = train['Papers'].mean()
plt.plot(y_hat_avg['average'], label='Average Forecast')
rmsaf = sqrt(mean_squared_error(test.Papers, y_hat_avg.average))
print('Average Forecast',rmsaf)

y_hat_avg['mov_average'] = train['Papers'].rolling(3).mean().iloc[-1]
plt.plot(y_hat_avg['mov_average'], label='Moving Average Forecast')
rmsmaf = sqrt(mean_squared_error(test.Papers, y_hat_avg.mov_average))
print('Moving Average Forecast',rmsmaf)

fit2 = SimpleExpSmoothing(np.asarray(train['Papers'])).fit(smoothing_level=0.6,optimized=True)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.plot(y_hat_avg['SES'], label='Simple Exponential Smoothing')
rmsses = sqrt(mean_squared_error(test.Papers, y_hat_avg.SES))
print('SES',rmsses)

fit1 = Holt(np.asarray(train['Papers'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_lin'] = fit1.forecast(len(test))
plt.plot(y_hat_avg['Holt_lin'], label='Holt Linear')
rmshl = sqrt(mean_squared_error(test.Papers, y_hat_avg.Holt_lin))
print('Holt Linear',rmshl)

step_fit = auto_arima(train, start_p=1, start_q=1, max_p=5, max_q=5, m=12, start_P=0, seasonal=False, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True, random=True, random_state=42, n_fits=25)
#step_fit.summary()
y_hat_avg['ARIMA_auto'] = step_fit.predict(n_periods=3)
plt.plot(y_hat_avg['ARIMA_auto'], label='Auto ARIMA Forecast')
rmshl = sqrt(mean_squared_error(test.Papers, y_hat_avg.ARIMA_auto))
print('Auto ARIMA Forecast',rmshl)

plt.xlim((1990,2020))
plt.ylim((0,350000))
plt.xlabel('Year')
plt.ylabel('Papers Published')
plt.legend(loc=2)
fig.tight_layout()
plt.savefig('dblp3TrainTest.png')
plt.show()


# Plot FORECAST confidence intervals
fig, ax = plt.subplots(figsize=(8,5))
trainN = yearNewSdf.loc[2002:2016]
timesN = yearNewSdf.loc[1990:2002]
logarN = yearNewSdf.loc[2000:2019]
logarB = yearNewSdf.loc[2017:2019]
t2d = np.arange(2000,2020,1)
s2d = func(t2d, *popt)

trainN.index = pd.DatetimeIndex(start='2002', end='2017', freq='A')
#timesN.index = pd.DatetimeIndex(start='1990', end='2003', freq='A')
ax = trainN.plot(ax=ax, color='blue',linestyle='-')
#ax = timesN.plot(ax=ax, color='blue',label='Old Series',linestyle=':')

model = ARIMA(trainN.Papers.astype(float), order=(0,1,0), dates=trainN.index)
model_fit = model.fit(disp=False)
model_for = model_fit.forecast(steps=3)
fig = model_fit.plot_predict(15,17,alpha=0.05,plot_insample=False,ax=ax)

logarN.index = pd.DatetimeIndex(start='2000', end='2020', freq='A')
logarN.Papers = s2d
ax = logarN.plot(ax=ax, label='Log Series',color='red',linestyle='--')
RMSE = 1774.5
bound_upper = (logarN.Papers + 2*RMSE).tail(3)
bound_lower = (logarN.Papers - 2*RMSE).tail(3)
logarB.index = pd.DatetimeIndex(start='2017', end='2020', freq='A')
ax.fill_between(logarB.index, bound_lower, bound_upper, color = 'pink', alpha = 0.50)

ax.set_xlim(pd.Timestamp('2010-12-31'), pd.Timestamp('2020-12-31'))
ax.set_ylim([230000,350000])
plt.xlabel('Year')
plt.ylabel('Papers Published')
blue_patch = mpatches.Patch(color='blue', label='Train Series')
red_patch = mpatches.Patch(color='red', label='Fitted Log Curve')
pink_patch = mpatches.Patch(color='pink', label='Log Prediction Interval')
cyan_patch = mpatches.Patch(color='teal', label='ARIMA(0,1,0) Forecast')
gray_patch = mpatches.Patch(color='gray', label='ARIMA Prediction Interval')
plt.legend(handles=[blue_patch,red_patch,pink_patch,cyan_patch,gray_patch],loc=2)
fig.tight_layout()
plt.savefig('dblp4Prediction.png')
plt.show()
