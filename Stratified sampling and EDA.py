# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:35:56 2018

@author: aman manawat
"""

import numpy as np 
import pandas as pd 
import json

from pandas.io.json import json_normalize
import seaborn as sns 

import os 

os.getcwd()


# Converting json to csv
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
def load_df(filename):
    path = "" + filename
    print(path + "Path")
    df = pd.read_csv(path, converters={column: json.loads for column in json_cols}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

train = load_df('C:/Users/aman manawat/Documents/Kaggle/Google Analytics Customer Prediction/all/train.csv')
#train.to_csv("train_clean.csv")
print ("There are " + str(train.shape[0]) + " rows and " + str(train.shape[1]) + " raw columns in this dataset")

###########################################################################################################


# Stratified Sampling 


#August 2016
aug2016 = train[(train.date > 20160800) & (train.date < 20160831 )]

aug2016.shape

aug2016RevenueNull = aug2016[aug2016.totals_transactionRevenue.isnull()]

aug2016RevenueNull.shape

aug2016Revenue = aug2016[aug2016.totals_transactionRevenue.notnull()]

aug2016Revenue.shape

aug2016RevenueisNull = aug2016RevenueNull.sample(frac=0.035)
aug2016RevenueisNull.shape

framesaug2016 = [aug2016RevenueisNull,aug2016Revenue]
aug2016frame = pd.concat(framesaug2016)
aug2016frame.shape


# Sept 2016

sept2016 = train[(train.date > 20160900) & (train.date < 20160930 )]

sept2016.shape

sept2016RevenueNull = sept2016[sept2016.totals_transactionRevenue.isnull()]

sept2016RevenueNull.shape

sept2016Revenue = sept2016[sept2016.totals_transactionRevenue.notnull()]

sept2016Revenue.shape

sept2016RevenueisNull = sept2016RevenueNull.sample(frac=0.035)
sept2016RevenueisNull.shape

framessept2016 = [sept2016RevenueisNull,sept2016Revenue]
sept2016frame = pd.concat(framessept2016)
sept2016frame.shape

# Oct 2016

oct2016 = train[(train.date > 20161000) & (train.date < 20161031 )]

oct2016.shape

oct2016RevenueNull = oct2016[oct2016.totals_transactionRevenue.isnull()]

oct2016RevenueNull.shape

oct2016Revenue = oct2016[oct2016.totals_transactionRevenue.notnull()]

oct2016Revenue.shape

oct2016RevenueisNull = oct2016RevenueNull.sample(frac=0.029)
oct2016RevenueisNull.shape

framesoct2016 = [oct2016RevenueisNull,oct2016Revenue]
oct2016frame = pd.concat(framesoct2016)
oct2016frame.shape

# Nov 2016 

nov2016 = train[(train.date > 20161100) & (train.date < 20161130 )]

nov2016.shape

nov2016RevenueNull = nov2016[nov2016.totals_transactionRevenue.isnull()]

nov2016RevenueNull.shape

nov2016Revenue = nov2016[nov2016.totals_transactionRevenue.notnull()]

nov2016Revenue.shape

nov2016RevenueisNull = nov2016RevenueNull.sample(frac=0.024)
nov2016RevenueisNull.shape

framesnov2016 = [nov2016RevenueisNull,nov2016Revenue]
nov2016frame = pd.concat(framesnov2016)
nov2016frame.shape

# Dec 2016 

dec2016 = train[(train.date > 20161200) & (train.date < 20161231 )]

dec2016.shape

dec2016RevenueNull = dec2016[dec2016.totals_transactionRevenue.isnull()]

dec2016RevenueNull.shape

dec2016Revenue = dec2016[dec2016.totals_transactionRevenue.notnull()]

dec2016Revenue.shape

dec2016RevenueisNull = dec2016RevenueNull.sample(frac=0.028)
dec2016RevenueisNull.shape

framesdec2016 = [dec2016RevenueisNull,dec2016Revenue]
dec2016frame = pd.concat(framesdec2016)
dec2016frame.shape

# Jan 2017

jan2017 = train[(train.date > 20170100) & (train.date < 20170131 )]

jan2017.shape

jan2017RevenueNull = jan2017[jan2017.totals_transactionRevenue.isnull()]

jan2017RevenueNull.shape

jan2017Revenue = jan2017[jan2017.totals_transactionRevenue.notnull()]

jan2017Revenue.shape

jan2017RevenueisNull = jan2017RevenueNull.sample(frac=0.044)
jan2017RevenueisNull.shape

framesjan2017 = [jan2017RevenueisNull,jan2017Revenue]
jan2017frame = pd.concat(framesjan2017)
jan2017frame.shape


# Feb 2017


feb2017 = train[(train.date > 20170200) & (train.date < 20170229 )]

feb2017.shape

feb2017RevenueNull = feb2017[feb2017.totals_transactionRevenue.isnull()]

feb2017RevenueNull.shape

feb2017Revenue = feb2017[feb2017.totals_transactionRevenue.notnull()]

feb2017Revenue.shape

feb2017RevenueisNull = feb2017RevenueNull.sample(frac=0.044)
feb2017RevenueisNull.shape

framesfeb2017 = [feb2017RevenueisNull,feb2017Revenue]
feb2017frame = pd.concat(framesfeb2017)
feb2017frame.shape

# March 2017


march2017 = train[(train.date > 20170300) & (train.date < 20170331 )]

march2017.shape

march2017RevenueNull = march2017[march2017.totals_transactionRevenue.isnull()]

march2017RevenueNull.shape

march2017Revenue = march2017[march2017.totals_transactionRevenue.notnull()]

march2017Revenue.shape

march2017RevenueisNull = march2017RevenueNull.sample(frac=0.039)
march2017RevenueisNull.shape

framesmarch2017 = [march2017RevenueisNull,march2017Revenue]
march2017frame = pd.concat(framesmarch2017)
march2017frame.shape

# April 2017


april2017 = train[(train.date > 20170400) & (train.date < 20170430 )]

april2017.shape

april2017RevenueNull = april2017[april2017.totals_transactionRevenue.isnull()]

april2017RevenueNull.shape

april2017Revenue = april2017[april2017.totals_transactionRevenue.notnull()]

april2017Revenue.shape

april2017RevenueisNull = april2017RevenueNull.sample(frac=0.039)
april2017RevenueisNull.shape

framesapril2017 = [april2017RevenueisNull,april2017Revenue]
april2017frame = pd.concat(framesapril2017)
april2017frame.shape


# May 2017


may2017 = train[(train.date > 20170500) & (train.date < 20170531 )]

may2017.shape

may2017RevenueNull = may2017[may2017.totals_transactionRevenue.isnull()]

may2017RevenueNull.shape

may2017Revenue = may2017[may2017.totals_transactionRevenue.notnull()]

may2017Revenue.shape

may2017RevenueisNull = may2017RevenueNull.sample(frac=0.038)
may2017RevenueisNull.shape

framesmay2017 = [may2017RevenueisNull,may2017Revenue]
may2017frame = pd.concat(framesmay2017)
may2017frame.shape

# June 2017


june2017 = train[(train.date > 20170600) & (train.date < 20170630 )]

june2017.shape

june2017RevenueNull = june2017[june2017.totals_transactionRevenue.isnull()]

june2017RevenueNull.shape

june2017Revenue = june2017[june2017.totals_transactionRevenue.notnull()]

june2017Revenue.shape

june2017RevenueisNull = june2017RevenueNull.sample(frac=0.040)
june2017RevenueisNull.shape

framesjune2017 = [june2017RevenueisNull,june2017Revenue]
june2017frame = pd.concat(framesjune2017)
june2017frame.shape

# July 2017


july2017 = train[(train.date > 20170700) & (train.date < 20170731 )]

july2017.shape

july2017RevenueNull = july2017[july2017.totals_transactionRevenue.isnull()]

july2017RevenueNull.shape

july2017Revenue = july2017[july2017.totals_transactionRevenue.notnull()]

july2017Revenue.shape

july2017RevenueisNull = july2017RevenueNull.sample(frac=0.038)
july2017RevenueisNull.shape

framesjuly2017 = [july2017RevenueisNull,july2017Revenue]
july2017frame = pd.concat(framesjuly2017)
july2017frame.shape


###### combined data 

GoogleFrame = [aug2016frame,
               sept2016frame,
               oct2016frame,
               nov2016frame,
               dec2016frame,
               jan2017frame,
               feb2017frame,
               march2017frame,
               april2017frame,
               may2017frame,
               june2017frame,
               july2017frame]

Googletrain = pd.concat(GoogleFrame)
Googletrain.shape

#exporting to csv
Googletrain.to_csv("Googletrain.csv")



###$ EDA ####
#EDA: Date plots 

import datetime

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

Googletrain['date'] = Googletrain['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
cnt_srs = Googletrain.groupby('date')['totals_transactionRevenue'].agg(['size', 'count'])
cnt_srs.columns = ["count", "count of non-zero revenue"]
cnt_srs = cnt_srs.sort_index()
#cnt_srs.index = cnt_srs.index.astype('str')
trace1 = scatter_plot(cnt_srs["count"], 'red')
trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')

fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Date - Count", "Date - Non-zero Revenue count"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')




##### Corrplot ##########

Googletrain.dtypes
Googletrain.corr()


                            ['channelGrouping',
                            'device_browser',
                            'device_deviceCategory',
                            'device_operatingSystem',
                            'geoNetwork_city',
                            'geoNetwork_continent',
                            'geoNetwork_country',
                            'geoNetwork_metro',
                            'geoNetwork_networkDomain',
                            'geoNetwork_region',
                            'geoNetwork_subContinent',
                            'trafficSource_adContent',
                            'trafficSource_adwordsClickInfo.adNetworkType',
                            'trafficSource_adwordsClickInfo.gclId',
                            'trafficSource_adwordsClickInfo.page',
                            'trafficSource_adwordsClickInfo.slot',
                            'trafficSource_campaign',
                            'trafficSource_keyword',
                            'trafficSource_medium',
                            'trafficSource_referralPath',
                            'trafficSource_source',
                            'trafficSource_adwordsClickInfo.isVideoAd',
                            'trafficSource_isTrueDirect',
                            'totals_transactionRevenue']

Googletrain['channelGrouping'].astype('str')
type(Googletrain['device_browser'][0])


 