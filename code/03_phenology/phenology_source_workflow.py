# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:53:58 2023

@author: ammar
"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt          
import matplotlib.dates as mdates
import datetime

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from datetime import datetime as dt
from scipy.stats.stats import pearsonr
import scipy
#########################################################################################################

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology\LAI_towers")

df=pd.read_csv("hf150-01-hem-lai.csv")  

df['TIMESTAMP'] = pd.to_datetime(df["date"])
df["TIMESTAMP"]=pd.to_datetime(df['TIMESTAMP'])
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.groupby([df['TIMESTAMP'].dt.date]).mean()
df=df.reset_index()
daily_hem_lai=df
plt.plot(df["year"], df["lai_masked"])


idx=pd.date_range(start='01/01/2004 00:00:00', periods=7305)  # daily. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df
df=df.reset_index()
daily_hem_lai=df



df["TIMESTAMP"]=df["index"]
df=df.drop(['index'], axis=1)
back=df
df=df.drop(['TIMESTAMP'], axis=1)
df=df.interpolate()
df['TIMESTAMP']=back['TIMESTAMP']
daily_hem_lai=df


df=daily_hem_lai

idx=pd.date_range(start='10/01/1991 00:00:00', periods=350640, freq='30T')  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df
df=df.reset_index()
df['TIMESTAMP']=df['index']
daily_hem_lai=df

df1 = pd.concat([daily_hem_lai]).set_index('TIMESTAMP')
df2= df1.resample('H').mean().ffill()
df=df2.reset_index()
hr_hem_lai=df




##################################################################################################
# EMS LAI 

#os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology")

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology\LAI_towers")

df=pd.read_csv("hf069-01-lai-ems.csv")  

category = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'H1', 'H2']

df=df[df['plot'].isin(category)]

#df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%Y-%m-%d')
df['TIMESTAMP'] = pd.to_datetime(df["date"])
df["TIMESTAMP"]=pd.to_datetime(df['TIMESTAMP'])
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.groupby([df['TIMESTAMP'].dt.date]).mean()
df=df.reset_index()
daily_ems_lai=df
plt.plot(df["year"], df["lai"])


idx=pd.date_range(start='04/15/1998 00:00:00', periods=9392)  # daily
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df
df=df.reset_index()
daily_ems_lai=df

df["TIMESTAMP"]=df["index"]
df=df.drop(['index'], axis=1)
back=df
df=df.drop(['TIMESTAMP'], axis=1)
df=df.interpolate()
df['TIMESTAMP']=back['TIMESTAMP']
daily_ems_lai=df


df=daily_ems_lai

idx=pd.date_range(start='10/01/1991 00:00:00', periods=11780)  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df
df=df.reset_index()
df['TIMESTAMP']=df['index']
daily_ems_lai=df

df1 = pd.concat([daily_ems_lai]).set_index('TIMESTAMP')
df2= df1.resample('H').mean().ffill()
df=df2.reset_index()
hr_ems_lai=df

hr_ems_lai.to_csv('hr_ems_lai.csv', index=False, header=True)
# used this in ML 
##############################################################################################













df.to_csv('ems_footprint.csv', index=False, header=True)

df=pd.read_csv("ems_mod_merg.csv")  

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
merg=df
df=df.groupby(['year','month']).mean()
df=df.reset_index()
month=df
plt.plot(df["year"], df["lai"])
df["Year"]=phen_month["year"]
phen_month=df

#######################################################################################################

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology\LAI_towers")
df=pd.read_csv("HF-ems-MOD15A2H-061-results.csv")
df["lai_mod"]=df["MOD15A2H_061_Lai_500m"]
df["year"]= pd.to_datetime(df['Date']).dt.year
df["month"]= pd.to_datetime(df['Date']).dt.month
df=df.groupby(['year','month']).mean()
df=df.reset_index()
month_mod=df
plt.plot(df["year"], df["lai_mod"])










##############################################################################3
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("daily_EMS_ML.csv")
mask = (df['TIMESTAMP'] >='1992-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]=pd.to_datetime(df.TIMESTAMP)
back=df

#########################################################################################################

## use this portion for yearly lai
df=merg
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.groupby(['year']).mean()
df=df.reset_index()
year=df
plt.plot(df["year"], df["lai"])












#########################################################################################
#sat

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology")
df=pd.read_csv("harvard-forest-MCD15A2H-061-results_hem.csv")  
df['TIMESTAMP'] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
df["Year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=pd.concat((df["TIMESTAMP"],df["Year"],df["MCD15A2H_061_Fpar_500m"],df["MCD15A2H_061_Lai_500m"]),axis=1)

modis_daily_lai=df



##############################################################################################
df=ems_daily_lai
x=df['year']
y=df['lai'].astype(float)
pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="EMS", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("daily",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

####################################################################################3

df=ems_daily_lai

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

df=df.groupby(['Year','month']).mean()

df=df.reset_index()

month_ems_lai=df

x=df['Year']
y=df['lai'].astype(float)

pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['lai'].astype(float),color="black", linestyle='--',label="EMS lai", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('monthly lai')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))
#ax.set_title("winter total ET: corr=0.51",fontsize=15)
#ax.set_title("spring total ET: corr=0.13",fontsize=15)
#ax.set_title("summer total ET: corr=0.008",fontsize=15)
#ax.set_title("Fall total ET: corr=0.065",fontsize=15)
ax.set_title("mean monthly lai: corr=0.03",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

df=month_ems_lai

def season_nam(df):    
    if (df['month'] ==12) or (df['month'] ==1) or (df['month'] ==2):
        return 'winter'
    
    elif (df['month'] ==3) or (df['month'] ==4) or (df['month'] ==5):

        return 'spring'

    elif (df['month'] ==6) or (df['month'] ==7) or (df['month'] ==8):
        return 'summer'

    elif (df['month'] ==9) or (df['month'] ==10) or (df['month'] ==11):
        return 'fall'
    
df['season_nam'] = df.apply(season_nam, axis = 1)    
#df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.groupby(['season_nam','Year']).mean()
df=df.reset_index()

seasonal_ems_lai=df


df=seasonal_ems_lai

df=ems_daily_lai
mask = (df['month'] >=6) & (df['month'] <= 6)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df.groupby(['month','year']).mean()
june_ems_lai=df

df=ems_daily_lai
mask = (df['month'] >=7) & (df['month'] <= 7)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df.groupby(['month','year']).mean()
jul_ems_lai=df

df=ems_daily_lai
mask = (df['month'] >=8) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df.groupby(['month','year']).mean()
aug_ems_lai=df



df=ems_daily_lai
df=df.groupby(['TIMESTAMP']).mean()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
annual_ems_lai=df
###################################################################
df=june_ems_lai
df = df.iloc[2:]
df=df.reset_index()
x=df['year']
y=df['lai'].astype(float)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="EMS", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("june",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##################################################################################################


df=jul_ems
df = df.iloc[2:]
df=df.reset_index()
x=df['year']
y=df['lai'].astype(float)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="EMS", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("july",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


##############################################################################################

df=aug_ems_lai
df = df.iloc[2:]
df=df.reset_index()
x=df['year']
y=df['lai'].astype(float)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="EMS", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("aug",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##########################################################################################################

df=aug_ems
df = df.iloc[2:]
df=df.reset_index()
x=df['year']
y=df['lai'].astype(float)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="EMS", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("aug",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#######################################################################################################
df=ems_daily_lai
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
df=df.dropna()


annual_ems_lai=df


df=annual_ems_lai
#df = df.iloc[2:]
df=df.reset_index()
x=df['year']
y=df['lai'].astype(float)

pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="EMS_Lai: 0.26", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("annual (1998-2022: 2000-2004 missing)",fontsize=10)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#######################################################################################################

modis_daily_lai

df=ems_daily_lai
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
df=df.dropna()

mask = (df['TIMESTAMP'] >= '1998-01-01') & (df['TIMESTAMP'] <= '2008-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1=df
#####################################################################################################

# HEM interannual varaibilty

######################################################################################################

df=annual_ems_lai
mask = (df['TIMESTAMP'] >= '2009-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2=df

#df=df_1
df=df_2

x=df['Year']
y=df['lai'].astype(float)
#y=df['HEM_ET'].astype(float)
pearsonr(x, y)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(x, df['lai'].astype(float),color="blue", linestyle='--',label="EMS_lai, -0.62", marker='o')
#plt.plot(x, df['HEM_ET'],color="green", linestyle='--',label="HEM_ET, -0.62", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '9')
ax.legend(loc='upper right', fontsize = '9')
#ax.legend(loc='lower right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('lai')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))

#slope is z[0]
ax.set_title("mean EMS lai",fontsize=11)

#ax.set_title(" mean EMS lai (1998-2008)",fontsize=12)
ax.set_title(" mean EMS lai (2009-2022)",fontsize=12)
#ax.set_title(" total annual ET (2013-2022): corr=0.57",fontsize=12)
#ax.set_title("total monthly ET: corr=0.05",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#######################################################################################################

df=seasonal_ems_lai
df=df.reset_index()

seasonal_ems_lai=df
mask = (df['season_nam']=='fall')
df = df.loc[mask]
df.index = np.arange(0, len(df))
fall_ems_lai=df


df=seasonal_ems_lai
mask = (df['season_nam']=='spring')
df = df.loc[mask]
df.index = np.arange(0, len(df))
spring_ems_lai=df

df=seasonal_ems_lai
mask = (df['season_nam']=='summer')
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer_ems_lai=df

df=seasonal_ems_lai
mask = (df['season_nam']=='winter')
df = df.loc[mask]
df.index = np.arange(0, len(df))
winter_ems_lai=df


#df=winter_ems_lai
#df=spring_ems_lai
#df=summer_ems_lai
df=fall_ems_lai

#df = df.iloc[2:]
x=df['year']
y=df['lai'].astype(float)
pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="ems_lai: corr -0.8", marker='o')
#plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="ems_lai: corr -0.04", marker='o')
#plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="ems_lai: corr 0.42", marker='o')
plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="ems_lai: corr 0.15", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('lai')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("fall EMS LAI",fontsize=15)
#ax.set_title("Spring EMS LAI",fontsize=15)
#ax.set_title("Summer EMS LAI",fontsize=15)
#ax.set_title("winter EMS LAI",fontsize=15)

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#ax.set_ylim(0,8)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()




############################################################################################################
#HEM lai
#############################################################################################################
os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology")
df=pd.read_csv("hf150-01-hem-lai.csv")  
df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%Y-%m-%d')
df["Year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df["lai"]= df["lai_masked"]
hem_daily_lai=df

df=hem_daily_lai

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

df=df.groupby(['Year','month']).mean()

df=df.reset_index()

month_hem_lai=df

x=df['Year']
y=df['lai'].astype(float)

pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['lai'].astype(float),color="black", linestyle='--',label="HEM lai", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('monthly lai')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))
#ax.set_title("winter total ET: corr=0.51",fontsize=15)
#ax.set_title("spring total ET: corr=0.13",fontsize=15)
#ax.set_title("summer total ET: corr=0.008",fontsize=15)
#ax.set_title("Fall total ET: corr=0.065",fontsize=15)
ax.set_title("mean monthly HEM lai: corr=-0.65",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

df=month_hem_lai

def season_nam(df):    
    if (df['month'] ==12) or (df['month'] ==1) or (df['month'] ==2):
        return 'winter'
    
    elif (df['month'] ==3) or (df['month'] ==4) or (df['month'] ==5):

        return 'spring'

    elif (df['month'] ==6) or (df['month'] ==7) or (df['month'] ==8):
        return 'summer'

    elif (df['month'] ==9) or (df['month'] ==10) or (df['month'] ==11):
        return 'fall'
    
df['season_nam'] = df.apply(season_nam, axis = 1)    
#df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.groupby(['season_nam','Year']).mean()
df=df.reset_index()

seasonal_hem_lai=df
df=seasonal_hem_lai

######################################################################################################

# hem inter annual variability

df=hem_daily_lai
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
df=df.dropna()
annual_hem_lai=df

mask = (df['TIMESTAMP'] >= '1998-01-01') & (df['TIMESTAMP'] <= '2008-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1=df


df=annual_hem_lai
mask = (df['TIMESTAMP'] >= '2009-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2=df


df=df_2

x=df['Year']
y=df['lai'].astype(float)
#y=df['HEM_ET'].astype(float)
pearsonr(x, y)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(x, df['lai'].astype(float),color="blue", linestyle='--',label="EMS_lai, -0.96", marker='o')
#plt.plot(x, df['HEM_ET'],color="green", linestyle='--',label="HEM_ET, -0.62", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
#ax.legend(loc='upper left', fontsize = '9')
ax.legend(loc='upper right', fontsize = '9')
#ax.legend(loc='lower right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('lai')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))

#slope is z[0]
ax.set_title("mean EMS lai",fontsize=11)
#ax.set_title(" mean EMS lai (1998-2008)",fontsize=12)
ax.set_title(" mean HEM lai (2009-2022)",fontsize=12)
#ax.set_title(" total annual ET (2013-2022): corr=0.57",fontsize=12)
#ax.set_title("total monthly ET: corr=0.05",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################
# hem seasonal trend

df=seasonal_hem_lai

df=df.reset_index()

seasonal_hem_lai=df
mask = (df['season_nam']=='fall')
df = df.loc[mask]
df.index = np.arange(0, len(df))
fall_hem_lai=df


df=seasonal_hem_lai
mask = (df['season_nam']=='spring')
df = df.loc[mask]
df.index = np.arange(0, len(df))
spring_hem_lai=df

df=seasonal_hem_lai
mask = (df['season_nam']=='summer')
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer_hem_lai=df

df=seasonal_hem_lai
mask = (df['season_nam']=='winter')
df = df.loc[mask]
df.index = np.arange(0, len(df))
winter_hem_lai=df


#df=winter_hem_lai
#df=spring_hem_lai
df=summer_hem_lai
#df=fall_hem_lai

#df = df.iloc[2:]
x=df['Year']
y=df['lai'].astype(float)
pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['lai'].astype(float),color="blue", linestyle='--',label="HEM_lai: corr -0.73", marker='o')
#plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="HEM_lai: corr -0.04", marker='o')
#plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="HEM_lai: corr 0.42", marker='o')
#plt.plot(df['year'], df['lai'].astype(float),color="blue", linestyle='--',label="HEM_lai: corr 0.15", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('lai')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
scipy.stats.pearsonr(x,y)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
#ax.set_title("fall HEM LAI",fontsize=15)
#ax.set_title("Spring HEM LAI",fontsize=15)
ax.set_title("Summer HEM LAI",fontsize=15)
#ax.set_title("winter HEM LAI",fontsize=15)

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#ax.set_ylim(0,8)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

############################################################################################################

# compare

























##############################################################################################################
## GPP other sites

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology")                                       
df=pd.read_csv("harvard-forest-MCD15A2H-061-results.csv")
df["month"]= pd.to_datetime(df['Date']).dt.month
plt.plot(df["MCD15A2H_061_Lai_500m"],'r-', label="LAI")

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
lai_daily_summer=df

plt.plot(df["MCD15A2H_061_Lai_500m"],'r-', label="LAI")

df["year"]= pd.to_datetime(df['Date']).dt.year
df=df.groupby(['month','year']).mean()
lai_month_summer=df

df=lai_daily_summer
df=df.groupby(['year']).mean()
lai_annual_summer=df
plt.plot(df["MCD15A2H_061_Lai_500m"],'r-', label="LAI")
plt.plot(df["MCD15A2H_061_Fpar_500m"],'b-', label="LAI")
df=df.rename_axis('TIMESTAMP').reset_index()


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df["TIMESTAMP"],df['MCD15A2H_061_Lai_500m'].astype(float),color="blue", linestyle='--',label="LAI", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('LAI')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Annual (growing season)",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#ax.set_ylim(0,40)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##############################################################################################################

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology")                                       
df=pd.read_csv("MOD16A2GF-061-results.csv")

df["month"]= pd.to_datetime(df['Date']).dt.month
plt.plot(df["MOD16A2GF_061_ET_500m"],'r-', label="ET")

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
ET_daily_summer=df

plt.plot(df["MOD16A2GF_061_ET_500m"],'r-', label="ET")

df["year"]= pd.to_datetime(df['Date']).dt.year
df=df.groupby(['month','year']).mean()
ET_month_summer=df

df=ET_daily_summer
df=df.groupby(['year']).mean()
ET_annual_summer=df
plt.plot(df["MOD16A2GF_061_ET_500m"],'r-', label="ET")
plt.plot(df["MOD16A2GF_061_ET_500m"],'b-', label="ET")
df=df.rename_axis('TIMESTAMP').reset_index()


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df["TIMESTAMP"],df['MOD16A2GF_061_ET_500m'].astype(float),color="blue", linestyle='--',label="ET", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET (mm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Annual (growing season)",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#ax.set_ylim(0,40)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

###########################################################################################################

# get data by periods 
os.chdir(r"C:\ammara_MD\a_harvard\Project\data\phenology")                                       
df=pd.read_csv("MOD16A2GF-061-results.csv")

df["month"]= pd.to_datetime(df['Date']).dt.month
plt.plot(df["MOD16A2GF_061_ET_500m"],'r-', label="ET")

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
ET_daily_summer=df
df["year"]= pd.to_datetime(df['Date']).dt.year

def seasonal(df):
    
    if (df['year'] >=2000) & (df['year'] <= 2003):
        return '1'
    
    elif (df['year'] >=2004) & (df['year'] <= 2008):
        return '2'

    elif (df['year'] >=2009) & (df['year'] <= 2014):     
        return '3'

    elif (df['year'] >=2015) & (df['year'] <= 2018):        
        return '4'
    
    elif (df['year'] >=2019) & (df['year'] <= 2023):
        return '5'




df['seasonal'] = df.apply(seasonal, axis = 1) 


df["year"]= pd.to_datetime(df['Date']).dt.year
df=df.groupby(['month','seasonal']).mean()
df=df.reset_index()
ET_month_summer=df

df=ET_month_summer
mask = (df['month'] >=6) & (df['month'] <= 6)
df = df.loc[mask]
df.index = np.arange(0, len(df))
ET_june=df



df=ET_month_summer
mask = (df['month'] >=7) & (df['month'] <= 7)
df = df.loc[mask]
df.index = np.arange(0, len(df))
ET_jul=df


df=ET_month_summer
mask = (df['month'] >=8) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
ET_aug=df


#df["year"]= pd.to_datetime(df["year"])




plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(ET_june['seasonal'],ET_june['MOD16A2GF_061_ET_500m'],color="blue", linestyle='--',label="June", marker='o')
#.plot(ET_jul["seasonal"].astype(float),ET_jul['MOD16A2GF_061_ET_500m'].astype(float),color="green", linestyle='--',label="Jul", marker='o')
plt.plot(ET_aug["seasonal"].astype(float),ET_aug['MOD16A2GF_061_ET_500m'].astype(float),color="purple", linestyle='--',label="aug", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET (mm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Monthly (growing season)",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
#ax.set_ylim(0,40)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()





















df=ET_daily_summer
df=df.groupby(['year']).mean()
ET_annual_summer=df
plt.plot(df["MOD16A2GF_061_ET_500m"],'r-', label="ET")
plt.plot(df["MOD16A2GF_061_ET_500m"],'b-', label="ET")
df=df.rename_axis('TIMESTAMP').reset_index()












































df_LAI=pd.concat((df["Date"],df["MCD15A2H_061_Fpar_500m"],df["MCD15A2H_061_Lai_500m"]),axis=1)

plt.plot(df_CC1_GPP.MYD17A2HGF_061_PsnNet_500m,'g-', label="Photosyn_US_CC1") 
plt.legend()
plt.show()
