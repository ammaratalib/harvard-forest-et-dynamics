# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:24:54 2021

@author: Ammara
"""




#potato data

#http://co2.aos.wisc.edu/data/potato/

##### data that prof. used is in local time

import datetime
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import numpy as np
from math import sqrt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import pymannkendall as mk
##################################################################################

import pandas as pd
import os
from sklearn.metrics import r2_score
import scipy
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
from scipy.stats.stats import pearsonr
import matplotlib.ticker as ticker
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from datetime import datetime as dt


##########################################################################################################


#os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")
# data is hourly
#df=pd.read_csv("EMS_ustarfiltered_thru2022.csv")  # 1991-2022, # of days 11688

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")
df=pd.read_csv("ems_ustfilt_2023.csv")

df.head(10)
df.tail(10)
back=df
df=back

idx=pd.date_range(start='01/01/1991 00:00:00', periods=289274, freq='60T')  #  hourly. 

df["TIMESTAMP"]=df["datetime"] # dont use datetime

df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df

#index is correct date.TIMESTAMP date mess up for missing values
# so let remove messed up timestamp column
df=df.iloc[:,1:167]
#df=df.astype(float) 
df.head(10)
df.tail(10)
temp=df
df=temp
df = df.iloc[:-2] 

#df=df.rename_axis('TIMESTAMP').reset_index()
df["date"] = pd.to_datetime(df.TIMESTAMP)
df["TIMESTAMP"] = pd.to_datetime(df.TIMESTAMP)

###########################################################################################################
df['year'] = df['TIMESTAMP'].dt.year
#df['day'] = df['TIMESTAMP'].dt.day
df['day'] = df['TIMESTAMP'].dt.dayofyear

# eddy proc need  these columns in the start
df.insert(1, 'Year',df['year'])
df.insert(2, 'DoY', df['day'])
#add hour column for eddy proc
#try using zero
s=pd.Series(np.arange(1,24.5,1))
#######################################################################################################
#change this based on data size (number of days of data), 
#([s]*(#number of days)
k=pd.concat([s]*(12053),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df=df.reset_index()
#df=df[:-1]
df.insert(3,'Hour',k)
df=df.iloc[:,1:170]
df.tail(10)
back["DateTime"]=back["TIMESTAMP"]
back=df

#df=df[:-1]
#########################################################################################################

cols = df.columns.tolist()

#######################################################################################################

# gap fill data

back["DateTime"]=back["TIMESTAMP"]
df=back

## gapfilling data

#FC_1_1_! is same as NEE
df=pd.concat((df["TIMESTAMP"],df['DoY'],df["Year"], df['Hour'],df['CO2_1_1_1'],df['H2O_1_1_1'],df["WD"], df['TAU'],
df['Ustar'], df['SC'], df['TS_PI_1_1_A'],df["tsoil_upper_mean"],df['NEE'],df['H'],
df['RH'],df['WS'],df['PA'],df['Tair'],df['lambda'],df['VPD'],df['Rg'],df['PPFD'],
df['NETRAD'],df['LE'],df["GPP_f"],df["sos"],df["eos"]),axis=1)

df["DateTime"]=df["TIMESTAMP"]

df["GPP"]=df["GPP_f"]
df["PAR"]=df["PPFD"]

back_up_EMS=df
back_up_EMS.to_csv('back_up_EMS.csv', index=False, header=True)
back=df

df=pd.concat((df["TIMESTAMP"],df['DoY'],df["Year"], df['Hour'],df['CO2_1_1_1'],df['H2O_1_1_1'],df["WD"], df['TAU'],
df['Ustar'], df['SC'], df['TS_PI_1_1_A'],df["tsoil_upper_mean"],df['NEE'],df['H'],
df['RH'],df['WS'],df['PA'],df['Tair'],df['lambda'],df['VPD'],df['Rg'],df['PAR'],
df['NETRAD'],df['LE'],df["GPP"],df["sos"],df["eos"]),axis=1)


df.to_csv('gaps_ML_EMS.csv', index=False, header=True)


###########################################################################################################
##################################################################################################################

df=back_up_EMS
reframed=pd.concat((df["DateTime"],df[cols[1]],df[cols[2]],df['DoY'],df["NEE"],df["GPP"],df["LE"],df["H"],df['Rg'],df["Tair"],df["VPD"],df["Ustar"],df["PAR"],df["RH"],df["PA"],df["NETRAD"],df["WS"]),axis=1)

reframed.to_csv('gaps_EMS.csv', index=False, header=True)
df=back_up_EMS


####################################################################################################

# end date will be one day more than your desired end day

#t=df.columns()

# brig filled data
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")

fill_ems=pd.read_csv("filled_EMS.csv") # to use fixed LE used file " filled_EMS to copy LE"
back_up_EMS=pd.read_csv("back_up_EMS.csv") 

fill_ems['TIMESTAMP']=back_up_EMS["TIMESTAMP"]
df=fill_ems

#mask = (df['TIMESTAMP'] >='1991-10-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
#df = df.loc[mask]
#df.index = np.arange(0, len(df))

#reframed.to_csv('check.csv', index=False, header=True)


plt.plot(df['LE'],color='green')

#df["LE1"] = np.where(df["LE"]<-200, np.NaN, df["LE"])
#df["LE2"] = np.where(df["LE1"]>800, np.NaN, df["LE1"])
#df["LE"]=df["LE2"]


df['lambda']=(3149000-2370*(df['Tair']+273.16))*1e-6
plt.plot(df['lambda'],color='green')


df['ET']=(df['LE']/df['lambda'])*(1/1e6)*3600
df["GPP_g"]=(df["GPP"])*((12/10**6)*3600)
df["NEE_g"]=(df["NEE"])*((12/10**6)*3600)
plt.plot(df['ET'],color='red')

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['Year'] = df['TIMESTAMP'].dt.year
df.isnull().values.any()
df1 = df[df.isna().any(axis=1)]
#df=df.dropna()
df["Hr"]=df['TIMESTAMP'].dt.hour
df["sos"]=back_up_EMS["sos"]
df["eos"]=back_up_EMS["eos"]
EMS_hour_ET=df

df=EMS_hour_ET

###################################################################################################
df=EMS_hour_ET
#df["Rnet1"] = np.where(df["NETRAD"]<-500, np.NaN, df["NETRAD"])
#df["Rnet2"] = np.where(df["Rnet1"]>1000, np.NaN, df["Rnet1"])
df["Rnet"]=df["NETRAD"]
mask = (df['TIMESTAMP'] >='1991-10-29 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
doc=df
df=df.drop(['TIMESTAMP'], axis=1)
#df=df.drop(['DateTime'], axis=1)
#df=df.interpolate()

df["TIMESTAMP"]=doc["TIMESTAMP"]
df.isnull().values.any()
#####################################################################################################
back2=df
df = df.iloc[: , 1:] # remove column
df["datetime"]=df['TIMESTAMP']
df["PPFD"]=df['PAR']
#df=df.fillna('NA')
#############################################################################################################
# hourly EMS gap filled data
df.to_csv('ems_ustfilt_2023_filled.csv', index=False, header=True)

##########################################################################################################
# part for machine learning 

df=pd.read_csv("ems_ustfilt_2023_filled.csv") 
reddy=df

#plt.plot(df['Rnet'],color='red')
# check missing after reddy proc
#after 273936 from 1992-10-01 to 2023-12-31, VPD 99%
x=df[df['VPD'].notnull()]
      
df["NEE_f"]=df["NEE"]
df["H_f"]=df["H"]
df["RH_f"]=df["RH"]
df["PA_f"]=df["PA"]
df["WS_f"]=df["WS"]
df["Tair_f"]=df["Tair"]
df["Rg_f"]=df["Rg"]
df["NETRAD_f"]=df["Rnet"]
df["LE_f"]=df["LE"]
df["ET_f"]=df["ET"]
df["VPD_f"]=df["VPD"]
df["lambda_f"]=df["lambda"]
df["PAR_f"]=df["PAR"]
mask= (back['TIMESTAMP'] >='1991-10-29 00:00:00') & (back['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
back_m = back.loc[mask]
back_m.index = np.arange(0, len(back_m))

######################################################################################################

# check what variables are important 

###################################################################################################
df=pd.concat((back_m["TIMESTAMP"],back_m['DoY'],df["Year"], df['Hour'],back_m['CO2_1_1_1'],back_m['H2O_1_1_1'],back_m["WD"], back_m['TAU'],
back_m['Ustar'], back_m['SC'], back_m['TS_PI_1_1_A'],back_m["tsoil_upper_mean"],df['NEE_f'],df['H_f'],
df['RH_f'],df['WS_f'],df['PA_f'],df['Tair_f'],df['lambda_f'],df['VPD_f'],df['GPP_g'],df['NEE_g'],df['Rg_f'],df['PAR_f'],
df['NETRAD_f'],df['LE_f'],df['ET_f'],back_m['sos'],back_m['eos']),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
back3=df

####################################################################################################

#daily ET                                       
df=df.drop(['TIMESTAMP'], axis=1)
df=df.interpolate()
df['TIMESTAMP']=back3['TIMESTAMP']
df.isnull().values.any()
##mean
s=df.groupby([df['TIMESTAMP'].dt.date]).mean()
s=s.reset_index()
##sum
s_et=df.groupby([df['TIMESTAMP'].dt.date]).sum()
s_et=s_et.reset_index()

plt.plot(s_et['ET_f'],color='red')
plt.plot(s['Rg_f'],color='red')



#s["NEE_f"]=s_et['NEE']
#s["H_f"]=s_et['H']
#s["GPP_s"]=s_et['GPP']
#s["LE_f"]=s_et['LE']


df=pd.concat((s["TIMESTAMP"],s['DoY'],s["Year"], s['Hour'],s['CO2_1_1_1'],s['H2O_1_1_1'],s["WD"],s['TAU'],
s['Ustar'], s['SC'], s['TS_PI_1_1_A'],s["tsoil_upper_mean"],s_et['NEE_f'],s_et['H_f'],
s['RH_f'],s['WS_f'],s['PA_f'],s['Tair_f'],s['lambda_f'],s['VPD_f'],s_et['GPP_g'],s_et['NEE_g'],s['Rg_f'],s['PAR_f'],
s['NETRAD_f'],s_et['LE_f'],s_et['ET_f'],s["sos"],s["eos"]),axis=1)



#df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
EMS_daily_ET=df
df.isnull().values.any()


df.to_csv('PET_EMS_input.csv', index=False, header=True)
EMS_daily_ET=df
#df.to_csv('daily_EMS_ML.csv', index=False, header=True)

#######################################################################################################

df=EMS_daily_ET

#######################################################################################################
df=EMS_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
#df= df.groupby(['water_year']).sum()
#df=df.reset_index()
mask = (df['TIMESTAMP'] >='1992-10-02 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))
doc=df


############################################################################################################


#df=df.fillna(method='ffill')
df["CO2_1_1_1"].fillna(method='backfill', inplace=True)
df["H2O_1_1_1"].fillna(method='backfill', inplace=True)
df.isnull().values.any()
df.to_csv('daily_EMS_ML.csv', index=False, header=True)

plt.plot(df["ET_f"])

plt.plot(df["GPP_g"])

# pick columns that you want for ML model


df.to_csv('daily_EMS_ML.csv', index=False, header=True)

#df= df.groupby(['water_year']).sum()
#df=df.reset_index()
############################################################################################################

import pyhomogeneity as hg
from scipy import stats
EMS_daily_ET=df

df=EMS_daily_ET

df.isnull().values.any()

mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))

EMS_daily_ET_f=df
df=EMS_daily_ET_f

##################################################################################################
# calculate annual 

df=EMS_daily_ET_f

df= df.groupby(['water_year']).mean()
df=df.reset_index()


x1=df['water_year']
y1=df['PAR_f'].astype(float)

mk.original_test(y1)

stats.theilslopes(y1, x1, 0.95)
hg.pettitt_test(y1, alpha=0.05)

df=EMS_daily_ET_f


def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
    
    
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)


seas=df

mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
gw=df

df=seas
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
ngw=df



df=gw
#df=ngw
df= df.groupby(['water_year']).mean()
df=df.reset_index()



x1=df['water_year']
y1=df['VPD_f'].astype(float)

mk.original_test(y1)
stats.theilslopes(y1, x1, 0.95)
hg.pettitt_test(y1, alpha=0.05)




####################################################################################

plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
#p=6.9*10$^{-5}$
plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=0.01, Tau=0.33, slope=0.83",linestyle='--', marker='D',markersize=4)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
#plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.03,  Tau=0.53, slope=2.7",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total Annual VPD > $VPD_{95}$')  # we already handled the x-label with ax1


#ax.set_ylim(10,90)

z1 = np.polyfit(df['water_year'], df['VPD_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')



ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

####################################################################################################

## percentile 
df=EMS_daily_ET_f

mask = (df['VPD_f'] > df['VPD_f'].quantile(0.90)) 
df = df.loc[mask]
df.index = np.arange(0, len(df))
per=df

num=df['water_year'].value_counts()
num=num.reset_index()

#top_10=df.groupby('water_year')['VPD_f'].nlargest(10)
#df=top_10


df=num

df=per
df= df.groupby(['water_year']).sum()
df=df.reset_index()



x1=df['water_year']
y1=df['VPD_f'].astype(float)

mk.original_test(y1)

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
#p=6.9*10$^{-5}$
plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=0.01, Tau=0.33, slope=0.83",linestyle='--', marker='D',markersize=4)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.03,  Tau=0.53, slope=2.7",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total Annual VPD > $VPD_{95}$')  # we already handled the x-label with ax1


ax.set_ylim(10,90)

z1 = np.polyfit(df['water_year'], df['VPD_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['VPD_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#######################################################################################################

df=num
df.columns=['water_year','VPD_f']

df=df.sort_values(by='water_year')

x1=df['water_year']
y1=df['VPD_f'].astype(float)

mk.original_test(y1)

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
#p=6.9*10$^{-5}$
plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=0.01, Tau=0.31, slope=0.71",linestyle='--', marker='D',markersize=4)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.03,  Tau=0.51, slope=2.5",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Number of days with VPD > $VPD_{95}$')  # we already handled the x-label with ax1


ax.set_ylim(10,80)

z1 = np.polyfit(df['water_year'], df['VPD_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['VPD_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
####################################################################################################

## growing, non growing season 

df=EMS_daily_ET_f


def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
    
    
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)


seas=df

mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
gw=df

df=seas
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
ngw=df



df=gw
df= df.groupby(['water_year']).mean()
df=df.reset_index()



x1=df['water_year']
y1=df['VPD_f'].astype(float)

mk.original_test(y1)

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
#p=6.9*10$^{-5}$
plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=0.26, Tau=0.14, p=8.7*10$^{-4}$",linestyle='--', marker='D',markersize=4)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.44,  Tau=0.20, slope=3.8*10$^{-3}$",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('NonGrowing season mean Annual VPD')  # we already handled the x-label with ax1


ax.set_ylim(10,90)

z1 = np.polyfit(df['water_year'], df['VPD_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['VPD_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()








####################################################################################################
# max VPD
#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=0.003, Tau=0.38, slope=0.01",linestyle='--', marker='D',markersize=4)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.35,   Tau=0.24, slope=0.03",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Maximum annual VPD (kpa)')  # we already handled the x-label with ax1


ax.set_ylim(0.97,2.2)

z1 = np.polyfit(df['water_year'], df['VPD_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['VPD_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()



###################################################################################################
# Rg

x1=df['water_year']
y1=df['Rg_f'].astype(float)

mk.original_test(y1)

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['Rg_f'],color="blue",label="$SW_{in}$ (1993-2023),  p=0.01, Tau=0.30, slope=0.79",linestyle='--', marker='D',markersize=4)
plt.plot(df['water_year'].tail(11), df['Rg_f'].tail(11),color="red",label="$SW_{in}$ (2013-2023),  p=0.64, Tau=-0.1, slope=-0.49",linestyle='--', marker='*',markersize=7)


#plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=6.9*10$^{-5}$, Tau=0.50, slope=0.003",linestyle='--', marker='D',markersize=4)
#plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.04,        Tau=0.50, slope=0.008",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(8) 
ax.set_ylabel('Incoming Shortwave Radiation ($SW_{in}$) $Wm^{-2}$')  # we already handled the x-label with ax1
#ax.set_ylabel('Annual mean Air Temperature (${C}^o$)')  # we already handled the x-label with ax1


ax.set_ylim(120,180)

z1 = np.polyfit(df['water_year'], df['Rg_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['Rg_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##############################################################################################


# air temp

df=EMS_daily_ET_f

df= df.groupby(['water_year']).mean()
df=df.reset_index()

x1=df['water_year']
y1=df['Tair_f'].astype(float)

mk.original_test(y1)

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['Tair_f'],color="blue",label="$T_{air}$ (1993-2023), p=8*10$^{-4}$,Tau=0.43, slope=0.06",linestyle='--', marker='D',markersize=4)
plt.plot(df['water_year'].tail(11), df['Tair_f'].tail(11),color="red",label="$T_{air}$ (2013-2023), p=0.12,    Tau=0.38, slope=0.13",linestyle='--', marker='*',markersize=7)


#plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=6.9*10$^{-5}$, Tau=0.50, slope=0.003",linestyle='--', marker='D',markersize=4)
#plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.04,        Tau=0.50, slope=0.008",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '6.8')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(8) 
#ax.set_ylabel('Incoming Shortwave Radiation ($SW_{in}$) $Wm^{-2}$')  # we already handled the x-label with ax1
ax.set_ylabel('Annual mean Air Temperature (${C}^o$)')  # we already handled the x-label with ax1


#ax.set_ylim(120,180)

z1 = np.polyfit(df['water_year'], df['Tair_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['Tair_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################




x1=df['water_year'].tail(11)
y1=df['Tair_f'].astype(float).tail(11)
mk.original_test(y1)


# air temp

df=EMS_daily_ET_f

df= df.groupby(['water_year']).mean()
df=df.reset_index()

x1=df['water_year']
y1=df['WS_f'].astype(float)

mk.original_test(y1)

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['water_year'], df['WS_f'],color="blue",label="WS (1993-2023), p=1.4*10$^{-7}$, Tau=-0.67, slope=-0.02",linestyle='--', marker='D',markersize=4)
plt.plot(df['water_year'].tail(11), df['WS_f'].tail(11),color="red",label="WS (2013-2023), p=0.12,       Tau=0.38, slope=0.13",linestyle='--', marker='*',markersize=7)


#plt.plot(df['water_year'], df['VPD_f'],color="blue",label="VPD (1993-2023),  p=6.9*10$^{-5}$, Tau=0.50, slope=0.003",linestyle='--', marker='D',markersize=4)
#plt.plot(df['water_year'].tail(11), df['VPD_f'].tail(11),color="red",label="VPD (2013-2023),  p=0.04,        Tau=0.50, slope=0.008",linestyle='--', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '6.8')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(8) 
ax.set_ylabel('Wind Speed (WS) $ms^{-1}$')  # we already handled the x-label with ax1
#ax.set_ylabel('Annual mean Air Temperature (${C}^o$)')  # we already handled the x-label with ax1


#ax.set_ylim(120,180)

z1 = np.polyfit(df['water_year'], df['WS_f'], 1)
p1 = np.poly1d(z1)
plt.plot(df['water_year'], p1(df['water_year']),color='blue')

z2 = np.polyfit(df['water_year'].tail(11), df['WS_f'].tail(11), 1)
p2 = np.poly1d(z2)
plt.plot(df['water_year'].tail(11), p2(df['water_year'].tail(11)),color='red',linestyle='--')


ax.set_title("Harvard Forest EMS:US-Ha1, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=6))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()








###################################################################################################
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.errorbar(x1, y1, std, color='purple', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5, label='Pearson Correlation (r)=0.63' )
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Mean Annual VPD (kpa)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='purple',ls='--')
ax.set_title("Mann Kendall test: p<0.0001, 0.014 kpa increase per year \n VPD= Year * 0.003 + 0.33",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
ax.set_ylim(0.2,0.6)
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#######################################################################
annual_ET=df
##############################################################################################################
# remove 2010

df=df.drop(17)

annual_ET=df

x1=df['water_year']
y1=df['ET'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)

std=df['ET'].std()

plt.rcParams['figure.figsize'] = (7, 3.3)
fig, ax = plt.subplots()

plt.plot(x1, y1, color='black', ls='-',marker='o',markersize=5, label='Pearson Correlation (r)=0.30' )
plt.fill_between(x1, y1-76.87,y1+76.87, color='lightgrey')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='black',ls='--')
ax.set_ylim(200,800)
ax.set_title("Harvard Forest EMS Tower: US-Ha1, Mann Kendall test: p=0.2",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#######################################################################################################

# split timeseries
df=annual_ET
mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2008)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ET1=df

df=annual_ET
mask = (df['water_year'] >= 2009) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ET2=df


df=annual_ET1
x1=df['water_year']
y1=df['ET'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)

std=df['ET'].std()

plt.rcParams['figure.figsize'] = (3.5, 3.3)
fig, ax = plt.subplots()
plt.plot(x1, y1, color='black', ls='-',marker='o',markersize=5, label='Pearson Correlation (r)=-0.7' )
plt.fill_between(x1, y1-64.9,y1+64.9, color='lightgrey')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='black',ls='--')
ax.set_ylim(200,800)
ax.set_title("US-Ha1, Mann Kendall test: p=0.017 \n 25.4 mm decline per year \n ET= Year * -9.17 + 552.6",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

###########################################################################################################

df=annual_ET2
x1=df['water_year']
y1=df['ET'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)

std=df['ET'].std()

plt.rcParams['figure.figsize'] = (3.5, 3.3)
fig, ax = plt.subplots()
plt.plot(x1, y1, color='black', ls='-',marker='o',markersize=5, label='Pearson Correlation (r)=0.13' )
plt.fill_between(x1, y1-76.87,y1+76.87, color='lightgrey')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='black',ls='--')
ax.set_ylim(200,800)
ax.set_title("US-Ha1, Mann Kendall test: p=0.9",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#############################################################################################################

### do this step for GPP
#df=EMS_daily_ET
# get this from PET code 
df=daily_ems_pet
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).sum()
df=df.reset_index()

mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))




plt.plot(df["GPP_g"])


df["WUE"]=(df["GPP_g"]/df["ET_f"])

x1=df['water_year']
y1=df['WUE'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)

#std=df['WUE'].std()


# peaked at 2008

plt.rcParams['figure.figsize'] = (4, 4)
fig, ax = plt.subplots()
plt.plot(x1, y1, color='crimson', ls='-',marker='o',markersize=5 )
#plt.errorbar(x1, y1, color='crimson', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5 )

#plt.fill_between(x1, y1-76.87,y1+76.87, color='lightgrey')
ax.tick_params(axis='y', labelcolor='black')
#ax.legend()
#ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual WUE (g C per kg H2O)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
#z1 = np.polyfit(x1, y1, 1)
#p1 = np.poly1d(z1)
#plt.plot(x1, p1(x1),color='black',ls='--')
#ax.set_ylim(1,6)
#ax.set_title("US-Ha1: Annual water use efficiency (WUE)",fontsize=8)
#ax.set_title("US-Ha1: Growing season water use efficiency (WUE)",fontsize=8)
ax.set_title("US-Ha1: Nongrowing season water use efficiency (WUE)",fontsize=8)

#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 10)
ax.tick_params(axis = 'y', which = 'major', labelsize =10)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#########################################################################################################

df=EMS_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).mean()
df=df.reset_index()

mask = (df['water_year'] >= 1993) & (df['water_year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))


x1=df['water_year']
y1=df['VPD'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)
std=df['VPD'].std()

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.errorbar(x1, y1, std, color='purple', ls='-', capsize=2, capthick=0.5,marker='o',markersize=5, label='Pearson Correlation (r)=0.63' )
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Mean Annual VPD (kpa)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='purple',ls='--')
ax.set_title("Mann Kendall test: p<0.0001, 0.014 kpa increase per year \n VPD= Year * 0.003 + 0.33",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
ax.set_ylim(0.2,0.6)
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################

df=EMS_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).mean()
df=df.reset_index()



x1=df['water_year']
y1=df['PAR'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)
std=df['PAR'].std()

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.errorbar(x1, y1, std, color='green', ls='-', capsize=2, capthick=0.5,marker='+',markersize=5, label='Pearson Correlation (r)=0.79' )
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Mean Annual PAR (µmol $m^{-2}$ $s^{-1}$)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='green',ls='--')
ax.set_title("Mann Kendall test: p<0.0001, 10.86 µmol$m^{-2}$ $s^{-1}$ increase per year \n PAR= Year * 1.82 + 280.3",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
#ax.set_ylim(0.2,0.6)
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


############################################################################################################

df=EMS_hour_ET

mask = (df['Hr'] >=7) & (df['Hr'] <= 17)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df=df.groupby([df['TIMESTAMP'].dt.year]).sum() # year
df['Year'] = df["TIMESTAMP"]

df=df.groupby([df['TIMESTAMP'].dt.date]).sum() # daily
df=df.reset_index()
df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year

df=EMS_total_month_ET
df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year

df=EMS_total_month_ET

x=df['Year']
y=df['ET_f'].astype(float)

pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['ET_f'].astype(float),color="black", linestyle='--',label="all ET", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################

# seasonal
df=EMS_total_month_ET
df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
df['month'] = pd.to_datetime(df["TIMESTAMP"]).dt.month


mask = (df['month'] >=4) & (df['month'] <= 4)
df = df.loc[mask]
df.index = np.arange(0, len(df))


#seasonal

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


seasonal_EMS=df
EMS_total_month_ET=df


df=seasonal_EMS
mask = (df['season_nam'] =='winter')
df = df.loc[mask]
df.index = np.arange(0, len(df))
winter_EMS=df


df=seasonal_EMS
mask = (df['season_nam'] =='spring')
df = df.loc[mask]
df.index = np.arange(0, len(df))
spring_EMS=df

df=seasonal_EMS
mask = (df['season_nam'] =='summer')
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer_EMS=df

df=seasonal_EMS
mask = (df['season_nam'] =='fall')
df = df.loc[mask]
df.index = np.arange(0, len(df))
fall_EMS=df

df=fall_EMS


x=df['Year']
y=df['ET_f'].astype(float)

pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['ET_f'].astype(float),color="black", linestyle='--',label="ET", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))
#ax.set_title("winter total ET: corr=0.51",fontsize=15)
#ax.set_title("spring total ET: corr=0.13",fontsize=15)
#ax.set_title("summer total ET: corr=0.008",fontsize=15)
#ax.set_title("Fall total ET: corr=0.065",fontsize=15)
ax.set_title("total monthly ET: corr=0.05",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

###################################################################################################
# interannual variability 

df=EMS_total_annual_ET
mask = (df['TIMESTAMP'] >= '1993-01-01') & (df['TIMESTAMP'] <= '2008-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1=df

df=EMS_total_annual_ET
mask = (df['TIMESTAMP'] >= '2009-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2=df

df=EMS_daily_ET
mask = (df['Year'] ==1993)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1993=df

df=EMS_daily_ET
mask = (df['Year'] ==1994)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1994=df


df=df_2


df=EMS_total_annual_ET


x=df['Year']
y=df['ET_f'].astype(float)

pearsonr(x, y)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['ET_f'].astype(float),color="black", linestyle='--',label="ET", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x))
#ax.set_title("winter total ET: corr=0.41",fontsize=15)
#ax.set_title("spring total ET: corr=0.15",fontsize=15)
#ax.set_title("summer total ET: corr=0.02",fontsize=15)
#ax.set_title(" total annual ET (1993-2008): corr=-0.73",fontsize=12)
ax.set_title(" total annual ET (2009-2022): corr=-0.08",fontsize=12)
#ax.set_title(" total annual ET (1992-2022): corr=0.2",fontsize=12)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

















plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df_1992['TIMESTAMP'], df_1992['ET_f'].astype(float),color="black", linestyle='--',label="1992-2008")
#plt.plot(df_1993['TIMESTAMP'], df_1993['ET_f'].astype(float),color="red", linestyle='--',label="2009-2022")

#plt.plot(df_1992['ET_f'].astype(float),color="black", linestyle='--',label="ET_1992")
plt.plot(df_1['ET_f'].astype(float),color="red", linestyle='--',label="1992-2008")
plt.plot(df_2['ET_f'].astype(float),color="green", linestyle='--',label="2009-2022")
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_title("winter total ET: corr=0.41",fontsize=15)
#ax.set_title("spring total ET: corr=0.15",fontsize=15)
#ax.set_title("summer total ET: corr=0.02",fontsize=15)
ax.set_title("Fall total ET: corr=-0.01",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()













# ET conversion

# ET = (LE / lambda_) * (1 / 18) * 3600  # hourly; then sum daily
# ET = (LE / lambda_) * (1 / 18) * 1800  # half-hourly; then sum daily

# lambda
# lambda_ = (3149000 - 2370 * (Tair + 273.16)) * 1e-6

df["Hr"]=df["TIMESTAMP"].dt.hour
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

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
##########################################################################################################
## focus on summer

back_up_EMS=df

df=back_up_EMS

##########################################################################################################

# bring hemlock tower

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")
# data is hourly
df=pd.read_csv("HEM_ustarfiltered_thru2022.csv")  # 2005-2022, # of days 11688

df.head(10)
df.tail(10)
#########################################################################################################

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")
# data is hourly
df=pd.read_csv("HEM_ustarfiltered_thru2022.csv")  # 2005-2022, # of days 11688
back=df      
df.head(10)
df.tail(10)
idx=pd.date_range(start='01/01/2005 00:00:00', periods=315552, freq='30T')  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df


#index is correct date.TIMESTAMP date mess up for missing values
# so let remove messed up timestamp column
df=df.iloc[:,1:98]
#df=df.astype(float) 
df.head(10)
df.tail(10)
temp=df

df=df.rename_axis('TIMESTAMP').reset_index()
df["date"] = pd.to_datetime(df.TIMESTAMP)


###########################################################################################################
df['year'] = df['TIMESTAMP'].dt.year
#df['day'] = df['TIMESTAMP'].dt.day
df['day'] = df['TIMESTAMP'].dt.dayofyear

# eddy proc need  these columns in the start
df.insert(1, 'Year',df['year'])
df.insert(2, 'DoY', df['day'])
#add hour column for eddy proc
#try using zero
s=pd.Series(np.arange(0.5,24.5,0.5))
#######################################################################################################
#change this based on data size (number of days of data), 
#([s]*(#number of days)
k=pd.concat([s]*(6574),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df.insert(3,'Hour',k)
df=df.iloc[:,1:100]
df.tail(10)
#########################################################################################################


df["Rnet"]=df["netr"]

df["rH"]=df["RH_1_1_1"]

df['Tsoil']=np.nan


cols = df.columns.tolist()

df["DateTime"]=df["date"]
df["TIMESTAMP"]=df["date"]

# end date will be one day more than your desired end day
back_up_HEM=df

##################################################################################################################
reframed=pd.concat((df["DateTime"],df[cols[0]],df[cols[1]],df[cols[2]],df["NEE"],df["LE"],df["H"],df['Rg'],df["Tair"],df['Tsoil'],df["rH"],df["VPD"],df["Ustar"]),axis=1)
reframed.to_csv('gaps_HEM.csv', index=False, header=True)
df=back_up_HEM
df["Hr"]=df["TIMESTAMP"].dt.hour
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

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
##########################################################################################################

back_up_HEM=df

df=back_up_HEM


fill=pd.read_csv("filled_HEM.csv")  # 2005-2022, # of days 11688

df["LE_fill"]=fill["LE"]
df["ET_fill"]=df["LE_fill"]*((1/1000)*(1/(2.5*1000000))*(86400*1000))
back_up_HEM=df

####################################################################################################3

df= back_up_EMS
plt.plot(df["ET"])


df= back_up_HEM
plt.plot(df["ET_fill"],color='green')
plt.plot(df["ET"],color='red')


## focus on summer
df= back_up_HEM

mask = (df['season_nam']=='summer')
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer=df
df=summer

# 10, 14  : 10:00 - 2:00 pm
#7, 17  # means 7:00- 5:00 pm

mask = (df['Hr'] >=10) & (df['Hr'] <= 14)
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer_day=df
df=summer_day


df=pd.concat((df["TIMESTAMP"],df["year"],df["LE"],df["LE_fill"],df["ET"],df["ET_fill"], df["VPD"]),axis=1)
df.isnull().values.any()

df = df.dropna()

x=df['year']
y=df['ET'].astype(float)
#y=df['VPD'].astype(float)
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['year'], df['ET'].astype(float),color="black", linestyle='--',label="summer ET", marker='o')
#plt.scatter(df['year'], df['VPD'].astype(float),color="black", linestyle='--',label="summer ET", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET')  # we already handled the x-label with ax1
#ax.set_ylabel('VPD')  # we already handled the x-label with ax1

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
#ax.set_title("Fall",fontsize=15)
#ax.set_title("Spring",fontsize=15)
#ax.set_title("Summer (growing season)",fontsize=15)
#ax.set_title("winter",fontsize=15)

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

####################################################################################################



annual_summ_day=df.resample('Y', on='TIMESTAMP').mean()

df=annual_summ_day

df.to_csv('annual_summ_day.csv', index=False, header=True)

############################################################################################
df= back_up_EMS
df=pd.concat((df["TIMESTAMP"],df["year"],df["LE"],df["ET"], df["VPD"]),axis=1)
mask = (df['TIMESTAMP'] >'2005-01-01 00:00:00') & (df['TIMESTAMP'] <= '2022-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df["ems_ET"]=df["ET"]
df["ems_VPD"]=df["VPD"]
ems_et=df

df= back_up_HEM
df=pd.concat((df["TIMESTAMP"],df["year"],df["LE"],df["ET"], df["VPD"]),axis=1)
mask = (df['TIMESTAMP'] >'2005-01-01 00:00:00') & (df['TIMESTAMP'] <= '2022-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["hem_ET"]=df["ET"]
df["hem_VPD"]=df["VPD"]
hem_et=df

df=pd.concat((ems_et["TIMESTAMP"],ems_et["year"],ems_et["ems_ET"], ems_et["ems_VPD"],hem_et["hem_ET"], hem_et["hem_VPD"]),axis=1)


















mask = (df['TIMESTAMP'] >'2005-01-01 00:00:00') & (df['TIMESTAMP'] <= '2022-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

fill_pot=df







df= back_up_HEM



















#df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%Y-%m-%d %H:%M:%S')
df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%Y-%m-%d')
mask = (df['TIMESTAMP'] >='2008-01-01') & (df['TIMESTAMP'] <= '2023-09-01')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

df.head(10)
df.tail(10)
back=df
##########################################################################################################

idx=pd.date_range(start='1/1/2008', periods=5722, freq='D')  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
df.head(10)
df.tail(10)

################################################################################################
# focus on precip

back=df
df=df.rename_axis('else').reset_index()
df=pd.concat((df['TIMESTAMP'],df.prec),axis=1)
prec=df
# annual precip
#########################################################################################################

# combibe streamflow and discharge

df=pd.concat((prec,all_stream_daily),axis=1)

df.to_csv('prec_stream.csv', index=False, header=True)
#########################################################################################################

# start from here

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\climate\fisher_station_since_2001")
df=pd.read_csv("prec_stream.csv")  
df["TIMESTAMP"]= pd.to_datetime(df['TIMESTAMP']) 
df = df.drop(columns=['date'])
df_daily=df
df=df.resample('Y', on='TIMESTAMP').sum()
df=df.rename_axis('TIMESTAMP').reset_index()
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
annual=df
annual['RO']=annual["nb.dis"]/annual["prec"]


df=df_daily
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

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
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
seasonal=df.groupby(['season_nam','year']).sum()
seasonal['RO']=seasonal["nb.dis"]/seasonal["prec"]


df=df_daily
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))

growing_seas=df.groupby(['month','year']).sum()

growing_seas['RO']=growing_seas["nb.dis"]/growing_seas["prec"]



df=annual
x=df['year']
y=df['nb.dis'].astype(float)
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df['year'], df['RO'].astype(float),color="blue", linestyle='--',label="RO", marker='o')
plt.plot(df['year'], df['nb.dis'].astype(float),color="blue", linestyle='--',label="prec", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('RO')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Annual (all seasons)",fontsize=15)
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

df=seasonal
df=df.reset_index()

seasonal=df
mask = (df['season_nam']=='fall')
df = df.loc[mask]
df.index = np.arange(0, len(df))
fall=df


df=seasonal
mask = (df['season_nam']=='spring')
df = df.loc[mask]
df.index = np.arange(0, len(df))
spring=df

df=seasonal
mask = (df['season_nam']=='summer')
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer=df

df=seasonal
mask = (df['season_nam']=='winter')
df = df.loc[mask]
df.index = np.arange(0, len(df))
winter=df


df=growing_seas
f

x=df['year']
y=df['prec'].astype(float)
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(fall['year'], fall['RO'].astype(float),color="blue", linestyle='--',label="Fall RO", marker='o')
#plt.plot(spring['year'], spring['RO'].astype(float),color="purple", linestyle='--',label="Spring RO", marker='o')
plt.scatter(df['year'], df['prec'].astype(float),color="green", linestyle='--',label="summer prec", marker='o')
#plt.plot(winter['year'], winter['RO'].astype(float),color="red", linestyle='--',label="winter RO", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('precip')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
#ax.set_title("Fall",fontsize=15)
#ax.set_title("Spring",fontsize=15)
ax.set_title("Summer (growing season)",fontsize=15)
#ax.set_title("winter",fontsize=15)

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

##########################################################################################################

df=df_daily
#df['RO']=df["nb.dis"]/df["prec"]
s=df.sort_values(by=['prec']).tail(286)
s.mean(axis=0)

#prec        11.126224
#nb.dis      39.970561

dis95 = df['prec']>33.65
dis95_days = dis95.groupby(df['year']).sum()

df=dis95_days
df=df.to_frame()
df=df.reset_index()


x=df['year']
y=df['prec'].astype(float)
plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(fall['year'], fall['RO'].astype(float),color="blue", linestyle='--',label="Fall RO", marker='o')
#plt.plot(spring['year'], spring['RO'].astype(float),color="purple", linestyle='--',label="Spring RO", marker='o')
plt.plot(df['year'], df['prec'].astype(float),color="blue", linestyle='--',label="days with top 5%", marker='o')
#plt.plot(winter['year'], winter['RO'].astype(float),color="red", linestyle='--',label="winter RO", marker='o')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('days with top 5% precip')  # we already handled the x-label with ax1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
#add trendline to plot
plt.plot(x, p(x))
ax.set_xlabel('Year')  # we already handled the x-label with ax1
#ax.set_title("Fall",fontsize=15)
#ax.set_title("Spring",fontsize=15)
#ax.set_title("Summer (growing season)",fontsize=15)
ax.set_title("precip",fontsize=15)

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







# Exploratory groupby note: df.groupby(['year', 'nb.dis'])
df.groupby(['station', df['date'].dt.year]).above_average.sum()




growing_seas=df.groupby(['month','year']).sum()





df=pd.concat((prec,all_stream_daily['date'],all_stream_daily['nb.stg'],all_stream_daily['nb.dis']),axis=1)


df=pd.concat((prec['TIMESTAMP'],prec['prec'],all_stream_daily['date'],all_stream_daily['nb.stg'],all_stream_daily['nb.dis']),axis=1)




os.chdir(r"C:\ammara_MD\a_harvard\Project\data\stream_discharge")
df=pd.read_csv("hf070-03-daily.csv")  
#df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%Y-%m-%d %H:%M:%S')
df['TIMESTAMP'] = pd.to_datetime(df["date"], format='%m/%d/%Y')
mask = (df['TIMESTAMP'] >='2008-01-01') & (df['TIMESTAMP'] <= '2023-09-01')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df=df.loc[~df['TIMESTAMP'].duplicated(), :] # drop data with dupliacted values 

df=df.sort_values('TIMESTAMP').drop_duplicates('TIMESTAMP')

df.head(10)
df.tail(10)
stream=df
# Incomplete exploratory concatenation removed; `stream` preserves this table.
########################################################################################################

# annual for growing season

df=summ_pre_daily
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.resample('Y', on='TIMESTAMP').sum()
summ_pre_annual=df
#df=df.rename_axis('TIMESTAMP').reset_index()


df=df.rename_axis('TIMESTAMP').reset_index()


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['TIMESTAMP'], df['prec'].astype(float),color="blue", linestyle='--',label="precip (mm)", marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Precipitation (mm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Annual (growing season)",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
#ax.locator_params(axis='x', nbins=6)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################
# daily 
# run the whole for the daily 
df=summ_pre_daily

df.set_index(('TIMESTAMP'), inplace=True)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['prec'],color="blue", linestyle='--',label="precip (mm)")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Precipitation (mm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("daily (growing season)",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,40)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

summ_pre_daily.to_csv('summ_pre_daily.csv', index=False, header=True)
###########################################################################################################

# make data categories 

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\stream_discharge")
summ_pre_daily=pd.read_csv("summ_pre_daily.csv") 
summ_stream_daily=pd.read_csv("summ_stream_daily.csv") 
df_pre=summ_pre_daily
df_stre=summ_stream_daily
# combine data
df=pd.concat((df_pre,df_stre),axis=1)
df=df.drop('month', axis=1)
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month

df["prec_-1"]=df['prec'].shift(3)

mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_daily_summer=df
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year

df=df.groupby(['month','year']).mean()

month_clim=df


scipy.stats.pearsonr(df['prec'], df['nb.dis'])

df["prec_-3"]=df['prec'].shift(3)

df1=df.dropna() 

scipy.stats.pearsonr(df1['prec_-1'], df1['nb.dis'])


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.scatter(df['prec'], df['nb.dis'], label='pearson=0.82')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('discharge')  # we already handled the x-label with ax1

ax.set_xlabel('precip')  # we already handled the x-label with ax1
ax.set_title("daily (growing season)",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,40)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
#fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

summ_pre_daily.to_csv('summ_pre_daily.csv', index=False, header=True)
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
#df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df=df.groupby(['month','seasonal']).mean()
df=df.reset_index()
cli_month_summer=df

df=cli_month_summer
mask = (df['month'] >=6) & (df['month'] <= 6)
df = df.loc[mask]
df.index = np.arange(0, len(df))
cli_june=df

df=cli_month_summer
mask = (df['month'] >=7) & (df['month'] <= 7)
df = df.loc[mask]
df.index = np.arange(0, len(df))
cli_jul=df


df=cli_month_summer
mask = (df['month'] >=8) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
cli_aug=df


#df["year"]= pd.to_datetime(df["year"])





df=cli_june

plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Precip (mm)', color='black')
ax1.set_xlabel('Year', color='black')
plt.plot(df["seasonal"],df["prec"],linestyle="-",label="Precipitation",color='blue', marker='o')

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax1.locator_params(axis='x', nbins=6)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(15))
#ax1.set_ylim(2,8)
ax1.set_title("monthly (growing season)",fontsize=15)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Discharge (Ls-1)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["seasonal"],df["nb.dis"],linestyle="--", color='black',label="Nelson big (discharge)", marker='o')
#ax2.scatter(df["seasonal"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
#ax2.set_ylim(2,8)
#ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


















plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(cli_june['seasonal'],cli_june['MOD16A2GF_061_ET_500m'],color="blue", linestyle='--',label="June", marker='o')
#.plot(cli_jul["seasonal"].astype(float),cli_jul['MOD16A2GF_061_ET_500m'].astype(float),color="green", linestyle='--',label="Jul", marker='o')
#plt.plot(cli_aug["seasonal"].astype(float),cli_aug['MOD16A2GF_061_ET_500m'].astype(float),color="purple", linestyle='--',label="aug", marker='o')

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


df_stre=summ_stream_daily

# combine data

df=pd.concat((df_pre,df_stre),axis=1)

df['pre_adj']= df['prec'].replace(0,1) 

df["norm"]=df["nb.dis"]/df["pre_adj"]


# make a plot of streamflow and precip together to see on same plot to track peaks
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])

plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Precip (mm)', color='black')
ax1.set_xlabel('Year', color='black')
plt.plot(df["TIMESTAMP"],df["prec"],linestyle="-",label="Precipitation",color='blue')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax1.locator_params(axis='x', nbins=6)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(15))
ax1.set_ylim(0,150)
ax1.set_title("daily (growing season)",fontsize=15)


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Discharge (Ls-1)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df["nb.dis"], color='black',label="Nelson big (discharge)")
#ax2.scatter(df["TIMESTAMP"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,120)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

########################################################################################################

# plot for normalized ET


plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Precip (mm)', color='black')
ax1.set_xlabel('Year', color='black')
plt.plot(df["TIMESTAMP"],df["norm"],linestyle="--",label="normalized streamflow",color='black')
plt.plot(df["TIMESTAMP"],df["nb.dis"],linestyle="-",label="streamflow",color='blue')

ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax1.locator_params(axis='x', nbins=6)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(15))
#ax1.set_ylim(0,150)
ax1.set_title("daily (growing season)",fontsize=15)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
############################################################################################################

# rating curve for growing season 

######################################################################################################
# rating curve by each year
back=df
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
back=df
mask = (df['year'] >=2008) & (df['year'] <= 2011)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2008=df
df=df_2008
sort1 = np.sort(df['nb.dis'])[::-1]
exceedence1 = np.arange(1.,len(sort1)+1) / len(sort1)


df=back
mask = (df['year'] >=2012) & (df['year'] <= 2014)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2009=df
df=df_2009
sort2 = np.sort(df['nb.dis'])[::-1]
exceedence2 = np.arange(1.,len(sort2)+1) / len(sort2)

df=back
mask = (df['year'] >=2015) & (df['year'] <= 2017)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2010=df
df=df_2010
sort3 = np.sort(df['nb.dis'])[::-1]
exceedence3 = np.arange(1.,len(sort3)+1) / len(sort3)

df=back
mask = (df['year'] >=2018) & (df['year'] <= 2020)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2011=df
df=df_2011
sort4 = np.sort(df['nb.dis'])[::-1]
exceedence4 = np.arange(1.,len(sort4)+1) / len(sort4)

df=back
mask = (df['year'] >=2021) & (df['year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2012=df
df=df_2012
sort5 = np.sort(df['nb.dis'])[::-1]
exceedence5 = np.arange(1.,len(sort5)+1) / len(sort5)



plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(exceedence1*100, sort1,color="green",label="2008-2011")
plt.plot(exceedence2*100, sort2,color="red",label="2012-2014")
plt.plot(exceedence3*100, sort3,color="purple",label="2015-2017")
plt.plot(exceedence4*100, sort4,color="black",label="2018-2020")
plt.plot(exceedence5*100, sort5,color="blue",label="2021-2023")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 

ax.set_xlabel("Exceedence [%]")  # we already handled the x-label with ax1
ax.set_title("Flow rating curve",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.set_ylabel(' Discharge (L/s)')  # we already handled the x-label with ax1
#ax.set_ylim(0,40)

plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################

# adjust rating curve grouping



######################################################################################################
# rating curve by each year
df=back
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
back=df
mask = (df['year'] >=2008) & (df['year'] <= 2014)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2008=df
df=df_2008
sort1 = np.sort(df['nb.dis'])[::-1]
exceedence1 = np.arange(1.,len(sort1)+1) / len(sort1)


df=back
mask = (df['year'] >=2015) & (df['year'] <= 2018)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2009=df
df=df_2009
sort2 = np.sort(df['nb.dis'])[::-1]
exceedence2 = np.arange(1.,len(sort2)+1) / len(sort2)

df=back
mask = (df['year'] >=2019) & (df['year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2010=df
df=df_2010
sort3 = np.sort(df['nb.dis'])[::-1]
exceedence3 = np.arange(1.,len(sort3)+1) / len(sort3)



plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(exceedence1*100, sort1,color="green",label="2008-2014")
#plt.plot(exceedence2*100, sort2,color="purple",label="2015-2018")
#plt.plot(exceedence3*100, sort3,color="blue",label="2019-2023")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 

ax.set_xlabel("Exceedence [%]")  # we already handled the x-label with ax1
ax.set_title("Flow rating curve",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.set_ylabel(' Discharge (L/s)')  # we already handled the x-label with ax1
#ax.set_ylim(0,40)

plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#####################################################################################################

# search for signals in monthly or biweekly data

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\stream_discharge")
summ_pre_daily=pd.read_csv("summ_pre_daily.csv") 

summ_stream_daily=pd.read_csv("summ_stream_daily.csv") 

df_pre=summ_pre_daily

df_stre=summ_stream_daily

# combine data

df=pd.concat((df_pre,df_stre),axis=1)

# group by month
s=df.groupby(['month','year']).sum()






################################################################################################

### rat9ing curve for two points

df=back
mask = (df['year'] >=2008) & (df['year'] <= 2015)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1=df
df=df_1
sort1 = np.sort(df['nb.dis'])[::-1]
exceedence1 = np.arange(1.,len(sort1)+1) / len(sort1)


df=back
mask = (df['year'] >=2015) & (df['year'] <= 2023)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2=df
df=df_2
sort2 = np.sort(df['nb.dis'])[::-1]
exceedence2 = np.arange(1.,len(sort2)+1) / len(sort2)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(exceedence1*100, sort1,color="green",label="2008-2015")
plt.plot(exceedence2*100, sort2,color="red",label="2016-2023")

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 

ax.set_xlabel("Exceedence [%]")  # we already handled the x-label with ax1
ax.set_title("Flow rating curve",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.set_ylabel(' Discharge (L/s)')  # we already handled the x-label with ax1
#ax.set_ylim(0,40)

plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################################

## get seasonal data

df=all_data_15
back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
# 1: winter  12,1,2
# 2 spring  march- may 3,4,5
# 3 summer   june-aug 6,7,8
# 4 fall   sep-nov 9,10,11

def season_df(df):
    
    if (df['month'] ==12) or (df['month'] ==1) or (df['month'] ==2):
        return '1'
    
    elif (df['month'] ==3) or (df['month'] ==4) or (df['month'] ==5):

        return '2'

    elif (df['month'] ==6) or (df['month'] ==7) or (df['month'] ==8):
        return '3'

    elif (df['month'] ==9) or (df['month'] ==10) or (df['month'] ==11):
        return '4'
    
df['season'] = df.apply(season_df, axis = 1)    
 

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


key=['year','season_nam']
s=df.groupby( key).mean()


s=s.reset_index()

def seasonal(df):
    
    if (df['season_nam'] =='winter'):
        return '1'
    
    elif (df['season_nam'] =='spring'):

        return '2'

    elif (df['season_nam'] =='summer'):
        
        return '3'

    elif (df['season_nam'] =='fall'):
        
        return '4'



s['seasonal'] = s.apply(seasonal, axis = 1) 


s.to_csv('seasonal_annual.csv', index=False, header=True)


####################################################################################################

## seasonal data
# fall  dec-feb
# spring  march- may
#summer   june-aug
#winter   sep-nov

df=all_data_15
back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=2) & (df['month'] <= 12)
df = df.loc[mask]
df.index = np.arange(0, len(df))
fall=df

df=back
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=3) & (df['month'] <= 5)
df = df.loc[mask]
df.index = np.arange(0, len(df))
spring=df

df=back
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=6) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
summ=df

df=back
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=9) & (df['month'] <= 11)
df = df.loc[mask]
df.index = np.arange(0, len(df))
win=df



#############################################################################################
# sum season by year
########################################################################################


df=summ
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()

df.set_index(('TIMESTAMP'), inplace=True)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label="nelson Big weir (discharge)",marker='o')


ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
#ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1
ax.set_ylabel('Stage (cm), Discharge (L/s)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Summer yearly mean",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,40)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


##########################################################################################333

# average by year fall
# run the code again to find seasonal data. 
df=fall
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()

df.set_index(('TIMESTAMP'), inplace=True)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
#ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1
ax.set_ylabel('Stage (cm), Discharge (L/s)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,40)

#ax.set_title("summer yearly mean",fontsize=15)
ax.set_title("fall yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


##############################################################################################################



# average by year winter
# run the code again to find seasonal data. 

df=win
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()


df.set_index(('TIMESTAMP'), inplace=True)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()

plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')


ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
ax.set_ylabel('Stage (cm), Discharge (L/s)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,40)

#ax.set_title("summer yearly mean",fontsize=15)
ax.set_title("winter yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#####################################################################################

# average by year
# run the code again to find seasonal data. 

df=spring
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()


df.set_index(('TIMESTAMP'), inplace=True)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("spring yearly mean",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

ax.set_ylim(0,40)
ax.set_ylabel('Stage (cm), Discharge (L/s)')  # we already handled the x-label with ax1

#ax.set_title("summer yearly mean",fontsize=15)
#ax.set_title("spring yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##########################################################################################333

# yearly mean by june



df=all_data_15

back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=4) & (df['month'] <= 4)
df = df.loc[mask]
df.index = np.arange(0, len(df))
april=df
df.set_index(('TIMESTAMP'), inplace=True)

plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
#plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("april",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

ax.set_ylim(0,30)
#ax.set_ylim(0,70)
#ax.set_ylabel('Stage (cm)')  # we already handled the x-label with ax1
#ax.set_ylabel('Discharge (L/s)')  # we already handled the x-label with ax1
#ax.set_title("summer yearly mean",fontsize=15)
#ax.set_title("spring yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################################

df=all_data_15
back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 5)
df = df.loc[mask]
df.index = np.arange(0, len(df))
may=df
df.set_index(('TIMESTAMP'), inplace=True)




plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
#ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1
ax.set_ylabel('Discharge (L/s)')  # we already handled the x-label with ax1
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("May",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,30)
ax.set_ylim(0,70)
#ax.set_ylabel('Stage (cm)')  # we already handled the x-label with ax1
ax.set_ylabel('Discharge (L/s)')  # we already handled the x-label with ax1
#ax.set_title("summer yearly mean",fontsize=15)
#ax.set_title("spring yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


#######################################################################################################

########################################################################################################

df=all_data_15
back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=6) & (df['month'] <= 6)
df = df.loc[mask]
df.index = np.arange(0, len(df))
jun=df
df.set_index(('TIMESTAMP'), inplace=True)



plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
#ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1
ax.set_ylabel('Discharge (L/s)')  # we already handled the x-label with ax1
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("Jun",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,30)
ax.set_ylim(0,70)
#ax.set_ylabel('Stage (cm)')  # we already handled the x-label with ax1
#ax.set_title("summer yearly mean",fontsize=15)
#ax.set_title("spring yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

########################################################################################################

df=all_data_15
back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=8) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
aug=df
df.set_index(('TIMESTAMP'), inplace=True)



plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
#ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1
#ax.set_ylabel('Discharge (L/s)')  # we already handled the x-label with ax1
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("aug",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,30)
ax.set_ylim(0,70)
ax.set_ylabel('Stage (cm)')  # we already handled the x-label with ax1
#ax.set_title("summer yearly mean",fontsize=15)
#ax.set_title("spring yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

############################################################################################################


df=all_data_15
back=df
df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=9) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))
sep=df
df.set_index(('TIMESTAMP'), inplace=True)



plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
#plt.plot(df['nb.stg'],color="green", linestyle='--',label="nelson Big weir (stage)",marker='o')
plt.plot(df['nb.dis'],color="red", linestyle='--',label=" nelson Big weir (discharge)",marker='o')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
#ax.set_ylabel('stage (cm)')  # we already handled te x-label with ax1
ax.set_ylabel('Discharge (L/s)')  # we already handled the x-label with ax1
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.set_title("sep",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(0,30)
ax.set_ylim(0,70)
#ax.set_ylabel('Stage (cm)')  # we already handled the x-label with ax1
#ax.set_title("summer yearly mean",fontsize=15)
#ax.set_title("spring yearly mean",fontsize=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()













#########################################################################################################
# extra 31 days 

def get_full_months(start, end):
    """ Returns the number of months including
        non-full months. """

    if end.year == start.year:
        return end.month - start.month + 1
    
    months = (end.year - start.year - 1) * 12
    return (12 - start.month + 1) + end.month + months































#########################################################################################
