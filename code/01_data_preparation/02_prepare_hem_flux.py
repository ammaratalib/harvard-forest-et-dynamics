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

#2023 data

os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")

# data is half hourly
#df=pd.read_csv("hem_hourlyustarfiltnongf.csv")  # 
df=pd.read_csv("US-Ha2_HH_200401010000_202401010000.csv")
back=df      
df.head(10)
df.tail(10)
idx=pd.date_range(start='01/01/2004 00:00:00', periods=350640, freq='30T')  # half hourly. 
df.index = pd.DatetimeIndex(df.datetime)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df


#index is correct date.TIMESTAMP date mess up for missing values
# so let remove messed up timestamp column
df=df.iloc[:,1:160]
#df=df.astype(float) 
df.head(10)
df.tail(10)
temp=df
df=df.rename_axis('TIMESTAMP').reset_index()
df = df.iloc[1:] 

###########################################################################################################
df["date"] = pd.to_datetime(df.TIMESTAMP)
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
k=pd.concat([s]*(7305),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df.insert(3,'Hour',k)
df=df.iloc[:,1:160]
df.tail(10)
k=df.columns
df['TIMESTAMP']=df['datetime']# don't use date time. date time has nans 


df['TIMESTAMP']=pd.to_datetime(df['TIMESTAMP'])
all_data=df
mask = (df['TIMESTAMP'] >='2023-01-01 00:00:00') & (df['TIMESTAMP'] <= '2023-12-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
back=df
#########################################################################################################

df['WD'] =df.apply(lambda row: row['WD_2_1_1'] if pd.isnull(row['WD_1_1_1']) else row['WD_1_1_1'], axis=1)
df['Ustar'] =df.apply(lambda row: row['USTAR_2_1_1'] if pd.isnull(row['USTAR_1_1_1']) else row['USTAR_1_1_1'], axis=1)
df['NEE'] =df.apply(lambda row: row['FC_2_1_1'] if pd.isnull(row['FC_1_1_1']) else row['FC_1_1_1'], axis=1)
df['H'] =df.apply(lambda row: row['H_2_1_1'] if pd.isnull(row['H_1_1_1']) else row['H_1_1_1'], axis=1)
df['RH'] =df.apply(lambda row: row['RH_2_1_1'] if pd.isnull(row['RH_1_1_1']) else row['RH_1_1_1'], axis=1)
df['RH'] =df.apply(lambda row: row['RH_2_2_1'] if pd.isnull(row['RH']) else row['RH'], axis=1)

df['WS'] =df.apply(lambda row: row['WS_2_1_1'] if pd.isnull(row['WS_1_1_1']) else row['WS_1_1_1'], axis=1)
df['PA']=df['PA_2_1_1']
df['Tair'] =df.apply(lambda row: row['TA_2_1_1'] if pd.isnull(row['TA_1_1_1']) else row['TA_1_1_1'], axis=1)
df['Tair'] =df.apply(lambda row: row['TA_5_1_1'] if pd.isnull(row['Tair']) else row['Tair'], axis=1)

df['VPD']=df['VPD_2_1_1'] 
es=0.6108*np.exp((17.27*df["Tair"])/(df["Tair"]+237.3))   # in kpa
ea=df["RH"]/100*es    ### same as multiplying with es
df["VPD_f"]=(es-ea)*10
df['VPD'] =df.apply(lambda row: row['VPD_f'] if pd.isnull(row['VPD']) else row['VPD'], axis=1)

df['Rnet']=df['SW_IN_2_1_1']-df['SW_OUT_2_1_1']+df['LW_IN_2_1_1']-df['LW_OUT_2_1_1']
df['NETRAD']=df.apply(lambda row: row['Rnet'] if pd.isnull(row['NETRAD_2_1_1']) else row['NETRAD_2_1_1'], axis=1)
df['Rg']=df['SW_IN_2_1_1']

df['PPFD']=df.apply(lambda row: row['PPFD_IN_4'] if pd.isnull(row['PPFD_IN_3']) else row['PPFD_IN_3'], axis=1)
df['LE']=df.apply(lambda row: row['LE_2_1_1'] if pd.isnull(row['LE_1_1_1']) else row['LE_1_1_1'], axis=1)
df['LE']=df.apply(lambda row: row['LE_2_1_2'] if pd.isnull(row['LE']) else row['LE'], axis=1)


df['Co2']=df.apply(lambda row: row['CO2_2_1_1'] if pd.isnull(row['CO2_1_1_1']) else row['CO2_1_1_1'], axis=1)
df['H2o']=df.apply(lambda row: row['H2O_2_1_1'] if pd.isnull(row['H2O_1_1_1']) else row['H2O_1_1_1'], axis=1)

varib=df
df=varib

df=pd.concat((df['TIMESTAMP'],df['DoY'],df['Year'], df['Hour'],df['CO2_1_1_1'],df['H2O_1_1_1'],df['WD'],
df['Ustar'], df['TS_1'],df['NEE'],df['H'],df['RH'],df['WS'],df['PA'],df['Tair'],df['VPD'],
df['Rg'],df['PPFD'],df['NETRAD'],df['LE'],df['sos'],df['eos']),axis=1)

df['DateTime']=df['TIMESTAMP']

#df["GPP"]=df["GPP_f"]
df['PAR']=df['PPFD']

back_up_HEM=df
back_up_HEM.to_csv('back_up_HEM.csv', index=False, header=True)

df.to_csv('gaps_ML_HEM.csv', index=False, header=True)

back=df

df=back_up_HEM

reframed=pd.concat((df['DateTime'],df['Year'],df['Hour'],df['DoY'],df['NEE'],df['LE'],df['H'],df['Rg'],df['Tair'],df['VPD'],df['Ustar'],df['PAR'],df['RH'],df['PA'],df['NETRAD'],df['WS']),axis=1)

reframed.to_csv('gaps_HEM.csv', index=False, header=True)
df=back_up_HEM

df['lambda']=(3149000-2370*(df['Tair']+273.16))*1e-6

df.to_csv('canopy_HEM_calc.csv', index=False, header=True)
##################################################################################################################

df=pd.read_csv("gaps_HEM.csv") 

# brig filled data
os.chdir(r"C:\ammara_MD\a_harvard\Project\HF_paper")

fill_hem=pd.read_csv("filled_HEM.csv") # to use fixed LE used file " filled_EMS to copy LE"
back_up_HEM=pd.read_csv("back_up_HEM.csv") 
df['TIMESTAMP']=back_up_HEM["TIMESTAMP"]
fill_hem=df
#


plt.plot(df['LE'],color='green')
reddy=fill_hem
#df["LE1"] = np.where(df["LE"]<-200, np.NaN, df["LE"])
#df["LE2"] = np.where(df["LE1"]>800, np.NaN, df["LE1"])
#df["LE"]=df["LE2"]
plt.plot(df['LE'],color='green')
df['lambda']=(3149000-2370*(df['Tair']+273.16))*1e-6
plt.plot(df['lambda'],color='green')


df['ET']=(df['LE']/df['lambda'])*(1/1e6)*1800
df["GPP_g"]=(df["GPP"])*((12/10**6)*1800)
plt.plot(df['ET'],color='red')
df["NEE_g"]=(df["NEE"])*((12/10**6)*1800)

#df=pd.concat((df["TIMESTAMP"],df["LE_f"],df['ET_f'],df['NETRAD'],df['RH'],df['WS'],df['PA'],df['Tair_f'],df['VPD_f']),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['Year'] = df['TIMESTAMP'].dt.year
df.isnull().values.any()
#df1 = df[df.isna().any(axis=1)]
#df=df.dropna()
df["Hr"]=df['TIMESTAMP'].dt.hour
HEM_hour_ET=df

df=HEM_hour_ET
###################################################################################################
df=HEM_hour_ET
#df["Rnet1"] = np.where(df["NETRAD"]<-500, np.NaN, df["NETRAD"])
#df["Rnet2"] = np.where(df["Rnet1"]>1000, np.NaN, df["Rnet1"])
df["Rnet"]=df["NETRAD"]
#mask = (df['TIMESTAMP'] >='2004-01-01 00:00:00') & (df['TIMESTAMP'] <= '2022-12-31 23:45:00')
# end date will be one day more than your desired end day
#df = df.loc[mask]
#df.index = np.arange(0, len(df))

#####################################################################################################
back2=df
df = df.iloc[: , 1:] # remove column
df["datetime"]=df['TIMESTAMP']
df["PPFD"]=df['PAR']
#df=df.fillna('NA')

#############################################################################################################
# hourly EMS gap filled data
df.to_csv('HEM_ustfilt_2023_filled.csv', index=False, header=True)

##########################################################################################################
# part for machine learning 

#hem_NEELEfillpartMR_9122_all
#df=pd.read_csv("hem_NEELEfillpartMR_9122_all.csv") gpp data
df=pd.read_csv("HEM_ustfilt_2023_filled.csv") 
#plt.plot(df['Rnet'],color='red')

#############################################################################################################
#df.columns = ['Year', 'DoY', 'Hour', 'NEE_f', 'GPP_f','LE_f', 'H_f', 'Rg_f', 'Tair_f', 'VPD_f',
#'PAR_f', 'RH_f', 'PA_f', 'NETRAD_f', 'WS_f', 'TIMESTAMP','LE1', 'LE2','lambda_f', 'ET_f','GPP_g', 'Hr', 'Rnet1','Rnet2',
#'Rnet_f','datetime', 'PPFD_f']          

df.columns = ['Year', 'DoY', 'Hour', 'NEE_f', 'GPP_f','LE_f', 'H_f', 'Rg_f', 'Tair_f', 'VPD_f',
'PAR_f', 'RH_f', 'PA_f', 'NETRAD_f', 'WS_f', 'TIMESTAMP','lambda_f', 'ET_f','GPP_g', 'NEE_g','Hr',
'Rnet_f','datetime', 'PPFD_f']          




#mask= (back['TIMESTAMP'] >='2004-01-01 00:00:00') & (back['TIMESTAMP'] <= '2022-12-31 23:45:00')
# end date will be one day more than your desired end day
#back_m = back.loc[mask]
#back_m.index = np.arange(0, len(back_m))
back_m=back

#mask= (df['TIMESTAMP'] >='2004-01-01 00:00:00') & (back['TIMESTAMP'] <= '2022-12-31 23:45:00')
# end date will be one day more than your desired end day
#df = df.loc[mask]
#df.index = np.arange(0, len(df))

#back_m1=df # filled data

######################################################################################################

# check what variables are important 


df=pd.concat((df["TIMESTAMP"],df['DoY'],df["Year"], df['Hour'],back_m['CO2_1_1_1'],back_m['H2O_1_1_1'],back_m["WD"],
back_m['Ustar'], back_m['FC_1_1_1'], back_m['TS_PI_1'],df['NEE_f'],df['H_f'],
df['RH_f'],df['WS_f'],df['PA_f'],df['Tair_f'],df['lambda_f'],df['VPD_f'],df['GPP_g'],df['NEE_g'],df['Rg_f'],df['PAR_f'],
df['NETRAD_f'],df['LE_f'],df['ET_f']),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
back3=df
######################################################################################################

#daily ET                                       
df=df.drop(['TIMESTAMP'], axis=1)
#df=df.interpolate()
df['TIMESTAMP']=back3['TIMESTAMP']
##mean
s=df.groupby([df['TIMESTAMP'].dt.date]).mean()
s=s.reset_index()
##sum
s_et=df.groupby([df['TIMESTAMP'].dt.date]).sum()
s_et=s_et.reset_index()

plt.plot(s_et['ET_f'],color='red')
plt.plot(s['Rg_f'],color='red')

df.isnull().values.any()
#s["NEE_f"]=s_et['NEE']
#s["H_f"]=s_et['H']
#s["GPP_s"]=s_et['GPP']
#s["LE_f"]=s_et['LE']


df=pd.concat((s["TIMESTAMP"],s['DoY'],s["Year"], s['Hour'],s['CO2_1_1_1'],s['H2O_1_1_1'],s["WD"],
s['Ustar'], s['FC_1_1_1'], s['TS_PI_1'],s_et['NEE_f'],s_et['H_f'],
s['RH_f'],s['WS_f'],s['PA_f'],s['Tair_f'],s['lambda_f'],s['VPD_f'],s_et['GPP_g'],s_et['NEE_g'],s['Rg_f'],s['PAR_f'],
s['NETRAD_f'],s_et['LE_f'],s_et['ET_f']),axis=1)



#df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
HEM_daily_ET=df
df.isnull().values.any()


df.to_csv('PET_HEM_input.csv', index=False, header=True)
HEM_daily_ET=df
#df.to_csv('daily_EMS_ML.csv', index=False, header=True)

#######################################################################################################

df=HEM_daily_ET

#######################################################################################################
df=HEM_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df['Year'] = df.TIMESTAMP.dt.year

#df= df.groupby(['water_year']).sum()
#df=df.reset_index()
############################################################################################################

back4=df

#df=df.fillna(method='ffill')
#df["CO2_1_1_1"].fillna(method='backfill', inplace=True)
#df.isnull().values.any()
df.to_csv('daily_HEM_ML.csv', index=False, header=True)

plt.plot(df["ET_f"])

plt.plot(df["GPP_g"])
plt.plot(df["NEE_g"])
# pick columns that you want for ML model


df.to_csv('daily_HEM_ML.csv', index=False, header=True)

#df= df.groupby(['water_year']).sum()
#df=df.reset_index()
############################################################################################################
# start from here ammaraMD

df=pd.read_csv("daily_HEM_ML.csv") 

df=HEM_daily_ET
df.isnull().values.any()
#df=df.groupby(['water_year']).sum()
df=df.groupby(['Year']).sum()
df=df.reset_index()


mask = (df['Year'] >= 2006) & (df['Year'] <= 2012)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df


df=annual_hem_sum
mask = (df['Year'] >= 2012) & (df['Year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))


df_e_y=df

x1=df_e_y['Year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['ET_f'].astype(float)
#y1=df_e_y['NEE_g'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)


#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e_y['Year'].head(7), df_e_y['ET_f'].head(7),color="green",label="ET (2006-2012),  p=0.37, Tau=-0.33, slope=-22",linestyle='-', marker='*',markersize=7)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
plt.plot(df_e_y['Year'].tail(11), df_e_y['ET_f'].tail(11),color="purple",label="ET (2012-2022),  p=0.16, Tau=0.34,  slope=11",linestyle='-', marker='*',markersize=7)

df_e_y.loc[4:9]
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual ET (mm)')  # we already handled the x-label with ax1


ax.set_ylim(250,525)

z1 = np.polyfit(df_e_y['Year'].head(7), df_e_y['ET_f'].head(7), 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['Year'].head(7), p1(df_e_y['Year'].head(7)),color='green')


# = np.polyfit(df_e_y['Year'].loc[4:9], df_e_y['ET_f'].loc[4:9], 1)
#p1 = np.poly1d(z1)
#plt.plot(df_e_y['Year'].loc[4:9], p1(df_e_y['Year'].loc[4:9]),color='red')

plt.axvline(x=2009,linestyle='--', color='black',linewidth=1)
plt.axvline(x=2015,linestyle='--', color='black',linewidth=1)

z1 = np.polyfit(df_e_y['Year'].tail(11), df_e_y['ET_f'].tail(11), 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['Year'].tail(11), p1(df_e_y['Year'].tail(11)),color='purple')

ax.text(0.20, 0.82, 'Before HWA',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=8)

ax.text(0.48, 0.82, 'Early HWA',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=8)

ax.text(0.90, 0.82, 'Later HWA',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=8)

ax.set_title("Harvard Forest HEM:US-Ha2, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################################################3333


# box plot 

df=annual_hem_sum
#df=grow_gs_iPM
# mask precip data data from 1992-2023 
mask = (df['Year'] >= 2006) & (df['Year'] <= 2013)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
before_ET=df["ET_f"]

df=annual_hem_sum
# mask precip data data from 1992-2023 
mask = (df['Year'] >= 2014) & (df['Year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
after_ET=df["ET_f"]

df=annual_hem_sum
#df=grow_gs_iPM
# mask precip data data from 1992-2023 
mask = (df['Year'] >= 2006) & (df['Year'] <= 2013)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
before_NEE=df["NEE_g"]

df=annual_hem_sum
# mask precip data data from 1992-2023 
mask = (df['Year'] >= 2014) & (df['Year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))
#df["TIMESTAMP"]=pd.to_datetime(df["date"])
after_NEE=df["NEE_g"]


my_dict1={'Pre-During HWA':before_ET}
my_dict2={'Post HWA':after_ET}
my_dict3={'Pre-During HWA':before_NEE}
my_dict4={'Post HWA':after_NEE}


#my_dict2={'gs_iPM':after_iPM, 'gs_FG':after_FG}

#ax.set_xticklabels(['gs_iPM (1992-2008)','gs_FG (1992-2008)','gs_iPM (2009-2023)','gs_FG (2009-2023)'])
c1="C0"
c2="C2"
c3="C0"
c2="C2"




plt.rcParams['figure.figsize'] = (3.5,2.5)
fig, ax = plt.subplots()

bp1=ax.boxplot(my_dict1.values(),sym='.',positions =[1],notch=False, widths=3, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='royalblue', linewidth=0.5),boxprops=dict(facecolor='navy',edgecolor='slateblue'))

bp2=ax.boxplot(my_dict2.values(),sym='.',positions = [4],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='pink',edgecolor='salmon'),whiskerprops = dict(linestyle='--'
               ,color='indianred' , linewidth=0.5))

plt.setp(bp1['medians'][0], color='slateblue')
plt.setp(bp2['medians'][0], color='salmon')

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Total annual NEE (g-C m$^{-2}$)', color='black')  # we already handled the x-label with ax1
ax.set_title("Harvard Forest HEM:US-Ha2, Deciduous Broadleaf Forest",fontsize=7)

bp3=ax2.boxplot(my_dict3.values(),sym='.',positions =[8],notch=False, widths=3, 
patch_artist=True,whiskerprops = dict(linestyle='--'
,color='royalblue', linewidth=0.5),boxprops=dict(facecolor='navy',edgecolor='slateblue'))

bp4=ax2.boxplot(my_dict4.values(),sym='.',positions = [11],notch=False, widths=3, 
 patch_artist=True, boxprops=dict(facecolor='pink',edgecolor='salmon'),whiskerprops = dict(linestyle='--'
               ,color='indianred' , linewidth=0.5))
ax.set_ylim(250,550)

plt.setp(bp3['medians'][0], color='slateblue')
plt.setp(bp4['medians'][0], color='salmon')


plt.xticks([1, 4, 8,11], ['ET','ET','NEE','NEE'])

ax.text(0.45, 0.80, 'p=0.005',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=7)

ax.text(0.92, 0.97, 'p<0.0001',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=7)

ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Pre-during HWA:  2006-2013', 'Post HWA:            2014-2022'], loc='upper left',prop={'size': 5})
ax2.set_ylim(-750,-150)
ax.set_ylabel('Total annual ET (mm)')  # we already handled the x-label with ax1
ax.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax2.tick_params(axis = 'y', which = 'major', labelsize = 8,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 8,direction='in')
#ax.set_ylim(-0.5,1.5)
#ax.set_title("Training Sites, Potatoes: (2018-2020), n=309",fontsize=6)
#ax.set_title("whole year",fontsize=7)
ax.yaxis.label.set_size(8)
ax2.yaxis.label.set_size(8)
ax2.yaxis.label.set_size(8)
ax.xaxis.label.set_size(10) #there is no label 
#ax.locator_params(axis='y', nbins=4)
ax = plt.gca()
plt.show()
fig.savefig('testfig.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()



















######################################################################################################
df_e_y=annual_hem_sum

x1=df_e_y['Year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['NEE_g'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)




#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df_e_y['Year'], df_e_y['NEE_g'],color="blue",label="NEE (2006-2022),  p=0.003, Tau=0.53, slope=28.4",linestyle='--', marker='D',markersize=4)
#plt.plot(df_e_y['Year'].loc[4:10], df_e_y['ET_f'].loc[4:10],color="red",label="During,  p=0.30, Tau=0.19, slope=2.7",linestyle='-', marker='*',markersize=7)
#plt.plot(df_e_y['Year'].tail(11), df_e_y['ET_f'].tail(11),color="purple",label="ET (2012-2022),  p=0.16, Tau=0.34,  slope=11",linestyle='-', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Total annual NEE (g-C m$^{-2}$)')  # we already handled the x-label with ax1


ax.set_ylim(-750,-100)

z1 = np.polyfit(df_e_y['Year'], df_e_y['NEE_g'], 1)
p1 = np.poly1d(z1)
plt.plot(df_e_y['Year'], p1(df_e_y['Year']),color='blue')

# = np.polyfit(df_e_y['Year'].loc[4:9], df_e_y['ET_f'].loc[4:9], 1)
#p1 = np.poly1d(z1)
#plt.plot(df_e_y['Year'].loc[4:9], p1(df_e_y['Year'].loc[4:9]),color='red')

plt.axvline(x=2009,linestyle='--', color='black',linewidth=1)
plt.axvline(x=2015,linestyle='--', color='black',linewidth=1)

ax.text(0.20, 0.82, 'Before HWA',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=8)

ax.text(0.48, 0.82, 'Early HWA',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=8)

ax.text(0.85, 0.82, 'Later HWA',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=8)

ax.set_title("Harvard Forest HEM:US-Ha2, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#####################################################################################################








## bowen ratio

#df=pd.read_csv("gaps_HEM.csv") 
df=back

df=pd.concat((df["date"],df['DoY'],df["Year"], df['Hour'],df['H'],df['LE'],df["NETRAD"],df['sos'],df['eos']),axis=1)

df['check'] = (df['LE'].isnull()) | (df['H'].isnull())


def check2(df): 
    
    if (df['check'] == True):  
        return 'nan'
    else:
        return '1' 


df['check2'] = df.apply(check2, axis = 1)
df["date"] = pd.to_datetime(df.date)
back2=df

df=back2.groupby([back2['date'].dt.date]).sum()
s=back2.groupby([back2['date'].dt.date]).mean()

df["sos"]=s["sos"]
df["eos"]=s["eos"]
df=df.reset_index()
df["B"]=df["H"]/df["LE"]

mask = (df['check'] <=18) 
df = df.loc[mask]
df.index = np.arange(0, len(df))



fil=df






# check column, 48 means there are 48 nans, so mask if check <=18

HEM_daily=df

df=HEM_daily
    
df["TIMESTAMP"]=df["date"]


df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
df["Year"]= pd.to_datetime(df['TIMESTAMP']).dt.year
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['DoY'] = df['TIMESTAMP'].dt.dayofyear
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)
df['season_nam'] = df.apply(season_nam, axis = 1)

mask = (df['B'] <= 5) & (df['B'] >= 0)
df = df.loc[mask]
df.index = np.arange(0, len(df))



#######################################################################################################
# growing, non growing
mask = (df['season_nam'] =='spring') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
#GS=df
#NGS=df

###############################################################################################

#df=GS
df=df.groupby(['Year']).mean()
df=df.reset_index()

month=df

mask = (df['Year'] >= 2006) & (df['Year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))



month=df


#####################################################################################################
## mask for each month
df=month

mask = (df['month'] ==12) 
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df
#########################################################################################
# mask by season

def season_nam(df):    
    if (df['month'] ==9) or (df['month'] ==10) or (df['month'] ==11):
        return 'fall'
    
    elif (df['month'] ==3) or (df['month'] ==4) or (df['month'] ==5):

        return 'spring'



mask = (df['season_nam'] =='fall') 
df = df.loc[mask]
df.index = np.arange(0, len(df))


df=df.groupby(['Year']).mean()
df=df.reset_index()





df=fall


#########################################################################################3
x1=df['Year']
#x1=df['month']
y1=df['B'].astype(float) # LE increasing, H increasing, B decreasing 
pearsonr(x1, y1)
mk.original_test(y1)
########################################################################################################

# by months

#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['B'],color="green",label="Bowen Ratio,  p=0.006, Tau=0.5, slope=0.06",linestyle='-', marker='*',markersize=7)

#plt.plot(df['Year'], df['B'],color="green",label="Bowen Ratio",linestyle='-', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Bowen Ration (H/LE)')  # we already handled the x-label with ax1


#ax.set_ylim(250,525)

z1 = np.polyfit(df['Year'], df['B'],1)
p1 = np.poly1d(z1)
plt.plot(df['Year'], p1(df['Year']),color='black')

# = np.polyfit(df_e_y['Year'].loc[4:9], df_e_y['ET_f'].loc[4:9], 1)
#p1 = np.poly1d(z1)
#plt.plot(df_e_y['Year'].loc[4:9], p1(df_e_y['Year'].loc[4:9]),color='red')


ax.set_title("Harvard Forest HEM:US-Ha2, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
#ax.set_xlabel('months')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()


##################################################################################################

## all years
#plt.rcParams['figure.figsize'] = (1.3, 1)
plt.rcParams['figure.figsize'] = (5, 4)
fig, ax = plt.subplots()
plt.plot(df['month'], df['B'],color="green",label="Bowen Ratio,  p=1.29*10-6, Tau=-0.47, slope=-0.07",linestyle='-', marker='*',markersize=7)

#plt.plot(df['month'], df['B'],color="green",label="Bowen Ratio",linestyle='-', marker='*',markersize=7)

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper right', fontsize = '7')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10) 
ax.set_ylabel('Bowen Ration (H/LE)')  # we already handled the x-label with ax1


#ax.set_ylim(250,525)

z1 = np.polyfit(df['month'], df['B'],1)
p1 = np.poly1d(z1)
plt.plot(df['month'], p1(df['month']),color='black')

# = np.polyfit(df_e_y['Year'].loc[4:9], df_e_y['ET_f'].loc[4:9], 1)
#p1 = np.poly1d(z1)
#plt.plot(df_e_y['Year'].loc[4:9], p1(df_e_y['Year'].loc[4:9]),color='red')


ax.set_title("Harvard Forest HEM:US-Ha2, Deciduous Broadleaf Forest",fontsize=10)
#ax.set_title("US-Ha1: Harvard Forest EMS Tower, Mann Kendall test: p=0.1",fontsize=2.5)
ax.set_xlabel('months')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 10,width=1)
ax.tick_params(axis = 'y', which = 'major', labelsize =10, width=1)
fig.autofmt_xdate() 
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x', rotation=0)
plt.setp(ax.spines.values(), lw=1, color='black', alpha=1);
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()



###########################################################################################################
##seasonal trend
df=temp
df["DateTime"]=pd.to_datetime(df["DateTime"])
df=df.groupby([df['DateTime'].dt.date]).mean()
gs=df.reset_index()

df=HEM_daily_ET

df["sos"]=gs["sos"]
df["eos"]=gs["eos"]

HEM_daily_ET=df

df=HEM_daily_ET


def Grow_sea(df):
    
    if (df['DoY'] >=df['sos']) and (df['DoY'] <=df['eos']):
        return 'GS'
    else:
        return 'NGS'
    
df['Grow_sea'] = df.apply(Grow_sea, axis = 1)

HEM_daily_ET=df


df=HEM_daily_ET
   
mask = (df['Grow_sea'] == 'GS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
growing=df


df=HEM_daily_ET   
mask = (df['Grow_sea'] == 'NGS') 
df = df.loc[mask]
df.index = np.arange(0, len(df))
nongrowing=df



#df=growing
df=nongrowing

df.isnull().values.any()
#df=df.groupby(['water_year']).sum()
df=df.groupby(['Year']).sum()
df=df.reset_index()
mask = (df['Year'] >= 2006) & (df['Year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_hem_sum=df


df_e_y=annual_hem_sum

x1=df_e_y['Year']
#y1=df_e_y['PET_ems'].astype(float)
y1=df_e_y['ET_f'].astype(float)
pearsonr(x1, y1)
mk.original_test(y1)

#2006-2009

#2010-2015
#2016-2022

df_e_y=annual_hem_sum

df=df_e_y
df=df.groupby(['Year']).sum()
df=df.reset_index()
mask = (df['Year'] >= 2006) & (df['Year'] <= 2012)
df = df.loc[mask]
df.index = np.arange(0, len(df))
mk.original_test(df['ET_f'])

df_e_y=annual_hem_sum

df=df_e_y
df=df.groupby(['Year']).sum()
df=df.reset_index()
mask = (df['Year'] >= 2012) & (df['Year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))
mk.original_test(df['ET_f'])
df_ear=df
#####################################################################################################

































###############################################################################################
df_e_y=annual_hem_sum

df=df_e_y

df.set_index(('Year'), inplace=True)


plt.rcParams['figure.figsize'] = (10, 5.5)
fig, ax = plt.subplots()
plt.plot(df.index,df.ET_f,'x', color="lime",markersize=8,marker='*',label="ET",mfc='none',linestyle='-')
#plt.plot(df.index,df.NEE_f,'x', color="lightcoral",markersize=10,marker='3',label="NEE",mfc='none')
ax2 = ax.twinx() 
plt.plot(df.index,df.NEE_g,'x', color="darkgreen",markersize=10,marker='*',label="NEE",mfc='none',linestyle='-')
#plt.plot(df.index,df.inso_vp_kpa,'x', color="darkorange",markersize=8,marker='v',label="WISP VP",mfc='none')


ax.tick_params(axis='y', labelcolor='green')
ax2.tick_params(axis='y', labelcolor='darkgreen')
ax.legend()
ax.legend(loc='lower left', fontsize = 'small')
ax2.legend(loc='lower right', fontsize = 'small')

ax.xaxis.label.set_size(12) # 
ax.spines['left'].set_color('green')
ax2.spines['right'].set_color('darkgreen') 

ax.tick_params(axis = 'x', which = 'major', labelsize = 12)
ax.tick_params(axis = 'y', which = 'major', labelsize = 14)
ax2.tick_params(axis = 'y', which = 'major', labelsize = 14)
ax.locator_params(axis='y', nbins=6)
ax.set_ylim([0,30])
ax2.set_ylim([0,6])

ax.set_title("Training Sites: Potatoes, (2018-2020), n=309",fontsize=15)
ax.set_ylabel('Air Temperature (\u00B0C)', color='green')  # we already handled the x-label with ax1

ax2.set_ylabel('Vapor pressure (kpa)', color='darkgreen')  # we already handled the x-label with ax1
ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.yaxis.label.set_size(14)
ax2.yaxis.label.set_size(14)

ax.text(0.30, 0.90, 'Air Temperature',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=12)

ax.text(0.30, 0.85, 'MAE = 0.66 (\u00B0C)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=12)

ax.text(0.30, 0.80, 'pbias = -0.44',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=12)

ax.text(0.30, 0.75, 'NSE = 0.97',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=12)

ax.text(0.30, 0.70, '$\mathregular{R^{2}}$ = 0.97',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=12)


ax.text(0.75, 0.90, 'Vapor Pressure',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkgreen', fontsize=12)

ax.text(0.75, 0.85, 'MAE = 0.08 (kpa)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkgreen', fontsize=12)

ax.text(0.75, 0.80, 'pbias = -3.5',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkgreen', fontsize=12)

ax.text(0.75, 0.75, 'NSE = 0.97',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkgreen', fontsize=12)

ax.text(0.75, 0.70, '$\mathregular{R^{2}}$ = 0.98',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='darkgreen', fontsize=12)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=100))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()





















































































####################################################################################################

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
#df = df.iloc[1:] 

###################################################################################################

back_m=df
df=back_m1

###########################################################################################################

# use this when GPP
#df=pd.concat((df["TIMESTAMP"],df['DoY'],df["Year"], df['Hour'],back_m['FC_1_1_1'],back_m['CO2_1_1_1'],back_m['H2O_1_1_1'],back_m["WD"],
#back_m['Ustar'], back_m['SC_2_1_1'], back_m['TS_PI_1'],df['NEE_f'],df['H_f'],
#df['RH_f'],df['WS_f'],df['PA_f'],df['Tair_f'],df['lambda_f'],df['VPD_f'],df['GPP_g'],df['Rg_f'],df['PAR_f'],
#df['NETRAD_f'],df['LE_f'],df['ET_f']),axis=1)


###########################################################################################################


df=pd.concat((df["TIMESTAMP"],df['DoY'],df["Year"], df['Hour'],back_m['CO2_1_1_1'],back_m['H2O_1_1_1'],back_m["WD"],
back_m['Ustar'], back_m['SC_2_1_1'], back_m['TS_PI_1'],df['NEE_f'],df['H_f'],
df['RH_f'],df['WS_f'],df['PA_f'],df['Tair_f'],df['lambda_f'],df['VPD_f'],df['Rg_f'],df['PAR_f'],
df['NETRAD_f'],df['LE_f'],df['ET_f']),axis=1)



df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
back3=df

                                       
df=df.drop(['TIMESTAMP'], axis=1)
df=df.interpolate()
df['TIMESTAMP']=back3['TIMESTAMP']
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


df=pd.concat((s["TIMESTAMP"],s['DoY'],s["Year"], s['Hour'],s['FC_1_1_1'],s['CO2_1_1_1'],s['H2O_1_1_1'],s["WD"],
s['Ustar'], s['TS_PI_1'],s_et['NEE_f'],s_et['H_f'],
s['RH_f'],s['WS_f'],s['PA_f'],s['Tair_f'],s['lambda_f'],s['VPD_f'],s['Rg_f'],s['PAR_f'],
s['NETRAD_f'],s_et['LE_f'],s_et['ET_f']),axis=1)



#df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
HEM_daily_ET=df

df.isnull().values.any()


df.to_csv('PET_HEM_input.csv', index=False, header=True)
HEM_daily_ET=df

#######################################################################################################



df=HEM_daily_ET

#######################################################################################################
df=HEM_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
###########################################################################################################
back4=df
df.isnull().values.any()



#df["CO2_1_1_1"].fillna(method='backfill', inplace=True)
df.isnull().values.any()
df.to_csv('daily_HEM_ML.csv', index=False, header=True)

plt.plot(df["ET_f"])

#plt.plot(df["GPP_g"])

# pick columns that you want for ML model
















































########################################################################################################
fill_hem=pd.read_csv("filled_HEM.csv") 
back_up_HEM=pd.read_csv("back_up_HEM.csv") 

df=back_up_HEM

df['NEE_f']=fill_hem['NEE']
df['LE_f']=fill_hem['LE']
df['H_f']=fill_hem['H']
df['Rg_f']=fill_hem['Rg']
df['Tair_f']=fill_hem['Tair']
df['VPD_f']=fill_hem['VPD']

plt.plot(df['LE'],color='green')
plt.plot(df['LE_f'],color='red')

df['lambda_f']=(3149000-2370*(df['Tair_f']+273.16))*1e-6

plt.plot(df['lambda'],color='green')
plt.plot(df['lambda_f'],color='red')

df['ET_f']=(df['LE_f']/df['lambda_f'])*(1/1e6)*1800

plt.plot(df['ET_f'],color='red')

back=df


df=pd.concat((df["TIMESTAMP"],df["LE_f"],df['ET_f'],df['Tair_f'],df['VPD_f']),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['Year'] = df['TIMESTAMP'].dt.year
df.isnull().values.any()
df1 = df[df.isna().any(axis=1)]
#df=df.dropna()
df["Hr"]=df['TIMESTAMP'].dt.hour
HEM_hour_ET=df

###################################################################################################
df=HEM_hour_ET
df["Rnet1"] = np.where(df["NETRAD"]<-500, np.NaN, df["NETRAD"])
df["Rnet2"] = np.where(df["Rnet1"]>1000, np.NaN, df["Rnet1"])
df["Rnet"]=df["Rnet2"]
mask = (df['TIMESTAMP'] >='1992-06-08 00:00:00') & (df['TIMESTAMP'] <= '2022-12-31 23:45:00')
# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))


# prepare data for PET

df=pd.concat((s["TIMESTAMP"],s_et["LE_f"],s_et['ET_f'],s['Rnet'],s['RH'],s['WS'],s['PA'],s['Tair_f'],s['VPD_f']),axis=1)

df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
HEM_daily_ET=df
df.isnull().values.any()

df.to_csv('PET_HEM_input.csv', index=False, header=True)

###########################################################################################################
df=back

df=pd.concat((df["TIMESTAMP"],df["LE_f"],df['ET_f']),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['Year'] = df['TIMESTAMP'].dt.year
df=df.dropna()
df["Hr"]=df['TIMESTAMP'].dt.hour
df['date']=df['TIMESTAMP'].dt.date
back_HEM_half_hour=df

df=back_HEM_half_hour


df=df.groupby(['Hr', 'date']).sum()
HEM_hour_ET=df

df=back_HEM_half_hour
s=df.groupby([df['TIMESTAMP'].dt.date]).sum()
plt.plot(s['ET_f'],color='red')

df=s
HEM_half_hour_ET=df
df=df.reset_index()
df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
HEM_daily_ET=df
df.isnull().values.any()

df=HEM_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df=df.resample('m', on='TIMESTAMP').sum()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
df['Year'] = df['TIMESTAMP'].dt.month
HEM_total_month_ET=df

###############################################################################################################



df=HEM_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['water_year'] = df.TIMESTAMP.dt.year.where(df.TIMESTAMP.dt.month < 10, df.TIMESTAMP.dt.year + 1)
df= df.groupby(['water_year']).sum()
df=df.reset_index()
############################################################################################################
mask = (df['water_year'] >= 2006) & (df['water_year'] <= 2022)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ET=df
##############################################################################################################
df["ET"]=df["ET_f"]

annual_ET=df

x1=df['water_year']
y1=df['ET'].astype(float)
pearsonr(x1, y1)

mk.original_test(y1)

std=df['ET'].std()


plt.rcParams['figure.figsize'] = (7, 3.3)
fig, ax = plt.subplots()
plt.plot(x1, y1, color='purple', ls='-',marker='o',markersize=5, label='Pearson Correlation (r)=0.34' )
plt.fill_between(x1, y1-y1.std(),y1+y1.std(), color='lightgrey')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='purple',ls='--')
ax.set_ylim(200,800)
ax.set_title("Harvard Forest Hemlock site: US-Ha2, Mann Kendall test: p=0.2",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################################

# split timeseries
df=annual_ET
mask = (df['water_year'] >= 2006) & (df['water_year'] <= 2012)
df = df.loc[mask]
df.index = np.arange(0, len(df))
annual_ET1=df

df=annual_ET
mask = (df['water_year'] >= 2013) & (df['water_year'] <= 2022)
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
plt.plot(x1, y1, color='purple', ls='-',marker='o',markersize=5, label='Pearson Correlation (r)=-0.50' )
plt.fill_between(x1, y1-y1.std(),y1+y1.std(), color='lightgrey')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='purple',ls='--')
ax.set_ylim(200,800)
ax.set_title("US-Ha2, Mann Kendall test: p=0.2",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
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
plt.plot(x1, y1, color='purple', ls='-',marker='o',markersize=5, label='Pearson Correlation (r)=0.55' )
plt.fill_between(x1, y1-y1.std(),y1+y1.std(), color='lightgrey')
ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('Total Annual ET (mm)')  # we already handled the x-label with ax1
ax.set_xlabel('Water Year')  # we already handled the x-label with ax1
z1 = np.polyfit(x1, y1, 1)
p1 = np.poly1d(z1)
plt.plot(x1, p1(x1),color='purple',ls='--')
ax.set_ylim(200,800)
ax.set_title("US-Ha2, Mann Kendall test: p=0.37",fontsize=9)
#ax.set_title("Whole year",fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate() 
plt.ticklabel_format(useOffset=False)
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()
#########################################################################################################

# comapre both
#hem_ET=annual_ET

#ems_ET=annual_ET

df=hem_ET
df=ems_ET

plt.plot(hem_ET)






















df=df.resample('m', on='TIMESTAMP').sum()
df=df.reset_index()


df=df.resample('Y', on='TIMESTAMP').sum()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
HEM_total_annual_ET=df

df=HEM_daily_ET
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
HEM_mean_annual_ET=df

########################################################################################################

df=back_HEM_half_hour
df['Year'] = df["TIMESTAMP"].dt.year

mask = (df['Hr'] >=7) & (df['Hr'] <= 17)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df=df.groupby(['Hr', 'date']).sum()
df=df.reset_index()
df['date'] = pd.DatetimeIndex(df.date)
df['Year'] = df.date.dt.year



df=HEM_mean_annual_ET

df=df.iloc[1: , :]
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
df=HEM_total_month_ET
df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
df['month'] = pd.to_datetime(df["TIMESTAMP"]).dt.month


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


seasonal_HEM=df

df=seasonal_HEM
mask = (df['season_nam'] =='winter')
df = df.loc[mask]
df.index = np.arange(0, len(df))
winter_HEM=df


df=seasonal_HEM
mask = (df['season_nam'] =='spring')
df = df.loc[mask]
df.index = np.arange(0, len(df))
spring_HEM=df

df=seasonal_HEM
mask = (df['season_nam'] =='summer')
df = df.loc[mask]
df.index = np.arange(0, len(df))
summer_HEM=df

df=seasonal_HEM
mask = (df['season_nam'] =='fall')
df = df.loc[mask]
df.index = np.arange(0, len(df))
fall_HEM=df


df=summer_HEM
x=df['Year']
y=df['ET_f'].astype(float)
pearsonr(x, y)


df=EMS_total_month_ET
mask = (df['TIMESTAMP'] >= '2006-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["EMS_ET"]=df["ET_f"]
df1=df


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
#ax.set_title("winter total ET: corr=0.20",fontsize=15)
#ax.set_title("spring total ET: corr=0.05",fontsize=15)
#ax.set_title("summer total ET: corr=0.29",fontsize=15)
ax.set_title("Fall total ET: corr=0.1",fontsize=15)
#ax.set_title("total monthly ET: corr=0.07",fontsize=15)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

#########################################################################################


# interannual variability 

df=HEM_total_annual_ET
mask = (df['TIMESTAMP'] >= '2005-01-01') & (df['TIMESTAMP'] <= '2012-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_1=df

df=HEM_total_annual_ET
mask = (df['TIMESTAMP'] >= '2013-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df_2=df

df=df_2

#df=df.iloc[1: , :]
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
#ax.set_title(" total annual ET (1992-2008): corr=-0.6",fontsize=12)
#ax.set_title(" total annual ET (2005-2012): corr=-0.5",fontsize=12)
ax.set_title(" total annual ET (2013-2022): corr=0.57",fontsize=12)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

##############################################################################################################

# compare both sites

ems=EMS_total_month_ET
hem=HEM_total_month_ET


ems.to_csv('HEM_total_month_ET.csv', index=False, header=True)

hem.to_csv('EMS_total_month_ET.csv', index=False, header=True)

df=EMS_total_month_ET
mask = (df['TIMESTAMP'] >= '2005-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["EMS_ET"]=df["ET_f"]
df1=df


df=HEM_total_month_ET
mask = (df['TIMESTAMP'] >= '2005-01-01') & (df['TIMESTAMP'] <= '2022-12-31')
df = df.loc[mask]
df.index = np.arange(0, len(df))
df["HEM_ET"]=df["ET_f"]
df2=df



df=pd.concat((df1["TIMESTAMP"],df1["Year"],df1["month"],df1["EMS_ET"],df2['HEM_ET']),axis=1)
df.to_csv('combine_total_month_ET.csv', index=False, header=True)


os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")
# data is hourly
df=pd.read_csv("combine_total_month_ET.csv")  # 2005-2022, # of days 11688

combine_month=df











x=df['Year']
y1=df['EMS_ET'].astype(float)
y2=df['HEM_ET'].astype(float)
pearsonr(x, y1)
pearsonr(x, y2)


plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df['Year'], df['EMS_ET'].astype(float),color="green", linestyle='--',label="EMS_ET (0.1)", marker='o')
plt.plot(df['Year'], df['HEM_ET'].astype(float),color="red", linestyle='--',label="HEM_ET (0.1)", marker='o')


ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
ax.set_ylabel('ET')  # we already handled the x-label with ax1
z = np.polyfit(x, y1, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x),color="green")

z = np.polyfit(x, y2, 1)
#slope is z[0]
p = np.poly1d(z)
plt.plot(x, p(x),color="red")

ax.set_title(" total monthly ET (2005-2022)",fontsize=12)
ax.set_xlabel('Year')  # we already handled the x-label with ax1
ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

back_up_EMS.to_csv('back_up_EMS.csv', index=False, header=True)









###################################################################################################
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



























































os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")
# data is hourly
df=pd.read_csv("EMS_ustarfiltered_thru2022.csv")  # 1991-2022, # of days 11688

df.head(10)
df.tail(10)
back=df
df=back
plt.plot(df["LE_1_1_2"])
plt.plot(df["LE_1_1_1"])
plt.plot(df["LE"])
plt.plot(df["ET"])

df=pd.concat((df["TIMESTAMP"],df["ET"]),axis=1)

et=df

df.isnull().values.any()

#########################################################################################################

os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")
# data is hourly
df=pd.read_csv("EMS_ustarfiltered_thru2022.csv")  # 1991-2022, # of days 11688
back=df
df.head(10)
df.tail(10)
idx=pd.date_range(start='01/01/1991 00:00:00', periods=280512, freq='60T')  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
back=df


#index is correct date.TIMESTAMP date mess up for missing values
# so let remove messed up timestamp column
df=df.iloc[:,1:111]
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
s=pd.Series(np.arange(1,24.5,1))
#######################################################################################################
#change this based on data size (number of days of data), 
#([s]*(#number of days)
k=pd.concat([s]*(11688),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df.insert(3,'Hour',k)
df=df.iloc[:,1:114]
df.tail(10)
#########################################################################################################

df["rH"]=df["RH_PI_F_1_1_1"]
df['Tsoil']=np.nan


cols = df.columns.tolist()

df["DateTime"]=df["date"]
df["TIMESTAMP"]=df["date"]

# end date will be one day more than your desired end day
back_up_EMS=df
back_up_EMS.to_csv('back_up_EMS.csv', index=False, header=True)


##################################################################################################################
reframed=pd.concat((df["DateTime"],df[cols[0]],df[cols[1]],df[cols[2]],df["NEE"],df["LE"],df["H"],df['Rg'],df["Tair"],df["VPD"],df["Ustar"]),axis=1)
reframed.to_csv('gaps_EMS.csv', index=False, header=True)
df=back_up_EMS

# brig filled data
os.chdir(r"C:\ammara_MD\a_harvard\Project\data\Evapotranspiration")

fill_ems=pd.read_csv("filled_EMS.csv") 
back_up_EMS=pd.read_csv("back_up_EMS.csv") 

df=back_up_EMS

df['NEE_f']=fill_ems['NEE']
df['LE_f']=fill_ems['LE']
df['H_f']=fill_ems['H']
df['Rg_f']=fill_ems['Rg']
df['Tair_f']=fill_ems['Tair']
df['VPD_f']=fill_ems['VPD']

plt.plot(df['LE'],color='green')
plt.plot(df['LE_f'],color='red')

df['lambda_f']=(3149000-2370*(df['Tair_f']+273.16))*1e-6

plt.plot(df['lambda'],color='green')
plt.plot(df['lambda_f'],color='red')

df['ET_f']=(df['LE_f']/df['lambda_f'])*(1/1e6)*3600

plt.plot(df['ET_f'],color='red')


df=pd.concat((df["TIMESTAMP"],df["LE_f"],df['ET_f']),axis=1)

df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df['Year'] = df['TIMESTAMP'].dt.year
df=df.dropna()
df["Hr"]=df['TIMESTAMP'].dt.hour
EMS_hour_ET=df


s=df.groupby([df['TIMESTAMP'].dt.date]).sum()
plt.plot(s['ET_f'],color='red')

df=s
df=df.reset_index()
df['Year'] = pd.to_datetime(df["TIMESTAMP"]).dt.year
EMS_daily_ET=df
df.isnull().values.any()

df=EMS_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df=df.resample('m', on='TIMESTAMP').sum()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
df['Year'] = df['TIMESTAMP'].dt.month
EMS_total_month_ET=df

df=EMS_daily_ET
df["TIMESTAMP"]=pd.to_datetime(df["TIMESTAMP"])
df=df.resample('Y', on='TIMESTAMP').sum()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
EMS_total_annual_ET=df

df=EMS_daily_ET
df=df.resample('Y', on='TIMESTAMP').mean()
df=df.reset_index()
df['Year'] = df['TIMESTAMP'].dt.year
EMS_mean_annual_ET=df



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


df=EMS_total_annual_ET


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

###################################################################################################
# interannual variability 

df=EMS_total_annual_ET
mask = (df['TIMESTAMP'] >= '1992-01-01') & (df['TIMESTAMP'] <= '2008-12-31')
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
#ax.set_title(" total annual ET (1992-2008): corr=-0.6",fontsize=12)
#ax.set_title(" total annual ET (2009-2022): corr=-0.05",fontsize=12)
ax.set_title(" total annual ET (1992-2022): corr=0.2",fontsize=12)
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
         
#ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1
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







##########################################################################################
# daily
########################################################################################333
df1=pd.read_csv("hf070-02-manual.csv") # 2005-2007
df2=pd.read_csv("hf070-02-manual.csv",skiprows=[2])
df=df2
df.rename(columns=df.iloc[0], inplace = True)
df.drop([0], inplace = True)
back=df


df["TIMESTAMP"]= pd.to_datetime(df["date"])
df.set_index(('TIMESTAMP'), inplace=True)




y1=df["nb.stg"] .astype(float)

df['nl.stg']=pd.to_numeric(df['nl.stg'], errors="ignore")


df['nl.stg'] = df['nl.stg'].astype('float')

df['nl.stg'].replace({'NA': np.nan}, inplace = True)
y2=df['nl.stg']



plt.rcParams['figure.figsize'] = (4, 3.3)
fig, ax = plt.subplots()
plt.plot(df.index, y1, color="green", linestyle='--',label="nelson Big weir",mfc='none')
plt.plot(df.index, y2, color="red", linestyle='--',label="nelson little weir",mfc='none')

ax.tick_params(axis='y', labelcolor='black')
ax.legend()
ax.legend(loc='upper left', fontsize = '9')
 # 
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(9) 
#ax.set_ylim(max(df["nb.stg"], key=float),min(df["nb.stg"], key=float),30)
         
ax.set_ylabel('stage (cm)')  # we already handled the x-label with ax1

ax.set_xlabel('Year-Month')  # we already handled the x-label with ax1

ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
ax.tick_params(axis = 'y', which = 'major', labelsize =8)

#ax.set_ylim(min(y1),max(y1),30)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#ax.locator_params(axis='y', nbins=100)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig.jpg',dpi=600,bbox_inches="tight")
plt.tight_layout()

















df.head(10)
df.tail(10)

idx=pd.date_range(start='06/28/2018 00:00:00', periods=96624, freq='30T')  # half hourly. 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()

# do not apply mask yet. Do it later 

back=df
#index is correct date.TIMESTAMP date mess up for missing values
# so let remove messed up timestamp column
df=df.iloc[:,1:38]
#df=df.astype(float) 
df.head(10)
df.tail(10)
temp=df
df["date"] = pd.to_datetime(df.TIMESTAMP)

df["LE"]=df.LE_1_1_1+df.SLE_1_1_1
df["H"]=df.H_1_1_1+df.SH_1_1_1
df=df.rename_axis('TIMESTAMP').reset_index()
df["date"] = pd.to_datetime(df.TIMESTAMP)

df["Rg"]=df['SW_IN_1_1_1']

#calculate vapor pressure deficiet
#prepare some variables for eddy proc for gap filling
df["Tair"]= df["TA_1_1_1"] 
df["rH"]=df["RH_1_1_1"]

es=0.6108*np.exp((17.27*df["Tair"])/(df["Tair"]+237.3))
#ea is actual vapoure pressure in millibar and es is saturated vapor pressure 

ea=df["rH"]/100*es
VPD=(es-ea)*10
df['VPD']=VPD   # in kpa
df["AVP"]= ea # in kpa
df.head(10)
df.tail(10)
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
k=pd.concat([s]*(2013),axis=0)
k.reset_index(drop=True, inplace=True)
#k.index = np.arange(1, len(k)+1)
df.insert(3,'Hour',k)
df=df.iloc[:,1:103]
df.tail(10)
#########################################################################################################

df["LE1"] = np.where(df["LE"]<-200, np.NaN, df["LE"])
df["LE2"] = np.where(df["LE1"]>800, np.NaN, df["LE1"])
df["LE"]=df["LE2"]


df["H1"] = np.where(df["H"]<-200, np.NaN, df["H"])
df["H2"] = np.where(df["H1"]>800, np.NaN, df["H1"])
df["H"]=df["H2"]


df["NEE1"] = np.where(df["NEE_F"]<-50, np.NaN, df["NEE_F"])
df["NEE2"] = np.where(df["NEE1"]>50, np.NaN, df["NEE1"])
df["NEE"]=df["NEE2"]
df["FC"]=df["NEE"]
df["Rnet1"] = np.where(df["NETRAD_1_1_1"]<-500, np.NaN, df["NETRAD_1_1_1"])
df["Rnet2"] = np.where(df["Rnet1"]>1000, np.NaN, df["Rnet1"])
df["Rnet"]=df["Rnet2"]

df["Rg1"] = np.where(df["Rg"]<-50, np.NaN, df["Rg"])
df["Rg2"] = np.where(df["Rg1"]>1200, np.NaN, df["Rg1"])
df["Rg"]=df["Rg2"]

df["Ustar1"] = np.where(df["USTAR_1_1_1"]<0, np.NaN, df["USTAR_1_1_1"])
df["Ustar2"] = np.where(df["Ustar1"]>3, np.NaN, df["Ustar1"])
df["Ustar"]=df["Ustar2"]

df["Tair1"] = np.where(df["Tair"]<-200, np.NaN, df["Tair"])
df["Tair2"] = np.where(df["Tair1"]>200, np.NaN, df["Tair1"])
df["Tair"]=df["Tair2"]


df["rH1"] = np.where(df["rH"]<0, np.NaN, df["rH"])
df["rH2"] = np.where(df["rH1"]>100, np.NaN, df["rH1"])
df["rH"]=df["rH2"]

df["VPD1"] = np.where(df["VPD"]<0, np.NaN, df["VPD"])
df["VPD2"] = np.where(df["VPD1"]>100, np.NaN, df["VPD1"])
df["VPD"]=df["VPD2"]

df["DateTime"]=df["date"]
df["TIMESTAMP"]=df["date"]

df['Tsoil']=np.nan


cols = df.columns.tolist()
back=df

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')


# end date will be one day more than your desired end day
df = df.loc[mask]
df.index = np.arange(0, len(df))

back_up=df

##################################################################################################################
reframed=pd.concat((df["DateTime"],df[cols[0]],df[cols[1]],df[cols[2]],df["NEE"],df["LE"],df["H"],df['Rg'],df["Tair"],df['Tsoil'],df["rH"],df["VPD"],df["Ustar"]),axis=1)
reframed.to_csv('gaps.csv', index=False, header=True)

##############################################################################################################

potat_back=back_up

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


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

#wis_pot=pd.read_csv("corrected.csv") 

fill=pd.read_csv("filled.csv") 
# Rnet is coming from df from top 
fill["pot_Rnet"]=potat_back["Rnet"]
fill["AVP_pot"]=potat_back["AVP"]
fill["rH_pot"]=potat_back["rH"]
fill["VPD_pot"]=potat_back["VPD"]
fill["Rg"]=potat_back["Rg"]
fill["Tair"]=potat_back["Tair"]
fill["SW_in"]=potat_back["SW_IN_1_1_1"]
fill["SW_out"]=potat_back["SW_OUT_1_1_1"]
fill["LW_in"]=potat_back["LW_IN_1_1_1"]
fill["LW_out"]=potat_back["LW_OUT_1_1_1"]


fill["TIMESTAMP"]=potat_back["DateTime"]

df=fill

df.head(10)
df.tail(10)

#### good

df["ET"]=df["LE"]*((1/1000)*(1/(2.5*1000000))*(86400*1000))
df["pot_ET_inches"]=df["ET"]*0.0393701  # convert into inches for farmers
df.isnull().values.any()



mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2023-08-31 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

fill_pot=df

###################################################################################################

### this part is for emissivity

# time zone change 6 hour different. this part is only for emissivity

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_csv("potsolar_2022.csv") 
df.index = pd.DatetimeIndex(df.TIMESTAMP)
idx = pd.date_range(start='2018/01/01', periods=87648, freq='30T')  # days*24*2

df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
#df = df.iloc[:, :-1]
df=df.rename_axis('TIMESTAMP').reset_index()
df.index = pd.DatetimeIndex(df.TIMESTAMP)

d = pd.to_datetime(df["TIMESTAMP"])
mydatetime = d # or whatever value you want
local = mydatetime - datetime.timedelta(hours=6)
df["local"]=local 
df["TIMESTAMP1"]=local 

df=df.drop('TIMESTAMP', axis=1)  
df=df.reset_index()
df["TIMESTAMP"]=df["TIMESTAMP1"]
df=df.drop('TIMESTAMP1', axis=1)  

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

back_hour=df

mask = (fill_pot['TIMESTAMP'] >'2018-06-29 23:45:00') & (fill_pot['TIMESTAMP'] <= '2022-09-30 23:45:00')

df = fill_pot.loc[mask]
df.index = np.arange(0, len(df))

df["pot_solar"]=back_hour["Solar_W_m-2"]

plt.plot(df["pot_solar"],'g-')
plt.plot(df["SW_in"],'r')

back=df

df["Hr"]=back["TIMESTAMP"].dt.hour

df.isnull().values.any()
df["sol_diff"]=df["pot_solar"]-df["SW_in"]
df=pd.concat((back_hour["TIMESTAMP"],df['Hr'],df),axis=1)

mask = (df['sol_diff'] >=0)
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
plt.plot(df["pot_solar"],'g-')
plt.plot(df["SW_in"],'r')
#df=pd.concat((fill_pot["TIMESTAMP"],fill_pot['Hr'],df),axis=1)

df=df.drop(['Hr'], axis=1)
df["Hr"]=back["TIMESTAMP"].dt.hour
back=df

#### good so far
mask = (df['Hr'] >=10) & (df['Hr'] <= 14)
df = df.loc[mask]
df.index = np.arange(0, len(df))


mask = (df['SW_in'] >=(.9*df['pot_solar']))
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))


plt.plot(df["pot_solar"],'g-')
plt.plot(df["SW_in"],'r')
back=df

#df["es_obs"]=0.96*(df["LW_in"]/df["LW_out"])

df.to_csv('emiss_hour.csv', index=False, header=True)

## remove one of the timestamp by hand

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_csv("emiss_hour.csv")

# hourly emissivity

plt.plot(df["es_obs"])

hourly_emmis=df

df["TIMESTAMP"]= pd.to_datetime(df['TIMESTAMP'])

df=df.resample('D', on='TIMESTAMP').mean()
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()

# daily emissivty 
plt.plot(df["es_obs"])

df["es_obs"]= np.where(df["es_obs"]>1,0.999,df["es_obs"])
emiss=df

plt.plot(df["es_obs"])
emiss.to_csv('emiss.csv', index=False, header=True)

### emissivity part ends
##########################################################################################################

# start here 1/9/2023

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="new_canopy",header=0)
#df=df.resample('D', on='TIMESTAMP').sum()
#df=df.reset_index()
idx=pd.date_range(start='01/01/2018', periods= 1765, freq='D')  ## days
df.index = pd.DatetimeIndex(df.TIMESTAMP)
#df["can_cov"]=df.mean(axis=1)
back_canop=df
#df=df.interpolate()


df=df.iloc[:,1:99]  # don't worry about end column. not so important
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
#df= df.replace(-9999, np.nan) 
df.isnull().values.any()
#df.resample('D', on='TIMESTAMP').mean()
#df=df.iloc[:,2:99]  # don't worry about end column. not so important

df=df.interpolate()

#df=df.fillna(df.rolling(2,1).mean())
#df=df.fillna(df.rolling(3,1).mean())
#df=df.fillna(df.rolling(9,1).mean())

df.isnull().values.any()
df["canop_cover"]=df["canop"]*100
canop=df

df["can_cov"]=df["canop_cover"]

#df=df.fillna(df.rolling(15,1).mean())
#df=df.fillna(df.rolling(20,1).mean())
#df.isnull().values.any()
#df=df.fillna(df.rolling(30,1).mean())
#df.isnull().values.any()
df=df.rename_axis('TIMESTAMP').reset_index()
canop=df
df=canop
canop=df

df=canop
#mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
#df.loc[mask]
#df = df.loc[mask]
#df.index = np.arange(0, len(df))
canop=df


## no need to apply mask here. Mask is applied next 
#import matplotlib.pyplot  as pyplot
plt.plot(canop["canop_cover"])

########################################################################################################33

### wisp ET
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="wisp_ET",header=0)
# no of days
idx=pd.date_range(start='01/01/2018', periods=1765, freq='D')

df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
df=df.rename_axis('TIMESTAMP').reset_index()
df.isnull().values.any()

df['ET_inches']=df['ET_inches'].astype(object).astype(float)

df=df.resample('D', on='TIMESTAMP').mean()
df=df.reset_index()

wisp=df
wisp["cum_ET"]=wisp["ET_inches"].cumsum()
wisp["canop"]=canop["can_cov"]

#wisp["ET_inches_1"]= np.where(wisp["ET_inches"]<0,0.00001,wisp["ET_inches"])
#wisp["ET_inches"]=wisp["ET_inches_1"]

def adj_AET (wisp):
    if wisp['canop']>=80:
        return wisp['ET_inches']
    else:
        return wisp['ET_inches']*((wisp['canop']/80+0.0833))

wisp["corr"]= wisp.apply(adj_AET, axis = 1)
wisp["wisp_ET_adj"]=wisp["corr"]
wisp["wisp_PET"]=wisp['ET_inches']
wisp["wisp_ETcum"]=wisp["corr"].cumsum()
plt.plot(wisp["ET_inches"])
plt.plot(wisp["wisp_ETcum"],'r')
df=wisp
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
wisp=df

plt.plot(df.wisp_PET,'r--', label="WISP PET")# daily 
#plt.legend()
#plt.show()


plt.plot(df.wisp_ET_adj,'g--', label="WISP AET")# daily 

#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()

# so far good 
#########################################################################################################

### add a precip inches 

#download hancock precip
#https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USC00473405/detail

#noaa standard unit inches and metric units mm
# use mm metric

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="precip",header=0)
df["TIMESTAMP"] = pd.to_datetime(df.TIMESTAMP)
idx=pd.date_range(start='01/01/2018', periods=1734, freq='D')
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
df.isnull().values.any()

df=df.interpolate()


#df=df.fillna(df.rolling(2,1).mean())
#df=df.fillna(df.rolling(3,1).mean())
#df=df.fillna(df.rolling(4,1).mean())

df.isnull().values.any()
df=df.rename_axis('TIMESTAMP').reset_index()

df=df.resample('D', on='TIMESTAMP').sum()
df=df.reset_index()

df["prec_inch"]=df["PRCP_mm"]*0.0393701
prec=df
prec["prec_cum"]=prec["prec_inch"].cumsum()
df=prec
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
prec=df

###########################################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="irrig",header=0)
df=df.resample('D', on='TIMESTAMP').sum()
df=df.reset_index()

idx=pd.date_range(start='01/01/2018', periods= 1734, freq='D')  ## days
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.iloc[:,1:99]  # don't worry about end column. not so important

df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, 0, regex=True) 
df=df.rename_axis('TIMESTAMP').reset_index()
df.index = pd.DatetimeIndex(df.TIMESTAMP)
d = pd.to_datetime(df["TIMESTAMP"])
mydatetime = d # or whatever value you want
df=df.iloc[:,1:99]  # don't worry about end column. not so important
df=df.rename_axis('TIMESTAMP').reset_index()

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
irrig=df
###################################################################################### FC is NEE
#df=daily potato data

df=fill_pot
df=df.resample('D', on='TIMESTAMP').mean()

df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
fill_pote=df

df=fill_pot

#df=df.rename_axis('TIMESTAMP').reset_index()
#df=df.resample('D', on='TIMESTAMP').mean()
#df=df.reset_index()

mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
#fill_pot=df


plt.plot(wisp.wisp_PET,'r--', label="WISP PET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
plt.legend()
plt.show()


plt.plot(wisp.wisp_ET_adj,'g--', label="WISP AET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()


#df["pot_ET_inches"]= np.where(df["pot_ET_inches"]<0,0.00001,df["pot_ET_inches"])
#df["pot_ET_EBC_inches"]= np.where(df["pot_ET_EBC_inches"]<0,0.00001,df["pot_ET_EBC_inches"])

##################################################################################################
###bring inso data
#
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_excel ("wisp.xlsx", sheet_name="inso",header=0)
idx=pd.date_range(start='01/01/2018', periods=1765, freq='D')
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df = df.reindex(idx, fill_value=-9999) #add missing dates and add  nan for missing values
df= df.replace(-9999, np.nan, regex=True) 
df=df.iloc[:,1:99]  # don't worry about end column. not so important
#df=df.rename_axis('TIMESTAMP').reset_index()
#df=df.set_index(['TIMESTAMP'])
df.isnull().values.any()
df=df.rename_axis('TIMESTAMP').reset_index()

#df=df.resample('D', on='TIMESTAMP').mean()
#df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
#this line will only work if index is timestamp
inso=df
inso["inso_rad"]=inso["inso_rad_MJ"]

###############################################################################################

df=fill_pot
df=df.resample('D', on='TIMESTAMP').mean()
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()
mask = (df['TIMESTAMP'] >'2018-06-29 23:45:00') & (df['TIMESTAMP'] <= '2022-09-30 23:45:00')
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))
fill_pota=df
####################################################################################################

# next without pine

data=pd.concat((fill_pota,inso["inso_rad"],inso["inso_vp_kpa"].astype(float),inso["inso_temp_c"].astype(float),wisp["canop"],prec["prec_inch"],wisp["wisp_PET"],wisp["wisp_ET_adj"],irrig["irrig_inches"]),axis=1)
df=data

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season.csv', index=False, header=True)
data=pd.read_csv('growing_season.csv')
df=data

plt.plot(wisp.wisp_PET,'r--', label="WISP PET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
plt.legend()
plt.show()

plt.plot(wisp.wisp_ET_adj,'g--', label="WISP AET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()
####################################################################################################
#### bring pine part

#######################################################3

data=pd.concat((fill_pota,fill_pine,inso["inso_rad"],inso["inso_vp_kpa"],inso["inso_temp_c"],wisp["canop"],prec["prec_inch"],irrig['irrig_inches'],wisp["wisp_PET"],wisp["wisp_ET_adj"]),axis=1)

#use this
#data=pd.concat((fill_pota,inso["inso_rad"],inso["inso_vp_kpa"],inso["inso_temp_c"],wisp["canop"],prec["prec_inch"],wisp["wisp_PET"],wisp["wisp_ET_adj"]),axis=1)

###make sure to check wisp temp and vapor pressure column is not greyed out 
# dont round off, it will effect wisp lower ET values and will make them zero 
#data=data.round(2)


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data.to_csv('combine_pot_pine.csv', index=False, header=True)

# manually remove one timestamp


os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data=pd.read_csv("combine_pot_pine.csv") 


data_back=data
df=data_back

df=df.iloc[:,3:99] 



df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(3,1).mean())
#df=df.fillna(df.rolling(4,1).mean())

df.isnull().values.any()

df["TIMESTAMP"]=data_back["TIMESTAMP"]
df["Year"]=data_back["Year"]

data=df

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data.to_csv('combine_pot_pin_fin.csv', index=False, header=True)


#################################################################################################
back=data
#########################################################################################
#### stop read line below
#####################################################################################
# manualy remove duplicate columns timestmp colum from data
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")

 #df= pd.concat((fill_pote,fill_pine),axis=1)

df=pd.read_csv("combine_pot_pin_fin.csv") 


df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=4) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season.csv', index=False, header=True)
data=pd.read_csv('growing_season.csv')
df=data

plt.plot(wisp.wisp_PET,'r--', label="WISP PET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
plt.legend()
plt.show()

plt.plot(wisp.wisp_ET_adj,'g--', label="WISP AET")# daily 
plt.plot(fill_pote["pot_ET_inches"],'b--', label="Tower AET" )
#plt.plot(canop["canopy"]/100,'b-', label="Daily canopy") # weekly
plt.legend()
plt.show()

#####################################################################################
# fix missing data: upto 4 days

os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
df=pd.read_csv("growing_season.csv") 
back=df
df=df.iloc[:,2:99]  # don't worry about end column. not so important
#df=df.rename_axis('TIMESTAMP').reset_index()
df=df.fillna(df.rolling(2,1).mean())
df=df.fillna(df.rolling(3,1).mean())
df=df.fillna(df.rolling(4,1).mean())
df.isnull().values.any()
df["TIMESTAMP"]=back["TIMESTAMP"]
data=df
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
data.to_csv('data.csv', index=False, header=True)

##############################################################################################
# save growinf season data after gap filling
os.chdir(r"C:\ammara_MD\flux_tower_data\Tower_WISP\correct_flux\all_data_2021\potatoes_2021")
 #df= pd.concat((fill_pote,fill_pine),axis=1)

df=pd.read_csv("data.csv") 

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=4) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))
growin=df
growin.to_csv('growing_season.csv', index=False, header=True)



























#############################################################################################
df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year

s=df.groupby(['month','year']).sum()

df=s
df["prec+irri"]=df["prec_inch"]+df["irrig_inches"]
df=df.reset_index()
df = df.assign(TIMESTAMP=pd.to_datetime(df[['year', 'month']].assign(day=1)))
import numpy as np
import matplotlib.pyplot as plt



df=df.sort_values(by='TIMESTAMP')
df = df.iloc[1: , :]
plt.rcParams['figure.figsize'] = (16, 8)
fig, ax = plt.subplots()
ax.set_ylabel('Monthly water (inches)', color='black')
N = len(df["TIMESTAMP"])
ind = np.arange(N)  # the x locations for the groups
width = 0.1     # the width of the bars
yvals = df["pin_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["pot_ET_inches"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')
kvals = df["prec_inch"]
rects3 = ax.bar(ind+width+0.15, kvals, width, color='purple',edgecolor='purple',linewidth=1)
vvals = df["irrig_inches"]
vrects3 = ax.bar(ind+width+0.15+0.15, vvals, width, color='black',edgecolor='black',linewidth=1)

ax.set_xticks(ind+width+0.27)
#ax.set_xticklabels( ('Jun2018','July2018','Aug2018','Sep2018','May2019','Jun2019','July2019','Aug2019','Sep2019','May2020','Jun2020','July2020','Aug2020','Sep2020','May2021','Jun2021','July2021','Aug2021','Sep2021','May2022','Jun2022','July2022','Aug2022'))
ax.set_xticklabels( ('July2018','Aug2018','Sep2018','May2019','Jun2019','July2019','Aug2019','Sep2019','May2020','Jun2020','July2020','Aug2020','Sep2020','May2021','Jun2021','July2021','Aug2021','Sep2021'))
ax.set_ylim(0,8)

ax.legend( (rects1[0], rects2[0],rects3[0],vrects3[0]), ('Pines','Potatoes','Precip','Irrigation') ,loc='upper right', fontsize = 'large')

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################

os.chdir(r"C:\ammara_MD\flux_tower_data\potato")
data=pd.read_csv("data.csv")

df=data
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.drop(columns='TIMESTAMP')
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()


mask = (df['TIMESTAMP'] >'2021-04-01') & (df['TIMESTAMP'] <= '2021-10-31')

df = df.loc[mask]
df.index = np.arange(0, len(df))
df["TIMESTAMP"]= pd.to_datetime(df["TIMESTAMP"])
plt.rcParams['figure.figsize'] = (7, 6)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Daily (ET) inches', color='black')
ax1.set_xlabel('Year-month', color='black')
plt.plot(df["TIMESTAMP"],df["pot_ET_inches"],linestyle="-",label="Observed ET",color='green')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_canopy"],linestyle="--",label="WISP ET",color='red')
ax1.axvline(pd.to_datetime('2021-04-27'), color='orange', linestyle='--', lw=2,label="Crop Planted")
ax1.axvline(pd.to_datetime('2021-08-16'), color='pink', linestyle='--', lw=2)
#ax1.axvline(pd.to_datetime('2021-08-27'), color='pink', linestyle='--', lw=2,label="Vine Killed")

#ax1.axvline(pd.to_datetime('2020-06-03'), color='black', linestyle='--', lw=2,label="Full Emergence")
#ax1.axvline(pd.to_datetime('2020-06-23'), color='red', linestyle='--', lw=2,label="100% canopy")
ax1.axvline(pd.to_datetime('2021-09-15'), color='blue', linestyle='--', lw=2,label="Harvested")


ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()
ax1.legend(loc='upper left', fontsize = 'large')
ax1.yaxis.label.set_size(15)
ax1.xaxis.label.set_size(15) #there is no label 
ax1.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax1.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax1.set_ylim(0,0.5)
ax1.set_title("Potatoes Variety#2 2020 (AET)",fontsize=15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Irrigation or Precipitation(inches)', color='black')  # we already handled the x-label with ax1
ax2.scatter(df["TIMESTAMP"],df.irrig_inches, color='black',label="Irrigation")
ax2.scatter(df["TIMESTAMP"],df.prec_inch, color='purple', label="Precipitation")
ax2.legend()
ax2.legend(loc='upper right', fontsize = 'large')
ax2.set_ylim(0,2.0)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis = 'y', which = 'major', labelsize = 15)
ax2.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax2.yaxis.label.set_size(15)
ax2.xaxis.label.set_size(15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()


####################################################################################

##cumulative

df=data
#df["wisp_ET_adj"]= np.where(df["wisp_ET_adj"]<0,0.00001,df["wisp_ET_adj"])
#df["wisp_AETidso_corr"]= np.where(df["wisp_AET_idso_corr"]<0,0.00001,df["wisp_AET_idso_corr"])
df["pot_ET_inches"]= np.where(df["pot_ET_inches"]<0,0.00001,df["pot_ET_inches"])
df["pin_ET_inches"]= np.where(df["pin_ET_inches"]<0,0.00001,df["pin_ET_inches"])
df.index = pd.DatetimeIndex(df.TIMESTAMP)
df=df.drop(columns='TIMESTAMP')
df=df.rename_axis('TIMESTAMP').reset_index()
df=df.reset_index()


month= pd.to_datetime(df['TIMESTAMP']).dt.month
year= pd.to_datetime(df['TIMESTAMP']).dt.year
#mask = (df['TIMESTAMP'] >'2019-04-30 23:45:00') & (df['TIMESTAMP'] <= '2020-04-15')
mask = (df['TIMESTAMP'] >='2019-04-30 00:00:00') & (df['TIMESTAMP'] <= '2019-09-30 00:00:00')
mask = (df['TIMESTAMP'] >='2020-04-30 00:00:00') & (df['TIMESTAMP'] <= '2020-09-30 00:00:00')
mask = (df['TIMESTAMP'] >='2021-04-30 00:00:00') & (df['TIMESTAMP'] <= '2021-09-30 00:00:00')

df = df.loc[mask]
df.index = np.arange(0, len(df))

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 9)
df = df.loc[mask]
df.index = np.arange(0, len(df))

df_2019=df
df_2020=df
df_2021=df

df_2021["pot_ET_inches"]=df['pot_ET_inches'].fillna(0.1)
df_2021['TIMESTAMP']=pd.to_datetime(df['TIMESTAMP'])

plt.rcParams['figure.figsize'] = (12, 7)
fig, ax = plt.subplots()
df["TIMESTAMP"]=pd.to_datetime(df['TIMESTAMP'])
ax.set_ylabel('ET (inches)', color='black')
ax.set_xlabel('Month', color='black')
#plt.plot(df_2019["TIMESTAMP"],df_2019["pot_ET_inches"].cumsum(),linestyle="-",label="Potatoes year 2019",color='orange')
#plt.plot(df_2019["TIMESTAMP"],df_2019["pin_ET_inches"].cumsum(),linestyle="-",label="Pine year 2019",color='green')

#plt.plot(df_2020["TIMESTAMP"],df_2020["pot_ET_inches"].cumsum(),linestyle="-",label="Potatoes year 2020",color='orange')
#plt.plot(df_2020["TIMESTAMP"],df_2020["pin_ET_inches"].cumsum(),linestyle="-",label="Pine year 2020",color='green')

plt.plot(df_2021["TIMESTAMP"],df_2021["pot_ET_inches"].cumsum(),linestyle="-",label="Potatoes year 2021",color='orange')
plt.plot(df_2021["TIMESTAMP"],df_2021["pin_ET_inches"].cumsum(),linestyle="-",label="Pine year 2021",color='green')

#plt.plot(df["TIMESTAMP"],df["wisp_ET_adj"].cumsum(),linestyle="-",label="Wisp",color='skyblue')
#plt.plot(df["TIMESTAMP"],df["wisp_AET_idso_corr"].cumsum(),linestyle="-",label="Corrected Wisp",color='brown')

#ax.legend()
ax.set_xticklabels( ('May','June','July','Aug','Sep','Oct'))

ax.legend(loc='upper left',prop={'size': 15})
#ax2.legend(loc='upper left', fontsize = 'large',prop={'size': 6})
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

#lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 15})
ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 18)
ax.set_ylim(0,25)
ax.set_title("Cumulative Growing Season ET (inches)",fontsize=18)
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()
###############################################################################################

#####################################################################################
df=data

df["year"]= pd.to_datetime(df['TIMESTAMP']).dt.year

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 8)
df = df.loc[mask]
df.index = np.arange(0, len(df))
df=df


s=df.groupby(['year']).sum()

df=s.rename_axis('year').reset_index()
df=df.reset_index()


df=df.sort_values(by='year')
#df = df.iloc[1: , :]
plt.rcParams['figure.figsize'] = (16, 8)
fig, ax = plt.subplots()
ax.set_ylabel('Total May-Aug ET (inches)', color='black')
N = len(df["year"])
ind = np.arange(N)  # the x locations for the groups
width = 0.1     # the width of the bars
yvals = df["pot_ET_inches"]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = df["wisp_ET_adj"]
rects2 = ax.bar(ind+width, zvals, width, color='orange')
kvals = df["wisp_AETidso_corr"]
rects3 = ax.bar(ind+width+0.1, kvals, width, color='purple',edgecolor='purple',linewidth=1)
#vvals = df["irrig_inches"]
#vrects3 = ax.bar(ind+width+0.15+0.15, vvals, width, color='black',edgecolor='black',linewidth=1)

ax.set_xticks(ind+width+0.15)
#ax.set_xticklabels( ('Jun2018','July2018','Aug2018','Sep2018','May2019','Jun2019','July2019','Aug2019','Sep2019','May2020','Jun2020','July2020','Aug2020','Sep2020','May2021','Jun2021','July2021','Aug2021','Sep2021'))
ax.set_xticklabels( ('2018','2019','2020','2021'))
ax.set_ylim(0,20)

ax.legend( (rects1[0], rects2[0],rects3[0]), ('Tower Observed ET','WISP ET','WISP Corrected ET') ,loc='upper left', fontsize = 'large')

ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
ax.set_title("Water use Central Sands",fontsize=15)
fig.autofmt_xdate() 
plt.show()
fig.savefig('testfig1.png',dpi=600, bbox_inches = "tight")
plt.tight_layout()

#########################################################################################
